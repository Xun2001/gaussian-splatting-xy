import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def load_camera_centers(images_txt_path):
    """
    解析 COLMAP images.txt，计算每张图的相机光心。
    C = -R^T @ T，其中 R 由四元数 (qx, qy, qz, qw) 转换而来。
    """
    centers = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            elems = line.strip().split()
            if len(elems) < 10:
                continue
            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz    = map(float, elems[5:8])
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            centers.append((-rot.T @ np.array([tx, ty, tz])))
    return np.array(centers)

def main(args):
    # 1. 加载相机中心
    camera_centers = load_camera_centers(args.images_txt)

    # 2. 分割地面平面，获取法线
    pcd = o3d.io.read_point_cloud(args.pcd_path)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.02,  # 根据点云尺度调整
        ransac_n=3,
        num_iterations=1000
    )
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)

    # 3. 构造旋转矩阵，将地面法线对齐到 [0,0,1]
    target = np.array([0.0, 0.0, 1.0])
    axis   = np.cross(normal, target)
    axis  /= np.linalg.norm(axis)
    angle  = np.arccos(np.dot(normal, target))
    R_align = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # 4. 旋转点云与相机中心到“水平”坐标系
    pcd_aligned            = pcd.rotate(R_align, center=(0,0,0))
    camera_centers_aligned = (R_align @ camera_centers.T).T

    # 5. 基于所有相机中心的 mean + max_offset 构建 AABB，裁剪
    center_mean = camera_centers_aligned.mean(axis=0)
    offsets     = np.abs(camera_centers_aligned - center_mean)
    max_offset  = offsets.max(axis=0)
    margin      = args.margin
    min_bound   = center_mean - max_offset - margin
    max_bound   = center_mean + max_offset + margin
    aabb        = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    pcd_cropped_aligned = pcd_aligned.crop(aabb)

    # 6. 逆旋转回原坐标系
    R_inv        = R_align.T
    pcd_cropped  = pcd_cropped_aligned.rotate(R_inv, center=(0,0,0))

    # 7. 构造 kept vs removed 掩码
    all_pts = np.asarray(pcd.points)
    inside_idx = set(aabb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(all_pts)))
    mask_kept  = np.array([i in inside_idx for i in range(len(all_pts))])
    kept_pts   = all_pts[mask_kept]
    removed_pts= all_pts[~mask_kept]

    # 8. 统计并打印
    total, kept, removed = len(all_pts), len(kept_pts), len(removed_pts)
    print(f"总点数：{total}，保留：{kept}，删除：{removed}")

    # 9. 保存 COLMAP 格式 points3D.txt
    all_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((total,3))
    kept_colors = all_colors[mask_kept]
    with open(args.output_points_txt, "w") as f:
        f.write("# COLMAP points3D.txt\n")
        for idx, (pt, col) in enumerate(zip(kept_pts, kept_colors), start=1):
            r, g, b = (col * 255).astype(int)
            f.write(f"{idx} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b} 0\n")

    # 10. 生成带红色删除点标记的 PLY
    removed_colors = np.tile([1.0,0.0,0.0], (removed,1))
    all_kept   = kept_pts
    all_removed= removed_pts
    all_colors_marked = np.vstack([kept_colors, removed_colors])
    pcd_marked = o3d.geometry.PointCloud()
    pcd_marked.points = o3d.utility.Vector3dVector(
        np.vstack([all_kept, all_removed]))
    pcd_marked.colors = o3d.utility.Vector3dVector(all_colors_marked)
    o3d.io.write_point_cloud(args.output_ply, pcd_marked)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="裁剪点云并保存为 COLMAP 格式和 PLY 文件")
    parser.add_argument("--images_txt", type=str, required=True, help="COLMAP images.txt 文件路径")
    parser.add_argument("--pcd_path", type=str, required=True, help="点云文件路径 (.pcd)")
    parser.add_argument("--output_points_txt", type=str, required=True, help="输出 COLMAP points3D.txt 文件路径")
    parser.add_argument("--output_ply", type=str, required=True, help="输出带标记的 PLY 文件路径")
    parser.add_argument("--margin", type=float, default=15.0, help="裁剪边距，默认为 15.0")
    args = parser.parse_args()

    # 执行主函数
    main(args)