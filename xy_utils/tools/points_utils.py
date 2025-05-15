import open3d as o3d
import numpy as np
import os

def pcd_2_ply(pcd_file_path,ply_file_path):
    point_cloud = o3d.io.read_point_cloud(pcd_file_path)
    o3d.io.write_point_cloud(ply_file_path, point_cloud)
    print(f"Converted {pcd_file_path} to {ply_file_path}")


def pcd_2_colmap_txt(pcd_file, txt_file, is_white=False):
    '''
    is_white: 是否将点云颜色设置为白色
    '''
    point_cloud = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)


    print(f"正在写入COLMAP TXT文件: {txt_file}")
    with open(txt_file, 'w') as f:
        f.write('# 3D point list with one line of data per point\n')
        f.write('#  POINT_ID, X, Y, Z, R, G, B, ERROR\n')
        f.write('# Number of points: {}\n'.format(len(points)))
        for i, point in enumerate(points):
            x, y, z = point
            if is_white:
                r, g, b = 255, 255, 255
            else:
                r, g, b = colors[i]
                r,g,b = int(r * 255), int(g * 255), int(b * 255)
            error = 0
            # 空的观察列表 (IMAGE_ID, POINT2D_IDX)
            # 写入一行
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error}\n")
    
    print(f"Converted {pcd_file} to Colmap TXT in: {txt_file}")

def voxel_downsample_and_save(voxel_size, input_ply_path, output_ply_path):
    """
    从PLY文件读取点云，进行体素化降采样，并保存结果。

    Args:
        input_ply_path (str): 输入PLY文件路径。
        voxel_size (float): 体素大小。
        output_ply_path (str): 保存降采样后点云的PLY文件路径。
    """
    # 读取输入点云
    pcd = o3d.io.read_point_cloud(input_ply_path)
    print(f"raw points len : {len(pcd.points)}")
    
    # 体素化降采样
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    print(f"downsample points len : {len(pcd_downsampled.points)}")
    
    # 保存降采样后的点云
    o3d.io.write_point_cloud(output_ply_path, pcd_downsampled)
    print(f"降采样后的点云已保存到 {output_ply_path}，体素大小：{voxel_size}")