import open3d as o3d
import torch
def fit_ground_plane(xyz, threshold=0.02):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=10000)
    return plane_model, inliers

def create_rotation_matrix(source_dir, target_dir):

    source_dir = source_dir / torch.norm(source_dir)
    target_dir = target_dir / torch.norm(target_dir)

    axis = torch.cross(source_dir, target_dir)
    cos_angle = torch.dot(source_dir, target_dir)

    if torch.norm(axis) < 1e-6:
        return torch.eye(3, device=source_dir.device)

    axis = axis / torch.norm(axis)
    # k = axis.unsqueeze(-1)
    # K = torch.zeros((3, 3), device=source_dir.device)
    # K[0, 1] = -k[2]
    # K[0, 2] = k[1]
    # K[1, 2] = -k[0]
    # K = K - K.T
    # R = torch.eye(3, device=source_dir.device) + K + K @ K * (1 - cos_angle) / (1 + cos_angle)
    R_align = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * torch.acos(cos_angle))
    return R_align