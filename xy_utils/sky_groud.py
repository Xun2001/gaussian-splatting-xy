from tools.points_utils import *
import numpy as np
import math
import open3d as o3d


def fit_ground_plane(xyz, threshold=0.02):
    """
    使用 RANSAC 拟合地面平面模型。
    返回：平面模型 [a, b, c, d] 和内点索引。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"拟合的地面平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    return plane_model, inliers