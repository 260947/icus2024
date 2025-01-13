import open3d
import numpy as np
import copy
import os
from render import *


def open3d_segment(sq):
    # 读取点云文件
    # = r"D:\colma_dat\output\point_cloud\iteration_30000\point_cloud.ply"
    # pcd = open3d.io.read_point_cloud(pcd_path)
    # pcd = copy.deepcopy(sq)
    pcd = open3d.geometry.PointCloud()
    pcd.points = sq.points
    # pcd = open3d.geometry.PointCloud(pcd)
    pcd.paint_uniform_color(color=[0.5, 0.5, 0.5])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.035, ransac_n=10, num_iterations=1000)
    [A, B, C, D] = plane_model
    print(f"Plane equation: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0")
    colors = np.array(pcd.colors)
    # colors = np.ones(colors.shape)
    colors[inliers] = [1, 0, 0]  # 平面内的点设置为红色

    pcd.colors = open3d.utility.Vector3dVector(colors)
    # 点云可视化

    open3d.visualization.draw_geometries([pcd],
                                         window_name="segment",
                                         width=800,
                                         height=600)
    return inliers


# 点云聚类测试
def open3d_cluster(sq, in_plane):
    # 读取点云文件
    pcd = open3d.geometry.PointCloud()
    a = np.array([True] * len(np.array(sq.points)))
    a[in_plane] = False
    not_in_plane = list(a)
    # np.array(sq.points)
    # not_in_plane=[i for i in range(len(np.array(sq.points))) if i not in in_plane]

    pcd.points = open3d.utility.Vector3dVector(np.array(sq.points)[not_in_plane])
    # pcd.points = sq.points

    # 聚类距离设置为0.025，组成一类至少需要25个点
    labels = pcd.cluster_dbscan(eps=0.025, min_points=10, print_progress=True)
    max_label = max(labels)
    print(max_label)
    # 随机构建n+1种颜色，这里需要归一化
    colors = np.random.randint(1, 255, size=(max_label + 1, 3)) / 255.
    colors = colors[labels]  # 每个点云根据label确定颜色
    colors[np.array(labels) < 0] = 0  # 噪点配置为黑色
    # np.array(pcd.points)[np.array(labels) >= 0] 去除噪点
    pcd.colors = open3d.utility.Vector3dVector(colors)  # 格式转换(由于pcd.colors需要设置为Vector3dVector格式)
    # 可视化点云列表
    open3d.visualization.draw_geometries([pcd],
                                         window_name="cluster",
                                         width=1600,
                                         height=1200)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    open3d.visualization.draw_geometries([inlier_cloud],
                                         zoom=0.3412,
                                         front=[0.4257, -0.2125, -0.8795],
                                         lookat=[2.6172, 2.0475, 1.532],
                                         up=[-0.0694, -0.9768, 0.2024])


pcd_path = r"/home/k/gaussian-splatting/sphere/output/point_cloud/iteration_30000/point_cloud.ply"

pcd = open3d.io.read_point_cloud(pcd_path)

##去除噪点
labels = pcd.cluster_dbscan(eps=0.05, min_points=20, print_progress=True)

points = np.array(pcd.points)[np.array(labels) >= 0]
pcd.points = open3d.utility.Vector3dVector(points)

# 均匀采样
# voxel_down_pcd=pcd.uniform_down_sample(every_k_points=15)
# open3d.visualization.draw_geometries([voxel_down_pcd],
#                                          window_name="sklearn cluster",
#                                          width=1500,
#                                          height=1200)

# cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=20, radius=0.15)
#
# inlier_cloud = voxel_down_pcd.select_by_index(ind)

# display_inlier_outlier(voxel_down_pcd, ind)

# 分割出地面,返回平面内的点
in_plane = open3d_segment(pcd)

open3d_cluster(pcd, in_plane)

open3d.visualization.draw_geometries([pcd])
# model_path='/home/k/gaussian-splatting/data/output'
# loaded_iter=10000
# gaussians = GaussianModel(1)
# gaussians.load_ply(os.path.join(model_path,"point_cloud","iteration_" + str(loaded_iter),"point_cloud.ply"))#点云初始化
