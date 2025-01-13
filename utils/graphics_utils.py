#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    """
    获取变换矩阵T
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()#转置
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    #主要计算相机中心是否发生平移和缩放，默认返回[R^T,t
    #                                       0,1]
    # Rt = np.zeros((4, 4))
    # Rt[:3, :3] = R.transpose(0,1)
    # Rt[:3, 3] = t
    # Rt[3, 3] = 1.0
    #
    # C2W = np.linalg.inv(Rt)
    # cam_center = C2W[:3, 3]
    # cam_center = (cam_center + translate) * scale#计算相机中心是否发生平移
    # C2W[:3, 3] = cam_center
    # Rt = np.linalg.inv(C2W)
    # return np.float32(Rt)

    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = torch.transpose(R, 0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + torch.from_numpy(translate) ) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt








def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    """
    根据给定的znear（近裁剪面距离）、zfar（远裁剪面距离）、fovX（水平视场角）和fovY（垂直视场角），首先计算了tan(fovY/2)和tan(fovX/2)，然后结合近裁剪面距离计算了投影平面的上、下、左、右边界。

    接着，根据这些边界值，构建了一个4x4的零矩阵P，并根据投影矩阵的定义填充了相应的数值，最终返回了计算得到的投影矩阵P。

    总的来说，这个函数实现了根据给定参数计算透视投影矩阵的功能。
    """
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))