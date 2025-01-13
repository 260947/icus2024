import time

from plyfile import PlyData
import numpy as np
import torch
import heapq
import math
import os
import json

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from render import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ply(path):
    plydata = PlyData.read(path)
    global xyz, opacities
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]


# self.camera_center = self.world_view_transform.inverse()[3, :3]
# self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
# self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
# self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()


# rot_x = lambda phi: torch.tensor([
#         [1., 0., 0.],
#         [0., torch.cos(phi), -torch.sin(phi)],
#         [0., torch.sin(phi), torch.cos(phi)]], dtype=torch.float32, device=device)
# def nerf_matrix_to_ngp_torch(pose, trans):
#     neg_yz = torch.tensor([
#         [1, 0, 0],
#         [0, -1, 0],
#         [0, 0, -1]
#     ], dtype=torch.float32)
#
#     flip_yz = torch.tensor([
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 0, 0]
#     ], dtype=torch.float32)
#     return flip_yz@ pose @ neg_yz, flip_yz @ trans

def get_cam_intr(path):
    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    intr = cam_intrinsics[0]
    height = intr.height  # 图像高度
    width = intr.width  # 图像宽度
    if intr.model == "SIMPLE_PINHOLE":
        focal_length_x = intr.params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
    elif intr.model == "PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    return FovX, FovY


def render_from_pose(pose, gaussians, pipeline, background):
    """
    从指定pose得到渲染rgb图,pose是世界坐标下4*4变换矩阵
    """
    # 这里是旋转到nerf坐标系？
    # rot = rot_x(torch.tensor(np.pi/2)) @ pose[:3, :3]
    # trans = pose[:3, 3]
    # pose, trans = nerf_matrix_to_ngp_torch(rot, trans)
    #
    # new_pose = torch.eye(4)
    # new_pose[:3, :3] = pose
    # new_pose[:3, 3] = trans
    # #以new_pose为变换矩阵
    # rays = self.get_rays(new_pose.reshape((1, 4, 4)))
    #
    # output = self.render_fn(rays["rays_o"], rays["rays_d"])
    # output also contains a depth channel for use with depth data if one chooses

    bg_color = [1, 1, 1] if background else [0, 0, 0]  # 是否白色背景，命令行参数
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # gs render!
    view = Pose_to_view(pose)

    img = render(view, gaussians, pipeline, background)["render"]
    rgb = img.permute(1, 2, 0).reshape((-1, 3))
    return rgb, img


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


class Pose_to_view:
    def __init__(self, pose):
        c2w = pose
        c2w[:3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w)
        # w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        # pose[:3, :3] = R
        # pose[:3, 3] = T
        # [[0.890663743019104, -0.026803040876984596, 0.4538719058036804,         1.134679731443595],
        #  [0.45466262102127075, 0.05250595882534981, -0.8891147375106812,        -2.2227868575679617],
        #  [0.0,                  0.9982608556747437,   0.058951497077941895,     0.14737873955980724], [0.0, 0.0, 0.0, 1.0]]

        #  R=np.array([[ 0.95891409, -0.06406249,  0.2763689 ],
        # [ 0.2830135 ,  0.14845106, -0.94755773],
        # [ 0.01967565,  0.98684258,  0.16048237]])
        #  T=np.array([ 0.02886221, -2.14467306,  4.13118268])
        self.image_height = 800
        self.image_width = 800
        self.FoVx = 0.6911112070083618
        self.FoVy = 0.6911112070083618
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)


def astar(occupied, start, goal):
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)

    def inbounds(point):
        for x, size in zip(point, occupied.shape):
            if x < 0 or x >= size: return False
        return True

    neighbors = [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1)]

    close_set = set()

    came_from = {}
    gscore = {start: 0}

    assert not occupied[start]
    assert not occupied[goal]

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            assert current == start
            data.append(current)
            return list(reversed(data))

        close_set.add(current)

        for i, j, k in neighbors:
            neighbor = (current[0] + i, current[1] + j, current[2] + k)
            if not inbounds(neighbor):
                continue

            if occupied[neighbor]:
                continue

            tentative_g_score = gscore[current] + 1

            if tentative_g_score < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score

                fscore = tentative_g_score + heuristic(neighbor, goal)
                node = (fscore, neighbor)
                if node not in open_heap:
                    heapq.heappush(open_heap, node)

    raise ValueError("Failed to find path!")


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


if __name__ == "__main__":

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # with torch.no_grad():
    gaussians = GaussianModel(1)
    gaussians.load_ply(os.path.join(args.model_path,
                                    "point_cloud",
                                    "iteration_" + str(30000),
                                    "point_cloud.ply"))  # 点云初始化

    # scene = Scene(model.extract(args), gaussians, load_iteration=-1, shuffle=False)
    bg_color = [1, 1, 1] if True else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = pipeline.extract(args)

    with open('/home/k/gaussian-splatting/sphere/transforms_val.json') as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
    i = 0
    start = time.time()
    for frame in frames:
        if i >= 100:
            break
        i += 1
        pose = frame["transform_matrix"]

        # pose = [[ 0.4954023, -0.2749849,  -0.8239901,  -0.216587 ],
        #     [-0.81351888, 0.1857191, -0.55108571, 0.471486985],
        #     [ 0.304571, 0.9433407,  -0.1316997,  0.205882],
        #     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]

        pose = torch.tensor(pose)

        out, rendering = render_from_pose(pose, gaussians, pipeline, 1)
        # torchvision.utils.save_image(rendering, os.path.join('/home/k/图片', '{0:05d}'.format(10086) + ".png"))

    print("cost time:", time.time() - start)
    # rendering = render(scene.getTrainCameras(), gaussians, pipeline, background)["render"]
