import numpy as np
import open3d
import json
import time
import subprocess
from utils.system_utils import searchForMaxIteration
from demos.render import *
from nav import vec_to_rot_matrix
from scipy.spatial import KDTree


# def load_ply(path):
#     plydata = PlyData.read(path)
#     global xyz,opacities
#     global xyz,opacities
#     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                     np.asarray(plydata.elements[0]["y"]),
#                     np.asarray(plydata.elements[0]["z"])), axis=1)
#     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

# load_ply('../gaussian-splatting/data/output/point_cloud/iteration_7000/point_cloud.ply')
# density = torch.sigmoid(torch.tensor(opacities))

class Planner:
    def __init__(self, start_state, end_state, xyz, scales, density, delta_t, path_ref):
        self.start_state = start_state
        self.end_state = end_state
        self.delta_t = delta_t  ##时间间隔
        #### 定义一些参数
        self.eta = 15  # 引力增益系数，15
        self.zeta = 20  # 斥力增益系数，20
        self.rho0 = 0.055  # 障碍物影响范围，0.055
        self.goal_tol = 0.03  # 目标点容差，0.01
        self.max_iter = 30  # 最大迭代次数，500
        self.n = 0.5  # 斥力修正因子 取值范围[0,1]
        ###高斯的一些参数
        self.density = density
        self.xyz = xyz
        self.scales = scales
        self.tree = KDTree(xyz)

        self.att_list = []
        self.rep_list = []
        self.ref_path = path_ref

    def att_potential(self, curr, goal):
        return 0.5 * self.eta * (np.linalg.norm(curr - goal)) ** 2

    def rep_potential(self, curr, goal, use_kdtree=False):
        rep_sum = 0
        if use_kdtree:
            indices = self.tree.query_ball_point(curr.tolist(), self.rho0 + self.scales.max())
            for i in indices:
                dist = np.linalg.norm(curr - self.xyz[i])
                if dist <= self.rho0 + self.scales[i][0]:
                    rep_sum += 0.5 * self.zeta * self.density[i][0] * (
                            1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                                   np.linalg.norm(curr - goal)) ** self.n
        else:
            distances = np.linalg.norm(self.xyz - curr, axis=1)
            for i in range(len(distances)):
                if not args.APF_nearby:
                    self.scales[i][0] = 0
                dist = distances[i]
                if dist <= self.rho0 + self.scales[i][0]:  # + self.scales[i][0]

                    rep_sum += 0.5 * self.zeta * (1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                        np.linalg.norm(curr - goal)) ** self.n
        return rep_sum * 1e-4

    # 计算引力
    def attractive_force(self, curr, goal):
        # 引力与点到目标的距离平方成正比

        # 返回引力势场梯度力
        return -self.eta * (curr - goal)

    def repulsive_force_with_all_gaussians(self, point, goal):
        grad = np.array([0., 0., 0.])
        rep_sum = 0

        distances = np.linalg.norm(self.xyz - point, axis=1)
        for i in range(len(distances)):
            dist = distances[i]
            # self.scales[i][0] = 0
            if dist <= self.rho0 + self.scales[i][0]:  # + self.scales[i][0]
                # self.scales[i][0] = 0
                rep_sum += 0.5 * self.zeta * self.density[i][0] * (
                        1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                               np.linalg.norm(point - goal)) ** self.n
                grad = grad + np.linalg.norm(point - goal) ** self.n * (
                        -self.zeta * self.density[i][0] * (1 / dist - 1.0 / (self.rho0 + self.scales[i][0])) * (
                        1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (
                               1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                               point - goal) * np.linalg.norm(point - goal) ** (self.n - 2)

        return -grad, rep_sum * 1e-4

    def repulsive_force(self, point, goal, use_kdtree=False):
        # 斥力与点到障碍物的距离成反比
        # 如果距离大于阈值，则斥力为零
        grad = np.array([0., 0., 0.])
        rep_sum = 0

        ###kd-tree版本######
        ### 查询距离点(x, y)在r范围内的所有点的索引,不考虑scales！！
        if use_kdtree:
            indices_may = self.tree.query_ball_point(point.tolist(), self.rho0 + self.scales.max())  ##self.scales.max()

            # valid_indices = np.linalg.norm(self.xyz[indices_may] - point, axis=1) <= (self.scales[indices_may] + self.rho0).reshape(-1)
            # valid_points = self.xyz[indices_may][valid_indices]
            # distances = np.linalg.norm(valid_points - point, axis=1)
            # valid_density = self.density[indices_may][valid_indices]
            # valid_scales = self.scales[indices_may][valid_indices]
            # rep_sum = np.sum(valid_density.reshape(-1) * (1 / (distances ) - 1 /( valid_scales.reshape(-1) + self.rho0)) ** 2) * 0.5 * self.zeta* (np.linalg.norm(point - goal)) ** self.n
            #
            # first_item =  -self.zeta * np.linalg.norm(point - goal) ** self.n * np.sum((valid_density.reshape(-1) * (1/distances - 1/( self.rho0+valid_scales.reshape(-1) ))  * 1/distances ** 3 ).reshape(-1,1) * (point - valid_points),axis=0)
            # second_item = 0.5 * self.n * self.zeta * np.linalg.norm(point - goal) ** (self.n - 2) * (point - goal) * np.sum( ( 1 / distances  - 1 /( valid_scales.reshape(-1) + self.rho0) ) ** 2 )
            # grad = first_item + second_item

            # 不考虑scales
            # for i in indices_may:
            #     dist = np.linalg.norm(point-self.xyz[i])
            #     rep_sum += 0.5 * self.zeta * self.density[i][0] * (1 / dist - 1 / (self.rho0 )) ** 2 * (
            #         np.linalg.norm(point - goal)) ** self.n
            #     grad = grad + np.linalg.norm(point - goal) ** self.n * (
            #                 -self.zeta * self.density[i][0] * (1 / dist - 1.0 / ( self.rho0 )) * (
            #                     1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (
            #                        1 / dist - 1 / (self.rho0 )) ** 2 * (point - goal) * np.linalg.norm(
            #         point - goal) ** (self.n - 2)

            ##考虑scales
            for i in indices_may:
                dist = np.linalg.norm(point - self.xyz[i])
                if dist <= self.rho0 + self.scales[i][0]:
                    rep_sum += 0.5 * self.zeta * self.density[i][0] * (
                            1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                                   np.linalg.norm(point - goal)) ** self.n
                    grad = grad + np.linalg.norm(point - goal) ** self.n * (
                            -self.zeta * self.density[i][0] * (1 / dist - 1.0 / (self.rho0 + self.scales[i][0])) * (
                            1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (
                                   1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                                   point - goal) * np.linalg.norm(
                        point - goal) ** (self.n - 2)


        else:
            distances = np.linalg.norm(self.xyz - point, axis=1)
            for i in range(len(distances)):
                if not args.APF_nearby:
                    self.scales[i][0] = 0
                dist = distances[i]
                if dist <= self.rho0 + self.scales[i][0]:  # + self.scales[i][0]
                    rep_sum += 0.5 * self.zeta * self.density[i][0] * (
                            1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                                   np.linalg.norm(point - goal)) ** self.n
                    grad = grad + np.linalg.norm(point - goal) ** self.n * (
                            -self.zeta * self.density[i][0] * (1 / dist - 1.0 / (self.rho0 + self.scales[i][0])) * (
                            1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (
                                   1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                                   point - goal) * np.linalg.norm(point - goal) ** (self.n - 2)
                    # grad = grad + np.linalg.norm(point - goal) ** self.n * (-self.zeta * self.density[i][0] * (1 / dist - 1.0 / self.rho0 ) * (1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (1 / dist - 1 / (self.rho0 )) ** 2 * (point - goal) * np.linalg.norm(point - goal) ** (self.n - 2)

        # 返回斥力势场梯度力以及势能大小
        return -grad, rep_sum * 1e-4

    def stuck_in_local_minimum(self, path):

        sum_1, sum_2 = 0, 0
        i, j = -5, 2
        pre = path[-1]
        pre_2 = path[0]
        for k in range(3):
            sum_1 += np.linalg.norm(pre - path[i])
            pre = path[i]
            i -= 4

            sum_2 += np.linalg.norm(pre_2 - path[j])
            pre_2 = path[j]
            j += 2
        # if sum_1  < 0.85 * sum_2 :
        #     print('极小值！！！！！')

        return sum_1 < 0.85 * sum_2

    def path_searching(self, start, goal_final):
        path = [start]
        last_iter = -5
        curr_pos = start
        t1 = time.time()
        ##while np.linalg.norm(curr_pos - goal) > self.goal_tol and iters < self.max_iter:
        # reference_path = np.vstack((self.ref_path,goal_final.reshape(1,3)))
        reference_path = goal_final.reshape(1, 3)
        for k in range(len(reference_path)):
            goal = reference_path[k]
            if k == len(reference_path) - 1:
                self.goal_tol /= 2
                self.max_iter = 120
            iters = 0
            record = []
            max_vel = 4.5  # 最大步长 5
            while iters < self.max_iter:
                if np.linalg.norm(curr_pos - goal) <= self.goal_tol:
                    break

                iters += 1
                ##计算斥力和引力
                att_force = self.attractive_force(curr_pos, goal)

                if args.pure_apf:
                    rep_force, rep_sum = self.repulsive_force_with_all_gaussians(curr_pos, goal_final)

                else:
                    rep_force, rep_sum = self.repulsive_force(curr_pos, goal_final)

                vel_k = 1
                # if rep_sum < 0.0005 and self.att_potential(curr_pos,goal_final) > 0.07:
                # vel_k = 1.3
                # print('speed up!!')

                if rep_sum > 0.0005:
                    vel_k = 0.5
                    # print('slow down!!')

                gradient = att_force + rep_force
                grad_len = np.linalg.norm(gradient)
                gradient = gradient / grad_len  # 归一化
                vel = vel_k * max_vel * (1 - np.exp(-grad_len / 5)) * gradient  # 世界坐标系下无人机的理论速度

                ##计算势能
                self.att_list.append(self.att_potential(curr_pos, goal))
                self.rep_list.append(rep_sum)

                ###处理局部极值###
                attr_v = att_force / np.linalg.norm(att_force)
                rep_v = rep_force / (np.linalg.norm(rep_force) + 0.000001)
                cross_ = np.cross(attr_v, rep_v)  #
                dot_ = np.dot(attr_v, rep_v)
                sign = 0

                record.append(0)
                if (np.linalg.norm(curr_pos - goal) > 1.5 * self.goal_tol and len(path) >= 10 and np.linalg.norm(
                        path[-3] - path[-1]) <= 0.005) or (np.linalg.norm(curr_pos - goal) > 2 * self.goal_tol and (
                        len(path) > 15 and self.stuck_in_local_minimum(path))):
                    record[-1] = 1
                    # sign = 1
                if dot_ < -0.85:
                    curr_potential = self.att_list[-1] + self.rep_list[-1]
                    next_potential = self.rep_potential(curr_pos + vel * self.delta_t, goal) + self.att_potential(
                        curr_pos + vel * self.delta_t, goal)
                    if next_potential - curr_potential > 0.2 and iters - last_iter > 5:
                        sign += 4.5

                if len(record) >= 5:
                    sign += sum(record[-5:]) / 2

                sign -= dot_
                if sign >= 3:
                    print('local min!!!')
                    if args.APF_random:
                        vel += np.random.normal(0, 0.01, (3,))
                    # print(np.random.normal(0, 0.05,(3,)))
                    #####break

                if len(path) > 1:
                    last_pos = path[-2]
                    rot_v = np.cross(vel, curr_pos - last_pos)
                    rot_v = rot_v / np.linalg.norm(rot_v)
                    curr_cos = np.dot(vel, curr_pos - last_pos) / (
                                np.linalg.norm(curr_pos - last_pos) * np.linalg.norm(vel))

                    if curr_cos - 0.5 < 0:
                        rot_v *= (np.arccos(curr_cos) - np.pi / 3)
                        rot_mat = vec_to_rot_matrix(
                            torch.tensor(rot_v, dtype=torch.float32, device='cuda')).cpu().numpy()
                        vel = rot_mat @ vel

                next_pos = curr_pos + vel * self.delta_t

                path.append(next_pos)
                curr_pos = next_pos

            print('turn:', iters)
        print('耗时：', time.time() - t1)
        return path


def cal_cueve(a, b, c):
    ##a,b,c,3个坐标点
    if np.linalg.norm(np.cross(a - b, b - c)) < 1e-10:
        return 0

    a_len, b_len, c_len = np.linalg.norm(a - b), np.linalg.norm(a - c), np.linalg.norm(c - b)
    if a_len * b_len * c_len == 0:
        return -1
    p = (a_len + b_len + c_len) / 2
    r = (a_len * b_len * c_len) / (4 * np.sqrt(p * (p - a_len) * (p - b_len) * (p - c_len)))

    return 1 / r


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--APF_random", default=False, type=bool)
    parser.add_argument("--APF_nearby", default=False, type=bool)
    parser.add_argument("--APF_quantization", default=False, type=bool)
    parser.add_argument("--pure_apf", default=False, type=bool)

    args = get_combined_args(parser)
    print("Loading " + args.model_path)

    ##############加载高斯######
    gaussians = GaussianModel(1)
    loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    pcd_path = os.path.join(args.model_path,
                            "point_cloud",
                            "iteration_" + str(loaded_iter),
                            "point_cloud.ply")
    gaussians.load_ply(pcd_path)  # 点云初始化
    print('高斯轮次：', loaded_iter)
    bg_color = [1, 1, 1] if True else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    xyz = gaussians.get_xyz  ##高斯位置
    scales = gaussians.get_scaling  ##高斯大小
    density = gaussians.get_opacity  ##高斯密度
    delta_t = 0.02  ##时间间隔

    if 'data' in args.model_path:
        blend_file = '../blender_scenes/stonehenge_test1121.blend'
        before = 'stonehenge/'
        pcd = open3d.io.read_point_cloud(pcd_path)
        # 去除噪点
        labels = pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True)
        mask = np.array(labels) >= 0
        del pcd

        ##初步筛选##
        xyz_mask = (xyz[mask]).detach().cpu().numpy()
        scales_mask = (scales[mask]).detach().cpu().numpy()
        density_mask = (density[mask]).detach().cpu().numpy()

        start = np.array([-0.036, 0.5531, 0.1637])
        goal = np.array([0.1456, -0.7668, 0.17111])

    elif 'sphere' in args.model_path:
        before = 'sphere/'
        blend_file = '../blender_scenes/sphere.blend'
        xyz_mask = xyz.detach().cpu().numpy()
        scales_mask = scales.detach().cpu().numpy()
        density_mask = density.detach().cpu().numpy()

        # 密度筛选(必须！！)
        xyz_mask = xyz_mask[np.copy(density_mask.flatten()) > 0.005]
        scales_mask = scales_mask[density_mask > 0.005].reshape(-1, 1)
        density_mask = density_mask[density_mask > 0.005].reshape(-1, 1)

        start = np.array([-0.2556, 0.7903, 0.2169])
        goal = np.array([-0.5554, -0.9017, 0.3058])
    else:
        assert False, 'Can not recognize scene Type!'

    if args.APF_quantization:
        # # 定义空间网格大小
        grid_size = 0.01  # 网格大小为0.1

        # 将点云数据映射到空间网格中
        grid_indices = np.floor(xyz_mask / grid_size).astype(int)

        # 创建空间网格
        grid = {}
        for i, point in enumerate(xyz_mask):
            grid_index = tuple(grid_indices[i])
            if grid_index in grid:
                grid[grid_index][0] += point
                grid[grid_index][1] += density_mask[i]
                grid[grid_index][2] += scales_mask[i]
                grid[grid_index][3] += 1

            else:
                grid[grid_index] = [point, density_mask[i], scales_mask[i], 1]

        # 计算每个网格内点的均值作为新的点

        xyz_mask = []
        scales_mask = []
        density_mask = []

        for grid_index, vals in grid.items():
            merged_xyz = vals[0] / vals[3]  #
            merged_den = vals[1] / vals[3]
            merged_sca = vals[2] / vals[3]

            xyz_mask.append(merged_xyz)
            scales_mask.append(merged_den)
            density_mask.append(merged_sca)

        xyz_mask = np.array(xyz_mask)
        scales_mask = np.array(scales_mask)
        density_mask = np.array(density_mask)

    # latest_init =  '/home/k/nerfnavigation_original/paths/sphere_nerf/init_poses/49.json'
    # latest_init = 'potential_visual.json'
    # with open(latest_init, 'r') as f:
    #     meta = json.load(f)
    #
    # points = (np.array(meta["poses"])[..., 3:][:, :3]).reshape(-1, 3)
    # min_distance = 99
    # for i in points:
    #     for j in xyz_mask:
    #         min_distance = min(min_distance,np.linalg.norm(i-j))
    #
    # print('min_dis',min_distance)

    # 定义起点和终点 0.0330 -0.793251 m  0.251245 m -0.69881 m
    # start=-0.54, -0.76, 0.17，end = 0.08,0.60, 0.12.png
    # stoneheng   start= np.array([-0.048, 0.55317, 0.1844])  goal= np.array([0.13292, -0.7668, 0.1666])
    # start= np.array([-0.036, 0.5531, 0.1637])  # 起点坐标 0.0829 0.6033 0.12724 -0.2556, 0.7903, 0.2169
    # goal= np.array([0.1456, -0.7668, 0.17111])  # 终点坐标 -0.5554, -0.9017, 0.3058
    init_pose = np.array([0, 0, 0])  ##初始姿态（欧拉角）
    end_pose = np.array([0, 0, 0])  ##目标姿态（欧拉角）
    init_rates = np.zeros(3)  ##初始角速度和线速度

    #############################
    side = 21  # 生成立方体的数量+1
    # 场景边界
    x_min, x_max = xyz_mask[:, 0].min(), xyz_mask[:, 0].max()
    y_min, y_max = xyz_mask[:, 1].min(), xyz_mask[:, 1].max()
    z_min, z_max = xyz_mask[:, 2].min(), xyz_mask[:, 2].max()

    # 生成3D坐标
    x_linspace = np.linspace(x_min, x_max, side)
    y_linspace = np.linspace(y_min, y_max, side)
    z_linspace = np.linspace(z_min, z_max, side)
    # （side-1，side-1，side-1）个方格
    # coods shape =（side，side，side,3）
    # coods = np.stack( np.meshgrid( x_linspace, y_linspace, z_linspace ),indexing='ij' )
    X, Y, Z = np.meshgrid(x_linspace, y_linspace, z_linspace, indexing='ij')
    coods = np.stack((X, Y, Z), axis=-1)
    # x,y,z=x_max,y_max,z_max
    inter_val = (np.array([x_max, y_max, z_max]) - np.array([x_min, y_min, z_min])) / (side - 1)
    occupy = np.zeros([i - 1 for i in coods.shape[:-1]] + [2])

    for i in range(xyz_mask.shape[0]):
        loc = (np.array([xyz_mask[i][0], xyz_mask[i][1], xyz_mask[i][2]]) - np.array([x_min, y_min, z_min])) / inter_val
        xx, yy, zz = min(side - 2, int(loc[0])), min(side - 2, int(loc[1])), min(side - 2, int(loc[2]))
        occupy[xx][yy][zz][0] += density_mask[i][0]  # 高斯密度
        occupy[xx][yy][zz][1] += 1  # 高斯数量

    # print(occupy)
    occupied = (occupy[..., 0] / (occupy[..., 1] + 0.0001)) > 0.2  # 计算每个立方体平均密度,密度大于0.2的视为被占据
    # occupied = occupy[..., 1] > 0
    # 计算起点和终点的网格坐标
    loc_s = (np.array([start[0], start[1], start[2]]) - np.array([x_min, y_min, z_min])) / inter_val
    loc_g = (np.array([goal[0], goal[1], goal[2]]) - np.array([x_min, y_min, z_min])) / inter_val
    xx_s, yy_s, zz_s = int(loc_s[0]), int(loc_s[1]), int(loc_s[2])
    xx_g, yy_g, zz_g = int(loc_g[0]), int(loc_g[1]), int(loc_g[2])

    start_a = (xx_s, yy_s, zz_s)
    end_a = (xx_g, yy_g, zz_g)

    #####状态初始化#######
    start_state = np.hstack((start, init_rates, init_pose, init_rates))  ##初始state
    end_state = np.hstack((goal, init_rates, end_pose, init_rates))  ##结束state

    traj = Planner(start_state, end_state, xyz_mask, scales_mask, density_mask, delta_t, None)

    path = traj.path_searching(start, goal)

    print('目标偏差：', np.linalg.norm(goal - path[-1]))
    # plt.plot(traj.att_list,color='r',label='att_potential')
    # plt.plot(traj.rep_list, color='g', label='rep_potential')
    # plt.plot(np.array(traj.att_list)+np.array(traj.rep_list), color='b', label='total_potential')
    # plt.legend(fontsize=12)
    # plt.tick_params(labelsize=10)
    # plt.xlabel('iters',fontsize=12,loc = 'right')
    # plt.ylabel('values',fontsize=12,loc = 'top')
    # # 显示图形
    # plt.show()

    #####visuialize###
    if args.pure_apf:
        filename = 'pure_apf.json'

    if args.APF_random:
        filename = 'APF_random.json'

    if args.APF_quantization:
        filename = 'APF_quantization.json'

    if args.APF_nearby:
        filename = 'APF_nearby.json'

    path.insert(0, (path[0] + path[1]) / 2)
    filename = before + filename
    poses = []
    pose_dict = {}
    with open(filename, "w+") as f:
        for pos in path:
            pose = np.zeros((4, 4))
            pose[:3, :3] = np.identity(3)
            pose[:3, 3] = pos
            pose[3, 3] = 1
            poses.append(pose.tolist())
        pose_dict["poses"] = poses
        json.dump(pose_dict, f, indent=4)

    subprocess.run(['blender', blend_file, '-P', 'viz_data_blend.py', '--', filename, '0.2'])
