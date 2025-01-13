import matplotlib.pyplot as plt
import numpy as np
import open3d
import random
import heapq
import json
import time
from utils.system_utils import searchForMaxIteration
from demos.render import *
from nav import vec_to_rot_matrix
from scipy.spatial import KDTree
import subprocess


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

def query(curr_pos, goal_final):
    collect_list = [curr_pos]
    # vel += np.random.normal(8,1,(3,)
    curr_int = list(map(int, (curr_pos - np.array([x_min, y_min, z_min])) / inter_val))
    goal_int = list(map(int, (goal_final - np.array([x_min, y_min, z_min])) / inter_val))

    while curr_int != goal_int:
        action = nav_map[curr_int[0], curr_int[1], curr_int[2]]
        curr_int = [x + y for x, y in zip(curr_int, action)]
        collect_list.append(curr_int * inter_val + np.array([x_min, y_min, z_min]))

    return collect_list


def execute(path, method_name, distur_sign):
    np.random.seed(76)  # 设置随机种子10 7 3 45 ！！！76！！！
    max_vel = 2  # max_vel = 2m/s
    inter_time = 0.025  # time
    curr = path[0]  #
    goal = path[-1]
    sign = 0
    distur_set = [np.array([0.25, 0.01, 0.15]),
                  np.array([0.2, -0.15, 0.1])]  # [np.random.normal(0, 0.03,(3,)),np.random.normal(0.5, 0.03,(3,))]
    if 'sphere' in args.model_path:
        distur_set = [np.array([-0.1, -0.05, 0.2]), np.array([0.15, -0.1, -0.2])]

    distur_set = [np.random.normal(0, 0.025, (3,)), np.random.normal(0, 0.025, (3,))]
    way_points_list = [curr]
    start_time = time.time()
    i = 1
    last = -4
    while i < len(path):
        point = path[i]
        if distur_sign:
            if random.randint(0, 10) > 3:
                distur_val = np.random.normal(0.01, 0.035, (3,))
                curr = curr + distur_val
                way_points_list.append(curr)
            else:
                distur_val = 0

            if np.linalg.norm(distur_val) > 0.1:
                sign += 1
                if method_name == 'dp':
                    query_time = time.time()
                    path = query(curr, path[-1])
                    print("第" + str(sign) + "次DP query cost time:", time.time() - query_time)

                else:
                    replan_time = time.time()
                    loc_s = (curr - np.array([x_min, y_min, z_min])) / inter_val
                    loc_g = (goal - np.array([x_min, y_min, z_min])) / inter_val
                    xx_s, yy_s, zz_s = int(loc_s[0]), int(loc_s[1]), int(loc_s[2])
                    xx_g, yy_g, zz_g = int(loc_g[0]), int(loc_g[1]), int(loc_g[2])
                    start_a = (xx_s, yy_s, zz_s)
                    end_a = (xx_g, yy_g, zz_g)
                    path_astar = astar(occupied, start_a, end_a)
                    path = np.array([x_min, y_min, z_min]) + np.array(path_astar, dtype=float) * inter_val
                    print("第" + str(sign) + "次A* replanning cost time:", time.time() - replan_time)

                i = 0
                continue

        while np.linalg.norm(curr - point) > 0.06:
            vel = max_vel * (point - curr) / np.linalg.norm(point - curr)
            curr = curr + inter_time * vel
            way_points_list.append(curr)
            ####if  sign < len(distur_set) and i < 3* len(path) / 4 and i >  len(path) / 3 and i - last > 1:
            last = i
            ####curr = curr +  distur_set[sign]
            # sign += 1
            # print("第" + str(sign) + "次扰动")
            # way_points_list.append(curr)

            # if method_name == 'dp':
            #     query_time = time.time()
            #     path = query(curr,path[-1])
            #     print("第"+str(sign)+"次DP query cost time:", time.time() - query_time)
            # else:
            #
            #     replan_time = time.time()
            #     loc_s = (curr - np.array([x_min, y_min, z_min])) / inter_val
            #     loc_g = (goal - np.array([x_min, y_min, z_min])) / inter_val
            #     xx_s, yy_s, zz_s = int(loc_s[0]), int(loc_s[1]), int(loc_s[2])
            #     xx_g, yy_g, zz_g = int(loc_g[0]), int(loc_g[1]), int(loc_g[2])
            #     start_a = (xx_s, yy_s, zz_s)
            #     end_a = (xx_g, yy_g, zz_g)
            #     path_astar = astar(occupied , start_a , end_a)
            #     path = np.array([x_min, y_min, z_min]) + np.array(path_astar, dtype=float) * inter_val
            #     print("第"+str(sign)+"次A* replanning cost time:", time.time() - replan_time)

            # i = 0
            # break

        i += 1

    print(method_name + "跟踪耗时：", time.time() - start_time)

    return sign, way_points_list


class Planner:
    def __init__(self, start_state, end_state, xyz, scales, density, delta_t, path_ref, APF_ASTAR):
        self.APF_ASTAR = APF_ASTAR
        self.start_state = start_state
        self.end_state = end_state
        self.delta_t = delta_t  ##时间间隔
        #### 定义一些参数
        self.eta = 15  # 引力增益系数，15
        self.zeta = 20  # 斥力增益系数，20
        self.rho0 = 0.05  # 障碍物影响范围，0.05
        self.goal_tol = 0.04  # 目标点容差，0.01
        self.max_iter = 15  # 最大迭代次数，500
        self.n = 0.5  # 斥力修正因子 取值范围[0,1] 0.5
        ###高斯的一些参数
        self.density = density
        self.xyz = xyz
        self.scales = scales
        self.tree = KDTree(xyz)

        self.att_list = []
        self.rep_list = []
        self.ref_path = path_ref

    def params(self):
        return [self.states]

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)

        try:
            for it in range(self.epochs_init):
                opt.zero_grad()
                self.epoch = it
                loss = self.rep_potential()
                print(it, loss)
                loss.backward()
                opt.step()

                # save_step = 50
                # if it%save_step == 0:
                #     if hasattr(self, "basefolder"):
                #         self.save_poses(self.basefolder / "init_poses" / (str(it//save_step)+".json"))
                #         self.save_costs(self.basefolder / "init_costs" / (str(it//save_step)+".json"))
                #     else:
                #         print("Warning: data not saved!")

        except KeyboardInterrupt:
            print("finishing early")

    def att_potential(self, curr, goal):
        return 0.5 * self.eta * (np.linalg.norm(curr - goal)) ** 2

    def rep_potential(self, curr, goal, use_kdtree=True):
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
                dist = distances[i]
                if dist <= self.rho0 + self.scales[i][0]:
                    rep_sum += 0.5 * self.zeta * (1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                        np.linalg.norm(curr - goal)) ** self.n
        return rep_sum

    # 计算引力
    def attractive_force(self, curr, goal):
        # 引力与点到目标的距离平方成正比

        # 返回引力势场梯度力
        return -self.eta * (curr - goal)

    def repulsive_force(self, point, goal, use_kdtree=True):
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
                # self.scales[i][0] = 0
                # self.density[i][0] = 1
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
                dist = distances[i]
                if dist <= self.rho0 + self.scales[i][0]:
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
        return -grad, rep_sum

    def plot_sth(self, path):
        inter_val = 0.05
        res = []
        for i in range(len(path)):
            count = 0
            for j in path:
                if np.linalg.norm(path[i] - j) < inter_val:
                    count += 1
            res.append(count)
        plt.plot(res, color='k', label='total_num')
        plt.legend()
        plt.show()

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
        ##np.random.seed(21)#设置随机种子
        path = [start]
        last_iter = -5
        curr_pos = start
        max_vel = 5  # 最大步长 5
        t1 = time.time()
        ##while np.linalg.norm(curr_pos - goal) > self.goal_tol and iters < self.max_iter:
        reference_path = np.vstack((self.ref_path, goal_final.reshape(1, 3)))
        # reference_path = goal_final.reshape(1,3)
        for k in range(len(reference_path)):
            goal = reference_path[k]
            if k == len(reference_path) - 1:
                self.goal_tol /= 2
                self.max_iter = 40
                next_goal = goal_final
            else:
                next_goal = reference_path[k + 1]

            if self.APF_ASTAR:
                goal_next_num = 1
                track_weight = 1
            else:
                goal_next_num = 3  ##待跟踪点数量 [3,0.9],  [5,0.8]
                track_weight = 0.9  ##跟踪权重
            iters = 0
            record = []
            while iters < self.max_iter:
                #################################
                break_sign = 1
                att_force = 0
                for num in range(min(len(reference_path) - k, goal_next_num)):
                    if np.linalg.norm(curr_pos - reference_path[k + num]) <= self.goal_tol:
                        break_sign = 0
                        break
                    else:
                        if k + num == len(reference_path) - 1:
                            track_weight = 1
                        else:
                            track_weight = 0.8

                        att_force += (track_weight ** num) * self.attractive_force(curr_pos, reference_path[k + num])

                if break_sign == 0:
                    break
                ###############################

                # if np.linalg.norm(curr_pos - goal) <= self.goal_tol:
                #     break

                iters += 1
                ##计算斥力和引力
                ###att_force = self.attractive_force(curr_pos,goal)
                rep_force, rep_sum = self.repulsive_force(curr_pos, goal_final)

                vel_k = 1
                if rep_sum < 0.01 and self.att_potential(curr_pos, goal_final) > 0.07:
                    vel_k = 1.35
                    # print('speed up!!')

                if rep_sum > 1:
                    vel_k = 0.4
                    print('slow down!!')

                gradient = att_force + rep_force
                grad_len = np.linalg.norm(gradient)
                gradient = gradient / grad_len  # 归一化

                vel = vel_k * max_vel * (1 - np.exp(-grad_len / 5)) * gradient  # 世界坐标系下无人机的理论速度
                # vel = vel_k * max_vel * gradient

                if np.linalg.norm(vel) < 1 and k != len(reference_path) - 1:
                    print('vel too small')
                    break

                ##计算势能
                self.att_list.append(self.att_potential(curr_pos, goal_final))
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
                if dot_ < -0.9:
                    curr_potential = self.att_list[-1] + self.rep_list[-1]
                    next_potential = self.rep_potential(curr_pos + vel * self.delta_t, goal) + self.att_potential(
                        curr_pos + vel * self.delta_t, goal)
                    if next_potential - curr_potential > 0.2 and iters - last_iter > 5:
                        sign += 4.5

                if len(record) >= 5:
                    sign += sum(record[-5:]) / 2

                sign -= dot_
                if sign >= 3 and k != len(reference_path) - 1:
                    print('next goal!!!')
                    break

                if not self.APF_ASTAR:
                    if len(path) > 1:
                        last_pos = path[-2]

                        curr_cos = np.dot(vel, curr_pos - last_pos) / (
                                np.linalg.norm(curr_pos - last_pos) * np.linalg.norm(vel))

                        if curr_cos - 0.5 < 0:
                            rot_v = np.cross(vel, curr_pos - last_pos)
                            rot_v = rot_v / np.linalg.norm(rot_v)
                            rot_v *= (np.arccos(curr_cos) - np.pi / 3)
                            rot_mat = vec_to_rot_matrix(
                                torch.tensor(rot_v, dtype=torch.float32, device='cuda')).cpu().numpy()
                            vel = rot_mat @ vel

                if args.nav_map:  # and k > 1
                    # print('random interupt!!!')
                    collect_list = []
                    # vel += np.random.normal(8,1,(3,))

                    # next_pos = curr_pos + vel * self.delta_t
                    # path.append(next_pos)

                    curr_int = list(map(int, (curr_pos - np.array([x_min, y_min, z_min])) / inter_val))
                    goal_int = list(map(int, (goal_final - np.array([x_min, y_min, z_min])) / inter_val))

                    while curr_int != goal_int:
                        # print(1)
                        action = nav_map[curr_int[0], curr_int[1], curr_int[2]]
                        # curr_int +=action
                        curr_int = [x + y for x, y in zip(curr_int, action)]
                        collect_list.append(curr_int * inter_val + np.array([x_min, y_min, z_min]))

                    print('gs-dp初始路径搜索耗时：', time.time() - t1)
                    return path + collect_list
                    # print('collect_list:',collect_list)

                next_pos = curr_pos + vel * self.delta_t

                next_potential = self.rep_potential(next_pos, goal_final)  # + self.att_potential( next_pos, goal_final)
                curr_potential = self.rep_list[-1]
                if next_potential - curr_potential > 30:
                    goal_next_num = 1
                    track_weight *= 0.3
                    continue

                path.append(next_pos)
                curr_pos = next_pos

            print('turn:', iters)
        print('耗时：', time.time() - t1)
        return path


def inbounds(point):
    for x, size in zip(point, occupied.shape):
        if x < 0 or x >= size: return False
    return True


def compute_potential(curr, goal, obs_set, tree):
    # curr是网格坐标，goal是世界坐标
    eta = 35
    zeta = 20
    n = 0.5
    rho = 0.35
    # center_curr = -1 + curr[0] / 15 + 1 / 30, -1 + curr[1] / 15 + 1 / 30, curr[2] / 30 + 1 / 60
    center_curr = np.array([x_min, y_min, z_min]) + np.array([curr[0], curr[1], curr[2]]) * inter_val

    def att_potential():
        return 0.5 * eta * (np.linalg.norm(center_curr - goal)) ** 2

    # indices = tree.query_ball_point(center_curr.tolist(), rho)
    indices = []
    distance, index = tree.query(center_curr.tolist())

    def rep_potential():
        rep_sum = 0
        for i in indices:
            obs = obs_set[i]
            # for obs in obs_set:
            dist = np.linalg.norm(obs - center_curr)
            if dist < rho:
                rep_sum += 0.5 * zeta * (1 / dist - 1 / rho) ** 2 * (
                    np.linalg.norm(center_curr - goal)) ** n
        return rep_sum

    return min(1 / distance, 10) + att_potential()


def compute_all_grid_cost(occupied, goal, goal_world, obs_set):
    tree = KDTree(obs_set)

    open_list = [(goal, 0)]
    all_cost = np.full(occupied.shape, float("inf"), dtype=float)  # 存储每个节点的代价
    visit = np.full(occupied.shape, False, dtype=bool)
    visit[goal] = True
    neighbors = [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1),
                 (-1, 1, 0), (1, -1, 0),
                 (-1, -1, 0), (1, 1, 0),
                 (0, 1, 1), (0, 1, -1),
                 (0, -1, -1), (0, -1, 1),
                 (1, 0, 1), (-1, 0, -1),
                 (-1, 0, 1), (1, 0, -1)]

    while open_list:
        current_node = open_list[0]  # 取节点
        open_list = open_list[1:]
        current = current_node[0]  # 位置
        current_cost = current_node[1]  # 代价

        for i, j, k in neighbors:
            neighbor = (current[0] + i, current[1] + j, current[2] + k)

            if not inbounds(neighbor):
                continue

            if occupied[neighbor]:
                continue

            if visit[neighbor]:
                continue

            visit[neighbor] = True

            # dist_cost = current_cost + 1
            dist_cost = 0
            potential_cost = compute_potential(neighbor, goal_world, obs_set, tree)
            total_cost = dist_cost + potential_cost
            all_cost[neighbor] = total_cost

            node = (neighbor, total_cost)
            open_list.append(node)

    all_cost[goal] = 0

    return all_cost


def build_nav_map_re(occupied, goal, goal_world, xyz_mask):
    visit = np.full(occupied.shape, False, dtype=bool)
    neighbors = [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1),
                 (-1, 1, 0), (1, -1, 0),
                 (-1, -1, 0), (1, 1, 0),
                 (0, 1, 1), (0, 1, -1),
                 (0, -1, -1), (0, -1, 1),
                 (1, 0, 1), (-1, 0, -1),
                 (-1, 0, 1), (1, 0, -1)]

    all_cost = compute_all_grid_cost(occupied, goal, goal_world, xyz_mask)
    nav_map = np.full(list(occupied.shape) + [3], (0, 0, 0))
    for i in range(occupied.shape[0]):
        for j in range(occupied.shape[1]):
            for k in range(occupied.shape[2]):
                if occupied[(i, j, k)] == False:
                    curr_cost = float("inf")
                    for x, y, z in neighbors:
                        neighbor = (i + x, j + y, k + z)

                        if not inbounds(neighbor):
                            continue

                        if occupied[neighbor]:
                            continue

                        if visit[neighbor]:
                            continue

                        visit[neighbor] = True

                        if all_cost[neighbor] < curr_cost:
                            curr_cost = all_cost[neighbor]
                            nav_map[(i, j, k)] = [x, y, z]

    return nav_map


def compute_grid_cost(occupied):
    coords = np.where(occupied == True)
    coords_array = np.array([list(zip(*coords))])
    obs_center = coords_array * inter_val + min_corner

    coords = np.where(occupied == False)
    coords_array = np.array([list(zip(*coords))])
    free_center = coords_array * inter_val + min_corner

    eta = 5
    zeta = 8
    n = 0.5
    rho = 0.2
    potential_cost = np.full(occupied.shape, float("inf"), dtype=float)
    for grid in free_center:
        rep_sum = 0
        for obs in obs_center:
            dist = np.linalg.norm(obs - grid)
            if dist < 0.2:
                rep_sum += 0.5 * zeta * (1 / dist - 1 / rho) ** 2 * (
                    np.linalg.norm(grid - goal)) ** n
        zuobiao = (grid - min_corner) / inter_val
        zuobiao = zuobiao.astype(int)
        potential_cost[zuobiao] = rep_sum

    return potential_cost


def build_nav_map(occupied, goal, goal_world, xyz_mask):
    # 应当使用欧式距离作为度量,
    # 每个节点应该存储距离终点的代价值（步数）
    all_cost = compute_all_grid_cost(occupied, goal, goal_world, xyz_mask)
    ####all_cost = np.full(occupied.shape, 10, dtype=float)####
    neighbors = [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1),
                 (-1, 1, 0), (1, -1, 0),
                 (-1, -1, 0), (1, 1, 0),
                 (0, 1, 1), (0, 1, -1),
                 (0, -1, -1), (0, -1, 1),
                 (1, 0, 1), (-1, 0, -1),
                 (-1, 0, 1), (1, 0, -1)]

    # coords = np.where(occupied == True)
    # coords_array = np.array([list(zip(*coords))])
    # obs_center = coords_array * np.array([1 / 15, 1 / 15, 1 / 30]) + np.array([-1 + 1 / 30, -1 + 1 / 30, 1 / 60])

    obs_set = xyz_mask

    nav_map = np.full(list(occupied.shape) + [3], (0, 0, 0))
    open_list = set()
    open_list.add((goal, 0))

    visit = np.full(occupied.shape, float("inf"), dtype=float)  # 存储每个节点的代价
    visit[goal] = 0

    while len(open_list) > 0:

        current_node = open_list.pop()

        current = current_node[0]  # 位置
        current_cost = current_node[1]  # 代价

        for i, j, k in neighbors:
            neighbor = (current[0] + i, current[1] + j, current[2] + k)

            if not inbounds(neighbor):
                continue

            if occupied[neighbor]:
                continue

            # if neighbor in close_set:
            #     continue

            # 若已经访问过，看看当前代价是否更小

            # 距离代价+势能代价
            # dist_cost = current_cost + 1
            # potential_cost = compute_potential(neighbor,goal_world,obs_set)
            # total_cost = dist_cost + potential_cost
            total_cost = current_cost + all_cost[neighbor]  # 获取节点的总势能值

            if total_cost < visit[neighbor]:
                visit[neighbor] = total_cost
                # neighbor节点的action应当更新 指向为current：
                nav_map[neighbor] = [-i, -j, -k]
                node = (neighbor, total_cost)
                if node not in open_list:
                    open_list.add(node)

    return nav_map


def astar(occupied, start, goal):
    def heuristic(a, b):
        # return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
        return abs(b[0] - a[0]) + abs(b[1] - a[1]) + abs(b[2] - a[2])  # 曼哈顿距离

    neighbors = [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1)]

    close_set = set()

    came_from = {}
    gscore = {start: 0}

    assert not occupied[start]
    assert not occupied[goal]

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))  # heapq-小顶堆

    while open_heap:
        current = heapq.heappop(open_heap)[1]  # 堆中取最小代价节点

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            assert current == start
            data.append(current)
            return list(reversed(data))

        close_set.add(current)  # 保存已经访问过的节点集合

        for i, j, k in neighbors:
            neighbor = (current[0] + i, current[1] + j, current[2] + k)
            if not inbounds(neighbor):
                continue

            if occupied[neighbor]:
                continue

            tentative_g_score = gscore[current] + 1
            # neighbor不存在则返回inf  ！！！
            if tentative_g_score < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                # f(n) = h(n) + g(n)
                fscore = tentative_g_score + heuristic(neighbor, goal)
                node = (fscore, neighbor)
                if node not in open_heap:
                    heapq.heappush(open_heap, node)

    raise ValueError("Failed to find path!")


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
    parser.add_argument("--APF_ASTAR", default=False, type=bool)
    parser.add_argument("--nav_map", default=False, type=bool)

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

    if 'data' in args.model_path or 'stonehenge_size2' in args.model_path:
        blend_file = '../blender_scenes/stonehenge_test1121.blend'
        filename = '../outputs/stonehenge/'
        if 'stonehenge_size2' in args.model_path:
            blend_file = '../blender_scenes/stonehenge_size2.blend'

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
        blend_file = '../blender_scenes/sphere.blend'
        filename = '../outputs/sphere/'
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

    # # 定义空间网格大小
    # grid_size = 0.001  # 网格大小为0.1
    #
    # # 将点云数据映射到空间网格中
    # grid_indices = np.floor(xyz_mask / grid_size).astype(int)
    #
    # # 创建空间网格
    # grid = {}
    # for i, point in enumerate(xyz_mask):
    #     grid_index = tuple(grid_indices[i])
    #     if grid_index in grid:
    #         grid[grid_index][0] += point
    #         grid[grid_index][1] += density_mask[i]
    #         grid[grid_index][2] += scales_mask[i]
    #         grid[grid_index][3] += 1
    #
    #     else:
    #         grid[grid_index] = [point,density_mask[i],scales_mask[i],1]
    #
    # # 计算每个网格内点的均值作为新的点
    #
    # xyz_mask = []
    # scales_mask = []
    # density_mask = []
    #
    # for grid_index, vals in grid.items():
    #         merged_xyz = vals[0] / vals[3]#
    #         merged_den = vals[1] / vals[3]
    #         merged_sca = vals[2] / vals[3]
    #
    #         xyz_mask.append(merged_xyz)
    #         scales_mask.append(merged_den)
    #         density_mask.append(merged_sca)
    #
    # xyz_mask = np.array(xyz_mask)
    # scales_mask = np.array(scales_mask)
    # density_mask = np.array(density_mask)

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
    # start= np.array([-0.036, 0.5531, 0.1637])  # 起点坐标 0.0829, 0.6033, 0.12724 #-0.2556, 0.7903, 0.2169
    # goal= np.array([0.1456, -0.7668, 0.17111])  # 终点坐标 -0.5554, -0.9017, 0.3058 # 0.16767, -0.9132, 0.2788 # 0.7506, -0.2858, 0.16455
    init_pose = np.array([0, 0, 0])  ##初始姿态（欧拉角）
    end_pose = np.array([0, 0, 0])  ##目标姿态（欧拉角）
    init_rates = np.zeros(3)  ##初始角速度和线速度

    #############################
    side = 31  # 生成立方体的数量+1
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

    a_star_time = time.time()

    # 计算偏移量
    offset = (xyz_mask - np.array([x_min, y_min, z_min])) / inter_val
    # 取整
    offset_int = np.minimum(side - 2, offset.astype(int))
    # 更新occupy
    occupy[offset_int[:, 0], offset_int[:, 1], offset_int[:, 2], 0] += density_mask[:, 0]  # 高斯密度
    occupy[offset_int[:, 0], offset_int[:, 1], offset_int[:, 2], 1] += 1  # 高斯数量

    # for i in range(xyz_mask.shape[0]):
    #     loc = (np.array([xyz_mask[i][0], xyz_mask[i][1], xyz_mask[i][2]]) - np.array([x_min, y_min, z_min])) / inter_val
    #     xx, yy, zz = min(side-2,int(loc[0])), min(side-2,int(loc[1])), min(side-2,int(loc[2]))
    #     occupy[xx][yy][zz][0]+=density_mask[i][0] #高斯密度
    #     occupy[xx][yy][zz][1]+=1 #高斯数量

    occupied = (occupy[..., 0] / (occupy[..., 1] + 0.0001)) > 0.1  # 计算每个立方体平均密度,密度大于0.2的视为被占据
    # occupied = occupy[..., 1] > 0

    neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    occupied_v = occupied
    for current in np.argwhere(occupied == True):
        for i in neighbors:
            neighbor = current + i
            if inbounds(neighbor):
                occupied[neighbor[0], neighbor[1], neighbor[2]] = True

    # a_star_time = time.time()
    # 计算起点和终点的网格坐标
    min_corner = np.array([x_min, y_min, z_min])
    loc_s = (start - min_corner) / inter_val
    loc_g = (goal - min_corner) / inter_val

    start_a = tuple(loc_s.astype(int))
    end_a = tuple(loc_g.astype(int))
    end_build_time = time.time()
    print('build occupy map cost time:', end_build_time - a_star_time)
    path_astar = astar(occupied, start_a, end_a)
    print('a-star search cost time:', time.time() - end_build_time)

    ############################# --3mon 31day
    nav_map_time = time.time()
    nav_map = build_nav_map(occupied_v, end_a, goal, xyz_mask)
    print('build_nav_map time: ', time.time() - nav_map_time)

    #####################################

    # 搜索到的路径变回世界坐标
    path_world = np.array([x_min, y_min, z_min]) + np.array(path_astar, dtype=float) * inter_val
    prev_smooth = np.concatenate((np.expand_dims(path_world[0, :], axis=0), path_world[:-1, :]), axis=0)
    next_smooth = np.concatenate((path_world[1:, :], np.expand_dims(path_world[-1, :], axis=0)),
                                 axis=0)  # next_smooth = np.concatenate([path_world[1:, :], path_world[-1, None, :], ], dim=0)

    ref_path = path_world

    sign_a, way_points_list_a = execute(ref_path, 'A*', False)  # 返回是否受到扰动， 路标点

    #####ref_path = path_world
    ref_curv = []
    for i in range(len(ref_path) - 2):
        a, b, c = ref_path[i:i + 3]
        ref_curv.append(cal_cueve(a, b, c))

    # print('参考路径长度：',len(ref_path))
    # print('参考路径曲率：',ref_curv)
    print('初始路径A*与目标点偏差：', np.linalg.norm(ref_path[-1] - goal))

    #####状态初始化#######
    start_state = np.hstack((start, init_rates, init_pose, init_rates))  ##初始state
    end_state = np.hstack((goal, init_rates, end_pose, init_rates))  ##结束state

    traj = Planner(start_state, end_state, xyz_mask, scales_mask, density_mask, delta_t, ref_path, args.APF_ASTAR)
    # traj_new = Planner(start_state, end_state, xyz_mask, scales_mask, density_mask, delta_t, ref_path, 1) 比较
    ###过滤掉参考路径中曲率小的点
    ref_path_filter = [0] * len(ref_path)
    for i in range(len(ref_curv)):
        if ref_curv[i] > 0.05 or i > len(ref_path) * 0.7:
            ref_path_filter[i + 1] = 1

    ref_path_filter = ref_path[np.array(ref_path_filter) > 0]
    #### 二次过滤 rep-potential超过0.2的舍去
    ref_path_filter_twice = []
    for i in ref_path_filter:
        if traj.rep_potential(i, goal) < 0.001 and np.linalg.norm(i - goal) > 0.055:
            ref_path_filter_twice.append(i)

    if not args.APF_ASTAR:
        traj.ref_path = np.array(ref_path_filter_twice)

    #####traj.ref_path = ref_path_filter
    # print('过滤后长度：',len(ref_path_filter_twice))

    path = traj.path_searching(start, goal)
    sign_dp, way_points_list_dp = execute(path, 'dp', False)  ##执行路径跟踪

    ##为了比较
    # traj_new.ref_path= np.array(ref_path_filter_twice)
    # path_new = traj_new.path_searching(start, goal)
    # path.insert(0,(path[0]+path[1])/2)
    print('初始路径DP与目标点偏差：', np.linalg.norm(path[-1] - goal))
    # a_star_filter = []
    # for i in np.array(ref_path_filter_twice):
    #     a_star_filter.append(traj.rep_potential(i,goal))
    # plt.plot(a_star_filter, color='r', label='rep_potential')
    # plt.xlabel('way-points',fontsize=12,loc = 'right')
    # plt.ylabel('values',fontsize=12,loc = 'top')
    # plt.title('a_star_filter rep_potential')
    # # 显示图形
    # plt.show()

    # plt.figure(figsize=(10, 8))
    # plt.plot(traj.att_list, color='#7E99F4')

    # plt.plot(np.sqrt(2 * np.array(traj.att_list) / traj.eta),label='APF-A*-Improved')
    # ###plt.plot(np.sqrt(2 * np.array(traj_new.att_list) / traj_new.eta), label='APF-A*-Basic')
    #
    # plt.xlabel('Node', fontsize=12, loc='center')
    # plt.ylabel('Distance to goal (m)', fontsize=12, loc='center')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # # plt.plot(traj.rep_list, color='#CC7C71', label='Rep potential energy')
    # # plt.plot(np.array(traj.att_list)+np.array(traj.rep_list), color='#7AB656', label='Total potential energy')
    # #plt.subplot( 1,2, 2)
    #
    # plt.legend(fontsize=12)
    # plt.tick_params(labelsize=10)
    #
    # # 显示图形
    # plt.show()

    # plt.figure(figsize=(10, 6), dpi=80)
    # # Plotting the data with labels for the legend
    # plt.plot(traj.att_list, label='att_potential', color='red')
    # plt.plot(traj.rep_list, label='rep_potential', color='green')
    # plt.plot(np.array(traj.att_list)+np.array(traj.rep_list), label='total_potential', color='blue')
    #
    # # Adding grid lines
    # plt.grid(True, linestyle='--', alpha=0.6)
    #
    # # Adding labels and title
    # plt.xlabel('iters')
    # plt.ylabel('Values')
    # plt.title('Enhanced Figure')
    #
    # # Placing the legend outside of the plot area
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #
    # plt.show()

    ################################
    # ref_rep = []
    # for i in ref_path:
    #     ww,ss = traj.repulsive_force(i,goal)
    #     #if ss < 0.05 :
    #     ref_rep.append(ss)
    # plt.plot(ref_rep, color='g', label='ref_rep_potential')
    # plt.legend(fontsize=12)
    # plt.xlabel('iters', fontsize=12, loc='right')
    # plt.ylabel('values', fontsize=12, loc='top')
    # # 显示图形
    # plt.show()
    #################################

    #####visuialize###APF-ASTAR(SMOOTH)
    if not (args.APF_ASTAR or args.nav_map):
        filename_sm = filename + 'potential_visual.json'
        exp_name = filename_sm
        poses = []
        pose_dict = {}
        with open(filename_sm, "w+") as f:
            for pos in path:
                pose = np.zeros((4, 4))
                pose[:3, :3] = np.identity(3)
                pose[:3, 3] = pos
                pose[3, 3] = 1
                poses.append(pose.tolist())
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

    elif args.APF_ASTAR:
        ##APF-ASTAR
        print('APF-ASTAR!!')
        filename_aa = filename + 'APF-ASTAR.json'
        poses = []
        pose_dict = {}
        with open(filename_aa, "w+") as f:
            for pos in path:
                pose = np.zeros((4, 4))
                pose[:3, :3] = np.identity(3)
                pose[:3, 3] = pos
                pose[3, 3] = 1
                poses.append(pose.tolist())
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

    else:
        filename_nav = filename + 'nav_map.json'
        exp_name = filename_nav
        poses = []
        pose_dict = {}
        if sign_dp:
            path = way_points_list_dp
        with open(filename_nav, "w+") as f:
            for pos in path:
                pose = np.zeros((4, 4))
                pose[:3, :3] = np.identity(3)
                pose[:3, 3] = pos
                pose[3, 3] = 1
                poses.append(pose.tolist())
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

    ##pure a*
    filename_ref = filename + 'a_star_ref.json'
    poses = []
    pose_dict = {}
    ref_path_filter = np.vstack((start.reshape(1, 3), ref_path_filter, goal.reshape(1, 3)))  ##A-star-filter
    ref_path = np.vstack((start.reshape(1, 3), ref_path))  ##A-star
    ref_path_filter_twice = np.array(ref_path_filter_twice)
    if sign_a:
        ref_path = way_points_list_a

    with open(filename_ref, "w+") as f:
        for pos in ref_path:
            pose = np.zeros((4, 4))
            pose[:3, :3] = np.identity(3)
            pose[:3, 3] = pos
            pose[3, 3] = 1
            poses.append(pose.tolist())
        pose_dict["poses"] = poses
        json.dump(pose_dict, f, indent=4)
        # ref_path_filter

    ##############画图###############################
    if 'sphere' in exp_name:
        latest_init = './outputs/sphere/nav_map.json'
        with open(latest_init, 'r') as f:
            meta = json.load(f)

        points = (np.array(meta["poses"])[..., 3:][:, :3]).reshape(-1, 3)

        print('apf_nerf waypoints的数量：', len(points))
        dist = 99
        sphere_info = './viz_utils/sphere_info.json'
        with open(sphere_info, 'r') as f:
            meta = json.load(f)
            sphere_list = meta['loc']

        dist_to_obs = []
        dist_to_obs_a = []

        for i in points:
            dist = 99
            for j in sphere_list:
                dist = min(dist, np.linalg.norm(np.array(j[:3]) - i) - j[-1])
            dist_to_obs.append(dist)

        plt.figure(figsize=(7.5, 6))
        plt.plot(np.linspace(0, 0.0004, len(dist_to_obs)), np.array(dist_to_obs), color='#7AB656',
                 label='GS-DP')  # 1.16107s
        plt.ticklabel_format(style='sci', scilimits=(0, 1), axis='x')
        plt.hlines(0.02, -0.1, 1, '#CC7C71', '--', 'Safety distance')
        plt.xlabel('time(s)', fontsize=15)
        plt.ylabel('Distance(m)', fontsize=15, loc='top')
        current_yticks = plt.gca().get_yticks()
        current_yticklabels = plt.gca().get_yticklabels()
        # 确保0.02被添加为刻度，而不替换现有刻度
        new_yticks = list(current_yticks)  # 复制现有的刻度位置
        if 0.02 not in new_yticks:
            new_yticks = sorted(new_yticks + [0.02])  # 添加0.02并排序

        # 设置新的y轴刻度位置
        plt.yticks(new_yticks, fontsize=13)
        plt.xticks(fontsize=13)
        plt.ylim([0, 0.42])
        plt.xlim([-0.03, 0.00042])
        # plt.title('Distance to the nearest obstacles',fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig('Distance_to_obs.png')
        # 显示图形
        plt.show()
        # print('min_dist:',dist)

    subprocess.run(['blender', blend_file, '-P', 'viz_data_blend.py', '--', exp_name, '0.2'])
    subprocess.run(['blender', blend_file, '-P', 'viz_data_blend.py', '--', filename_ref, '0.2'])  # A*
