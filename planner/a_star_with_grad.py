import numpy as np
import heapq
import json
from utils.system_utils import searchForMaxIteration
from demos.render import *


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
    def __init__(self, start_state, end_state, xyz, scales, density, ):
        self.start_state = start_state
        self.end_state = end_state

        #### 定义一些参数
        self.eta = 15  # 引力增益系数，15
        self.zeta = 20  # 斥力增益系数，20
        self.rho0 = 0.05  # 障碍物影响范围，0.05
        self.goal_tol = 0.03  # 目标点容差，0.01
        self.max_iter = 35  # 最大迭代次数，500
        self.n = 0.5  # 斥力修正因子 取值范围[0,1]
        ###高斯的一些参数
        self.density = density
        self.xyz = xyz
        self.scales = scales
        self.lr = 0.01
        self.epochs_init = 2500
        self.epoch = None

        self.att_list = []
        self.rep_list = []
        self.ref_path = None

    def params(self):
        return [self.states]

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)

        try:
            for it in range(self.epochs_init):
                opt.zero_grad()
                self.epoch = it
                loss = self.rep_potential(self.states, self.end_state[:3])
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

    def rep_potential(self, curr_list, goal):
        rep_sum = 0
        curr_list_np = curr_list.clone().detach().cpu().numpy()
        for curr in curr_list_np:
            for i in range(self.xyz.shape[0]):
                dist = np.linalg.norm(curr - self.xyz[i])
                if dist <= self.rho0 + self.scales[i][0]:
                    rep_sum += 0.5 * self.zeta * (1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                        np.linalg.norm(curr - goal)) ** self.n
        return torch.tensor(rep_sum, dtype=torch.float32, requires_grad=True)

    # 计算引力
    def attractive_force(self, curr, goal):
        # 引力与点到目标的距离平方成正比

        # 返回引力势场梯度力
        return -self.eta * (curr - goal)

    def repulsive_force(self, point, goal):
        # 斥力与点到障碍物的距离成反比
        # 如果距离大于阈值，则斥力为零
        grad = np.array([0., 0., 0.])
        rep_sum = 0
        for i in range(self.xyz.shape[0]):
            dist = np.linalg.norm(point - self.xyz[i])
            if dist <= self.rho0 + self.scales[i][0]:
                rep_sum += 0.5 * self.zeta * (1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                    np.linalg.norm(point - goal)) ** self.n
                grad = grad + np.linalg.norm(point - goal) ** self.n * (
                            -self.zeta * self.density[i][0] * (1 / dist - 1.0 / (self.rho0 + self.scales[i][0])) * (
                                1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (
                                   1 / dist - 1 / (self.rho0 + self.scales[i][0])) ** 2 * (
                                   point - goal) * np.linalg.norm(point - goal) ** (self.n - 2)
                # grad = grad + np.linalg.norm(point - goal) ** self.n * (-self.zeta * self.density[i][0] * (1 / dist - 1.0 / self.rho0 ) * (1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (1 / dist - 1 / (self.rho0 )) ** 2 * (point - goal) * np.linalg.norm(point - goal) ** (self.n - 2)

        # 返回斥力势场梯度力以及势能大小
        return -grad, rep_sum * 1e-4

    def astar(self, occupied, start, goal):
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

    def a_star_init(self):
        #############################
        side = 21  # 生成立方体的数量+1
        # 场景边界
        x_min, x_max = self.xyz[:, 0].min(), self.xyz[:, 0].max()
        y_min, y_max = self.xyz[:, 1].min(), self.xyz[:, 1].max()
        z_min, z_max = self.xyz[:, 2].min(), self.xyz[:, 2].max()

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
        inter_val = (torch.tensor([x_max, y_max, z_max], dtype=torch.float) - torch.tensor([x_min, y_min, z_min],
                                                                                           dtype=torch.float)) / (
                                side - 1)
        occupy = np.zeros([i - 1 for i in coods.shape[:-1]] + [2])

        for i in range(self.xyz.shape[0]):
            loc = (np.array([self.xyz[i][0], self.xyz[i][1], self.xyz[i][2]]) - np.array(
                [x_min, y_min, z_min])) / inter_val
            xx, yy, zz = min(side - 2, int(loc[0])), min(side - 2, int(loc[1])), min(side - 2, int(loc[2]))
            occupy[xx][yy][zz][0] += density_mask[i][0]  # 高斯密度
            occupy[xx][yy][zz][1] += 1  # 高斯数量

        # print(occupy)
        occupied = (occupy[..., 0] / (occupy[..., 1] + 0.0001)) > 0.1  # 计算每个立方体平均密度,密度大于0.2的视为被占据
        # occupied = occupy[..., 1] > 0
        # 计算起点和终点的网格坐标
        loc_s = (torch.tensor([start[0], start[1], start[2]], dtype=torch.float32) - torch.tensor([x_min, y_min, z_min],
                                                                                                  dtype=torch.float32)) / inter_val
        loc_g = (torch.tensor([goal[0], goal[1], goal[2]], dtype=torch.float32) - torch.tensor([x_min, y_min, z_min],
                                                                                               dtype=torch.float32)) / inter_val
        xx_s, yy_s, zz_s = int(loc_s[0]), int(loc_s[1]), int(loc_s[2])
        xx_g, yy_g, zz_g = int(loc_g[0]), int(loc_g[1]), int(loc_g[2])

        start_a = (xx_s, yy_s, zz_s)
        end_a = (xx_g, yy_g, zz_g)
        path_astar = self.astar(occupied, start_a, end_a)
        # 搜索到的路径变回世界坐标
        path_world = torch.tensor([x_min, y_min, z_min], dtype=torch.float) + torch.tensor(path_astar,
                                                                                           dtype=torch.float) * inter_val

        # ref_path = np.vstack(( path_world,goal.reshape(1,3) ))
        prev_smooth = torch.cat([path_world[0, None, :], path_world[:-1, :]], dim=0)
        next_smooth = torch.cat([path_world[1:, :], path_world[-1, None, :], ], dim=0)

        self.ref_path = (prev_smooth + next_smooth + path_world) / 3
        self.states = self.ref_path.clone().detach().requires_grad_(True)
        print('参考路径长度：', len(self.ref_path))
        # ref_path = path_world


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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
    ##初步筛选##
    # xyz_mask = (xyz[mask]).detach().cpu().numpy()
    # scales_mask = (scales[mask]).detach().cpu().numpy()
    # density_mask = (density[mask]).detach().cpu().numpy()

    xyz_mask = xyz.detach().cpu().numpy()
    scales_mask = scales.detach().cpu().numpy()
    density_mask = density.detach().cpu().numpy()

    # 定义起点和终点 0.0330 -0.793251 m  0.251245 m -0.69881 m
    # start=-0.54, -0.76, 0.17，end = 0.08,0.60, 0.12.png
    start = np.array([-0.2556, 0.7903, 0.2169])  # 起点坐标 0.0829 0.6033 0.12724 -0.2556, 0.7903, 0.2169
    goal = np.array([-0.5554, -0.9017, 0.3058])  # 终点坐标 -0.5554, -0.9017, 0.3058
    init_pose = np.array([0, 0, 0])  ##初始姿态（欧拉角）
    end_pose = np.array([0, 0, 0])  ##目标姿态（欧拉角）
    init_rates = np.zeros(3)  ##初始角速度和线速度

    #####状态初始化#######
    start_state = np.hstack((start, init_rates, init_pose, init_rates))  ##初始state
    end_state = np.hstack((goal, init_rates, end_pose, init_rates))  ##结束state

    traj = Planner(start_state, end_state, xyz_mask, scales_mask, density_mask)
    traj.a_star_init()
    traj.learn_init()
    ###过滤掉参考路径中斥力大的点####
    # ref_path_filter = []
    # for i in ref_path:
    #     ww, ss = traj.repulsive_force(i, goal)
    #     if ss < 0.05:
    #         ref_path_filter.append(i)
    #
    # traj.ref_path = np.array(ref_path_filter)
    #
    # print('过滤后路径长度：', len(ref_path_filter))
    # print(ref_path_filter)

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

    path = traj.states
    #####visuialize###
    filename = 'potential_visual.json'
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

    # filename_ref = 'a_star_ref.json'
    # poses = []
    # pose_dict = {}
    # with open(filename_ref, "w+") as f:
    #     for pos in ref_path_filter:
    #         pose = np.zeros((4, 4))
    #         pose[:3, :3] = np.identity(3)
    #         pose[:3, 3] = pos
    #         pose[3, 3] = 1
    #         poses.append(pose.tolist())
    #     pose_dict["poses"] = poses
    #     json.dump(pose_dict, f, indent=4)
    # ref_path_filter
