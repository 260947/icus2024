import numpy as np
import json
import time
from utils.system_utils import searchForMaxIteration
from nav.quad_helpers import  next_rotation
import matplotlib.pyplot as plt
from demos.render import *
#from nav import vec_to_rot_matrix, rot_matrix_to_vec



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
    def __init__(self, start_state, end_state, xyz,scales,density,delta_t):
        self.start_state = start_state
        self.end_state = end_state
        self.delta_t = delta_t ##时间间隔
        #### 定义一些参数
        self.eta = 15  # 引力增益系数，15
        self.zeta = 20  # 斥力增益系数，20
        self.rho0 = 0.05  # 障碍物影响范围，0.05
        self.goal_tol = 0.02  # 目标点容差，0.01
        self.max_iter = 500  # 最大迭代次数
        self.n = 0.5  # 斥力修正因子 取值范围[0,1]
        ###高斯的一些参数
        self.density = density.detach().cpu().numpy()
        self.xyz = xyz.detach().cpu().numpy()
        self.scales = scales.detach().cpu().numpy()

    #计算引力
    def attractive_force(self,curr,goal):
        # 引力与点到目标的距离平方成正比
        #force = eta * sum((goal - curr)**2)，引力大小
        #返回引力势场梯度力
        return -self.eta * (curr - goal)

    def repulsive_force(self,point,goal):
        # 斥力与点到障碍物的距离成反比
        # 如果距离大于阈值，则斥力为零
        #valid_gs = np.array([0.,0.,0.])
        grad=np.array([0.,0.,0.])
        for i in range(self.xyz.shape[0]):
            dist = np.linalg.norm(point-self.xyz[i])
            if dist <=self.rho0+self.scales[i][0]:
                #valid_gs = valid_gs + self.xyz[i] - goal
            #if dist <= self.rho0 :
                grad=grad + np.linalg.norm(point-goal)**self.n * (-self.zeta * self.density[i][0]*(1/dist - 1.0 / (self.rho0+self.scales[i][0]))* (1.0 / dist**3)*(point - self.xyz[i])) + 0.5 * self.n * self.zeta*(1/dist - 1/(self.rho0+self.scales[i][0]) )**2 * (point-goal) * np.linalg.norm(point-goal)**(self.n-2)
                #grad = grad + np.linalg.norm(point - goal) ** self.n * (-self.zeta * self.density[i][0] * (1 / dist - 1.0 / self.rho0 ) * (1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (1 / dist - 1 / (self.rho0 )) ** 2 * (point - goal) * np.linalg.norm(point - goal) ** (self.n - 2)

        #返回斥力势场梯度力
        return -grad #, valid_gs

    def plot_point_num(self,path):
        inter_val = 0.05
        res = []
        for i in path:
            count = 0
            for j in path:
                if np.linalg.norm(i - j) < inter_val:
                    count += 1
            res.append(count)
        plt.plot(res)
        plt.pause(100)
        #return res

    def stuck_in_local_minimum(self,path):
        # inter_val = 0.05
        # res = []
        # for i in range(len(path)):
        #     count = 0
        #     for j in path:
        #         if np.linalg.norm(path[i] - j) < inter_val:
        #             count += 1
        #     if count > 20 :
        #         return i
        #     res.append(count)
        # return -1
        #res[i] / i
        sum_1,sum_2=0,0
        i,j=-5,2
        pre = path[-1]
        pre_2 = path[0]
        for k in range(3):
            sum_1+=np.linalg.norm(pre - path[i])
            pre = path[i]
            i-=4

            sum_2+=np.linalg.norm(pre_2 - path[j])
            pre_2 = path[j]
            j+=2
        if sum_1  <   0.85*sum_2 :
            print('极小值！！！！！')

        return sum_1  <  0.85*sum_2






    def path_searching(self,start,goal):
        max_vel = 4 #最大步长
        path=[start]
        path_reverse = [goal]
        iters=0
        curr_pos = path[-1]
        curr_pos_2 = path_reverse[-1]
        last_iter,last_iter_2=0,0 ################################
        t1 = time.time()
        ##while np.linalg.norm(curr_pos - goal) > self.goal_tol and iters < self.max_iter:
        while  iters < self.max_iter:

            if np.linalg.norm(curr_pos - goal) <= self.goal_tol:
                print('start to goal')
                print('len:',len(path))
                print('耗时：', time.time() - t1)
                return path

            iters+=1
            ##计算斥力和引力
            att_force = self.attractive_force(curr_pos,goal)
            rep_force = self.repulsive_force(curr_pos, goal)
            #att_force_2, rep_force_2 = self.attractive_force(curr_pos_2, start), self.repulsive_force(curr_pos_2, start)

            gradient = att_force + rep_force
            grad_len = np.linalg.norm(gradient)
            gradient = gradient/grad_len#归一化
            vel = max_vel * (1 - np.exp(-grad_len / 5 )) * gradient   # 世界坐标系下无人机的理论速度


            ###处理局部极值###
            attr_v = att_force / np.linalg.norm(att_force)
            rep_v = rep_force / (np.linalg.norm(rep_force) + 0.000001)
            cross_ = np.cross(attr_v, rep_v)  #
            dot_ = np.dot(attr_v, rep_v)

            #if dot_ < -0.8:
            #if np.linalg.norm(curr_pos - goal) >  2 * self.goal_tol:
                #if np.dot(valid_gs, curr_pos - goal) > 0  and  len(path) > 15 :
                        # path[:] = path[:local_point+5]
                        # curr_pos = path[-1]
                        # ##重新计算斥力和引力
                        # att_force_new = self.attractive_force(curr_pos, goal)
                        # rep_force_new, valid_gs_new = self.repulsive_force(curr_pos, goal)
                        # attr_v = att_force_new / np.linalg.norm(att_force_new)
                        # rep_v = rep_force_new / (np.linalg.norm(rep_force_new) + 0.000001)
                        # dot_ = np.dot(attr_v, rep_v)
            if  np.linalg.norm(curr_pos - goal) >  2 * self.goal_tol and ( dot_ < -0.95 or  ( len(path) > 15 and self.stuck_in_local_minimum(path) ) ):  #or (len(path)>=3 and np.linalg.norm(path[-3] - path[-1]) <= 0.005)


                min_distur = (-dot_/(np.linalg.norm(attr_v))**2)*rep_v + attr_v #和斥速度垂直，与吸速度同一大致方向
                #min_distur = -min_distur #和斥速度垂直，与吸速度同一反方向


                dist = 0.05 ##0.05
                min_distur[:] =  -dist * min_distur/np.linalg.norm(min_distur)
                #min_distur = min_distur + 2*att_force

                gs_pos = (curr_pos + min_distur).reshape(1, 3)
                gs_scale = np.array([0.25]).reshape(1, 1) ##0.25
                gs_density = np.array([0.9]).reshape(1, 1) ##0.9
                if np.linalg.norm(gs_pos - goal) > 2.5 * self.goal_tol:
                    print('高斯位置：',gs_pos)
                    self.xyz = np.append(self.xyz, values=gs_pos, axis=0)
                    self.scales = np.append(self.scales, values=gs_scale, axis=0)
                    self.density = np.append(self.density, values=gs_density, axis=0)


                new_force = np.linalg.norm(curr_pos - goal) ** self.n * (-self.zeta * gs_density[0] * (1 / dist - 1.0 / (self.rho0 + gs_scale[0])) * (1.0 / dist ** 3) * (curr_pos - gs_pos[0])) + 0.5 * self.n * self.zeta * (1 / dist - 1 / (self.rho0 + gs_scale[0])) ** 2 * (curr_pos - goal) * np.linalg.norm(curr_pos - goal) ** (self.n - 2)
                new_force = -new_force
                new_force = 2 * np.linalg.norm(rep_force) * new_force / np.linalg.norm(new_force)

                gradient =  att_force +  rep_force + new_force ### 2 * np.linalg.norm(rep_force) * cross_/np.linalg.norm(cross_)

                grad_len = np.linalg.norm(gradient)
                gradient = gradient / grad_len  # 归一化
                vel =  max_vel * (1 - np.exp(-grad_len / 5 )) * gradient

                print('11垂直速度扰动！')

            if iters % 15==0:
                max_vel = max(3,0.9*max_vel)
                #print(path,'\n',path_reverse)

                print(path[-15:])

            next_pos = curr_pos + vel * self.delta_t
            ####next_pos_2 = curr_pos_2 + vel_2 * self.delta_t  #####
            ###############
            path.append(next_pos)
            ####path_reverse.append(next_pos_2) #####
            # if iters == 1:
            #     vel_list =  vel
            # if iters>1 :
            #     vel_list = np.vstack((vel_list, vel))
            curr_pos = next_pos
            ####curr_pos_2 = next_pos_2  #####

        # ##加上终止速度，0
        # vel_list = np.vstack((vel_list, np.zeros(3)))
        print('寻路失败！！')
        return path


    def calc_everything(self):
        # 这里增加了第一维1
        start_pos = self.start_state[None, 0:3]  # 位置
        start_v = self.start_state[None, 3:6]  # 速度(vx,vy,vz)
        start_R = self.start_state[6:15].reshape((1, 3, 3))
        start_omega = self.start_state[None, 15:]

        end_pos = self.end_state[None, 0:3]
        end_v = self.end_state[None, 3:6]
        end_R = self.end_state[6:15].reshape((1, 3, 3))
        end_omega = self.end_state[None, 15:]
        # 下一时刻的rotation
        next_R = next_rotation(start_R, start_omega, self.delta_t)  # 先左乘角速度的旋转矩阵再乘start_R，得到在世界坐标下的rotation

        # start, next, decision_states, last, end
        # 计算重力和驱动力的合力加速度，start_R @ torch.tensor([0,0,1.0])-在世界坐标系下的表达
        start_accel = start_R @ torch.tensor([0, 0, 1.0]) * self.initial_accel[0] + self.g
        next_accel = next_R @ torch.tensor([0, 0, 1.0]) * self.initial_accel[1] + self.g
        # 下2时刻速度
        next_vel = start_v + start_accel * self.delta_t
        after_next_vel = next_vel + next_accel * self.delta_t
        # 下3时刻位置
        next_pos = start_pos + start_v * self.delta_t
        after_next_pos = next_pos + next_vel * self.delta_t
        after2_next_pos = after_next_pos + after_next_vel * self.delta_t

        # position 2 and 3 are unused - but the atached roations are
        # 把计算到的pos和搜索到的Waypoint拼接，头部加入start_pos，尾部加入end_pos
        current_pos = torch.cat([start_pos, next_pos, after_next_pos, after2_next_pos, self.states[2:, :3], end_pos],
                                dim=0)

        prev_pos = current_pos[:-1, :]
        next_pos = current_pos[1:, :]
        # 计算相邻2个pos间的速度
        current_vel = (next_pos - prev_pos) / self.delta_t
        current_vel = torch.cat([current_vel, end_v], dim=0)  # 加上终止速度

        prev_vel = current_vel[:-1, :]
        next_vel = current_vel[1:, :]
        # 计算相邻2个速度间的加速度
        current_accel = (next_vel - prev_vel) / self.delta_t - self.g

        # 重复最后一个加速度  duplicate last accceleration - its not actaully used for anything (there is no action at last state)
        current_accel = torch.cat([current_accel, current_accel[-1, None, :]], dim=0)
        # 计算每个加速度的模长
        accel_mag = torch.norm(current_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration,加速度方向的单位向量
        z_axis_body = current_accel / accel_mag

        # remove states with rotations already constrained
        z_axis_body = z_axis_body[2:-1, :]
        # 与z轴夹角
        z_angle = self.states[:, 3]
        # ??????????
        in_plane_heading = torch.stack([torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)
        #   z_axis_body叉乘in_plane_heading
        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body / torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack([x_axis_body, y_axis_body, z_axis_body], dim=-1)

        rot_matrix = torch.cat([start_R, next_R, rot_matrix, end_R], dim=0)
        # 前一矩阵的逆乘以后一矩阵，，计算世界坐标下相邻2个pose的旋转向量
        current_omega = rot_matrix_to_vec(rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1, -2)) / self.delta_t
        current_omega = torch.cat([current_omega, end_omega], dim=0)

        prev_omega = current_omega[:-1, :]
        next_omega = current_omega[1:, :]

        angular_accel = (next_omega - prev_omega) / self.delta_t
        # duplicate last ang_accceleration - its not actaully used for anything (there is no action at last state)
        angular_accel = torch.cat([angular_accel, angular_accel[-1, None, :]], dim=0)

        # S, 3    3,3      S, 3, 1
        torques = (self.J @ angular_accel[..., None])[..., 0]
        actions = torch.cat([accel_mag * self.mass, torques], dim=-1)

        return current_pos, current_vel, current_accel, rot_matrix, current_omega, angular_accel, actions






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
    gaussians.load_ply(os.path.join(args.model_path,
                                    "point_cloud",
                                    "iteration_" + str(loaded_iter),
                                    "point_cloud.ply"))  # 点云初始化
    bg_color = [1, 1, 1] if True else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    xyz = gaussians.get_xyz ##高斯位置
    scales = gaussians.get_scaling ##高斯大小
    density = gaussians.get_opacity ##高斯密度
    delta_t = 0.02 ##时间间隔


    # 定义起点和终点
    start= np.array([-0.54, -0.76, 0.17])  # 起点坐标 0.0829 0.6033 0.12724
    goal= np.array([0.08,0.60, 0.12])  # 终点坐标
    init_pose = np.array([0,0,0])  ##初始姿态（欧拉角）
    end_pose = np.array([0,0,0]) ##目标姿态（欧拉角）
    init_rates = np.zeros(3) ##初始角速度和线速度

    #####状态初始化#######
    start_state = np.hstack((start,init_rates,init_pose,init_rates))##初始state
    end_state = np.hstack((goal, init_rates, end_pose, init_rates))##结束state

    traj = Planner(start_state,end_state,xyz,scales,density,delta_t)
    path = traj.path_searching(start,goal)


    #####visuialize###
    filename = 'potential_visual.json'
    poses=[]
    pose_dict={}
    with open(filename, "w+") as f:
        for pos in path:
            pose = np.zeros((4, 4))
            pose[:3, :3] = np.identity(3)
            pose[:3, 3] = pos
            pose[3, 3] = 1
            poses.append(pose.tolist())
        pose_dict["poses"] = poses
        json.dump(pose_dict, f, indent=4)

    traj.plot_point_num(path)
    # path = np.array(path)
    # #####初始化参考state(12维)###### [pos(3), vel(3), R (3), omega(3)]
    # vel_list = (path[1:] - path[:-1]) / traj.delta_t
    # # ##加上终止速度，0
    # vel_list = np.vstack((vel_list, np.zeros(3)))
    #
    # #path, vel_list去除start_state
    # path, vel_list = path[1:],vel_list[1:]
    #
    # #todo:加速度，移除头2个state 中 rotation,已经被其他条件限制了！！
    # accel = ( vel_list[1:] - vel_list[:-1])/traj.delta_t
    # push_dir = accel - np.array([0, 0, -10])#无人机推力方向
    #
    # curr_pose = vec_to_rot_matrix(start_state[6:9])
    #
    # curr_omegas = start_state[9:]
    #
    # curr_pose = curr_pose @ curr_omegas
    #
    # ###根据给出的waypoint计算姿态和角速度
    # for i in range(len(push_dir)-1):
    #     ##旋转角度
    #     sida = np.arccos( np.dot(push_dir[i+1],push_dir[i])/ ( np.linalg.norm(push_dir[i+1])*np.linalg.norm(push_dir[i]) ) )
    #     ##旋转向量(世界坐标)
    #     cross_vector = np.cross(push_dir[i],push_dir[i+1])
    #     n_rot =  sida * cross_vector / np.linalg.norm(cross_vector)
    #     ###根据当前pose将旋转向量变换到机体坐标
    #     n_rot_loc = curr_pose.T @ n_rot
    #     curr_omega = n_rot_loc / delta_t #绕机体坐标xyz的角速度
    #     ref_state[i,9:] = curr_omega
    #     ref_state[i, 6:9] = rot_matrix_to_vec(curr_pose)
    #     ###下一时刻pose
    #     next_pose = vec_to_rot_matrix(n_rot) @ curr_pose
    #     curr_pose = next_pose
    #     if i == 0:
    #         curr_poses = next_pose
    #         curr_omegas = curr_omega
    #     else:
    #         curr_poses.vstack((curr_poses,next_pose))##拼接Pose
    #         curr_omegas.vstack((curr_omegas,curr_omega))##拼接omega
    #
    # ##补上倒数第二个state的角速度和end_omega
    # rot_vel = curr_poses[-1].T @ rot_matrix_to_vec(end_pose @ curr_poses[-1].T)
    # curr_omegas.vstack(curr_omegas,rot_vel / delta_t,np.zeros(3))
    #
    # ref_state = np.concatenate([np.array(path),vel_list,curr_poses,curr_omegas],axis=1)



















