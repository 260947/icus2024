import matplotlib.pyplot as plt
import numpy as np
import json
import time
from utils.system_utils import searchForMaxIteration
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
        self.rho0 = 0.06  # 障碍物影响范围，0.05
        self.goal_tol = 0.03  # 目标点容差，0.01
        self.max_iter = 180  # 最大迭代次数，500
        self.n = 0.5  # 斥力修正因子 取值范围[0,1]
        ###高斯的一些参数
        self.density = density.detach().cpu().numpy()
        self.xyz = xyz.detach().cpu().numpy()
        self.scales = scales.detach().cpu().numpy()

        self.att_list = []
        self.rep_list = []

    def att_potential(self,curr,goal):
        return 0.5 * self.eta * (np.linalg.norm(curr-goal))**2

    def rep_potential(self,curr,goal):
        rep_sum=0
        for i in range(self.xyz.shape[0]):
            dist = np.linalg.norm(curr-self.xyz[i])
            if dist <=self.rho0+self.scales[i][0]:
                rep_sum+=0.5*self.zeta*(1/dist - 1/(self.rho0+self.scales[i][0]) )**2 * (np.linalg.norm(curr-goal))**self.n
        return rep_sum* 1e-4


    #计算引力
    def attractive_force(self,curr,goal):
        # 引力与点到目标的距离平方成正比

        #返回引力势场梯度力
        return -self.eta * (curr - goal)

    def repulsive_force(self,point,goal):
        # 斥力与点到障碍物的距离成反比
        # 如果距离大于阈值，则斥力为零
        grad=np.array([0.,0.,0.])
        for i in range(self.xyz.shape[0]):
            dist = np.linalg.norm(point-self.xyz[i])
            if dist <=self.rho0+self.scales[i][0]:
            #if dist <= self.rho0 :
                grad=grad + np.linalg.norm(point-goal)**self.n * (-self.zeta * self.density[i][0]*(1/dist - 1.0 / (self.rho0+self.scales[i][0]))* (1.0 / dist**3)*(point - self.xyz[i])) + 0.5 * self.n * self.zeta*(1/dist - 1/(self.rho0+self.scales[i][0]) )**2 * (point-goal) * np.linalg.norm(point-goal)**(self.n-2)
                #grad = grad + np.linalg.norm(point - goal) ** self.n * (-self.zeta * self.density[i][0] * (1 / dist - 1.0 / self.rho0 ) * (1.0 / dist ** 3) * (point - self.xyz[i])) + 0.5 * self.n * self.zeta * (1 / dist - 1 / (self.rho0 )) ** 2 * (point - goal) * np.linalg.norm(point - goal) ** (self.n - 2)

        #返回斥力势场梯度力
        return -grad

    def plot_sth(self,path):
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

    def stuck_in_local_minimum(self,path):

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
        # if sum_1  < 0.85 * sum_2 :
        #     print('极小值！！！！！')

        return sum_1  < 0.85 * sum_2





    def path_searching(self,start,goal):
        max_vel = 4 #最大步长
        path=[start]
        path_reverse = [goal]
        iters=0
        last_iter = -5
        curr_pos = path[-1]
        curr_pos_2 = path_reverse[-1]
        record=[]

        t1 = time.time()

        ##while np.linalg.norm(curr_pos - goal) > self.goal_tol and iters < self.max_iter:
        while  iters < self.max_iter:
            # if np.linalg.norm(curr_pos - curr_pos_2) <= self.goal_tol + 0.08 :
            #     print('斥力：',np.linalg.norm( self.repulsive_force(curr_pos,goal) ))
            #     if  np.linalg.norm( self.repulsive_force(curr_pos,goal) )  < 100 :
            #         print('斥力：',np.linalg.norm( self.repulsive_force(curr_pos,goal) ))
            #         print('耗时：', time.time() - t1)
            #         print('路径长度1：', len(path), '路径长度2：', len(path_reverse))
            #         return path+path_reverse[::-1]
        ##while iters < self.max_iter:
            if np.linalg.norm(curr_pos - goal) <= self.goal_tol:
                print('start to goal')
                print('耗时：', time.time() - t1)
                return path

            # if np.linalg.norm(curr_pos_2 - start) <= self.goal_tol:
            #     print('goal to start')
            #     print('耗时：', time.time() - t1)
            #     return path_reverse[::-1]

            iters+=1
            ##计算斥力和引力
            att_force , rep_force = self.attractive_force(curr_pos,goal) , self.repulsive_force(curr_pos,goal)
            #att_force_2, rep_force_2 = self.attractive_force(curr_pos_2, start), self.repulsive_force(curr_pos_2, start)

            gradient = att_force + rep_force
            grad_len = np.linalg.norm(gradient)
            gradient = gradient/grad_len#归一化
            vel = max_vel * (1 - np.exp(-grad_len / 5 )) * gradient   # 世界坐标系下无人机的理论速度

            ##计算势能
            self.att_list.append(self.att_potential(curr_pos,goal))
            self.rep_list.append(self.rep_potential(curr_pos,goal))


            ###处理局部极值###
            attr_v = att_force / np.linalg.norm(att_force)
            rep_v = rep_force / (np.linalg.norm(rep_force) + 0.000001)
            cross_ = np.cross(attr_v, rep_v)  #
            dot_ = np.dot(attr_v, rep_v)
            sign = 0

            record.append(0)
            if  ( np.linalg.norm(curr_pos - goal) >  1.5 * self.goal_tol and len(path)>=10 and np.linalg.norm(path[-3] - path[-1]) <= 0.005) or ( np.linalg.norm(curr_pos - goal) >  2 * self.goal_tol and     ( len(path) > 15 and self.stuck_in_local_minimum(path) ) ):
                record[-1] = 1
                #sign = 1
            if dot_ < -0.9:
                curr_potential = self.att_list[-1] + self.rep_list[-1]
                next_potential = self.rep_potential(curr_pos + vel * self.delta_t,goal) + self.att_potential(curr_pos + vel * self.delta_t,goal)
                if next_potential - curr_potential > 0.2 and iters - last_iter> 5:
                    sign+=4.5

            if len(record)>=5:
                sign+=sum(record[-5:])/2

            sign-=dot_
            if sign >= 3 and iters - last_iter > 5:
                last_iter = iters
                min_distur = (-dot_/(np.linalg.norm(attr_v))**2)*rep_v + attr_v #和斥速度垂直，与吸速度同一大致方向
                #min_distur = -min_distur #和斥速度垂直，与吸速度同一反方向

                #if np.linalg.norm(cross_)== 0: 恰好方向相反
                #u =
                dist = 0.03 ##0.05
                min_distur[:] =  -dist * min_distur/np.linalg.norm(min_distur)
                #min_distur = min_distur + 2*att_force

                gs_pos = (curr_pos + min_distur).reshape(1, 3)
                gs_scale = np.array([0.25]).reshape(1, 1) ##0.25
                gs_density = np.array([0.9]).reshape(1, 1) ##0.9
                if np.linalg.norm(gs_pos - goal) > 2.5 * self.goal_tol:
                    self.xyz = np.append(self.xyz, values=gs_pos, axis=0)
                    self.scales = np.append(self.scales, values=gs_scale, axis=0)
                    self.density = np.append(self.density, values=gs_density, axis=0)


                new_force = np.linalg.norm(curr_pos - goal) ** self.n * (-self.zeta * gs_density[0] * (1 / dist - 1.0 / (self.rho0 + gs_scale[0])) * (1.0 / dist ** 3) * (curr_pos - gs_pos[0])) + 0.5 * self.n * self.zeta * (1 / dist - 1 / (self.rho0 + gs_scale[0])) ** 2 * (curr_pos - goal) * np.linalg.norm(curr_pos - goal) ** (self.n - 2)
                new_force = -new_force

                new_force =  2 * np.linalg.norm(rep_force) * new_force / np.linalg.norm(new_force)

                print('轮次：', iters ,'高斯位置：',gs_pos)


                # adjust_ = 0
                # if last_iter:
                #     cos_ = np.dot(last_distur, min_distur) / (
                #                 np.linalg.norm(min_distur) * np.linalg.norm(last_distur))
                #     if cos_ < -0.7 and iters - last_iter <= 10:
                #         #min_distur = -min_distur
                #         min_distur = last_distur
                #         adjust_ = iters - last_iter

                # last_distur = min_distur
                # last_iter = iters



                gradient =  att_force +  rep_force + new_force ### 2 * np.linalg.norm(rep_force) * cross_/np.linalg.norm(cross_)

                grad_len = np.linalg.norm(gradient)
                gradient = gradient / grad_len  # 归一化
                vel =  max_vel * (1 - np.exp(-grad_len / 5 )) * gradient
                #vel = (1+ adjust_/10 )*vel

                print('11垂直速度扰动！')

            if iters % 20==0:
                max_vel = max(3,0.9*max_vel)
                #print(path,'\n',path_reverse)
                #print(path[-15:])

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
            #last_potential = curr_potential


        # ##加上终止速度，0
        # vel_list = np.vstack((vel_list, np.zeros(3)))
        print('寻路失败！！')
        #return path







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
    delta_t = 0.01 ##时间间隔


    # 定义起点和终点 0.0330 -0.793251 m  0.251245 m -0.69881 m
    #start=-0.54, -0.76, 0.17，end = 0.08,0.60, 0.12.png
    start= np.array([-0.2556, 0.7903, 0.2169])  # 起点坐标 0.0829 0.6033 0.12724
    goal= np.array([ -0.5554, -0.9017, 0.3058])  # 终点坐标
    init_pose = np.array([0,0,0])  ##初始姿态（欧拉角）
    end_pose = np.array([0,0,0]) ##目标姿态（欧拉角）
    init_rates = np.zeros(3) ##初始角速度和线速度

    #####状态初始化#######
    start_state = np.hstack((start,init_rates,init_pose,init_rates))##初始state
    end_state = np.hstack((goal, init_rates, end_pose, init_rates))##结束state

    traj = Planner(start_state,end_state,xyz,scales,density,delta_t)
    path = traj.path_searching(start,goal)

    plt.plot(traj.att_list,color='r',label='att_potential')
    plt.plot(traj.rep_list, color='g', label='rep_potential')
    plt.plot(np.array(traj.att_list)+np.array(traj.rep_list), color='b', label='total_potential')
    plt.legend(fontsize=12)
    plt.tick_params(labelsize=10)
    plt.xlabel('iters', fontsize=12, loc='right')
    plt.ylabel('values', fontsize=12, loc='top')

    # 显示图形
    plt.show()
    #plt.pause(10)
    #traj.plot_sth(path)


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



















