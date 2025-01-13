import torch
from torch._C import device
import numpy as np
import json
from plyfile import PlyData
from .math_utils import rot_matrix_to_vec
from .quad_helpers import astar, next_rotation
#from path_search import repulsive_force,path_searching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Planner:
    def __init__(self, start_state, end_state, cfg, density_fn,gaussian=None):
        self.nerf = density_fn
        self.gaussian = gaussian
        self.cfg                = cfg
        self.T_final            = cfg['T_final']#2.0
        self.steps              = cfg['steps']#20
        self.lr                 = cfg['lr']#学习率0.001
        self.epochs_init        = cfg['epochs_init']#2500
        self.epochs_update      = cfg['epochs_update']#250
        self.fade_out_epoch     = cfg['fade_out_epoch']#0
        self.fade_out_sharpness = cfg['fade_out_sharpness']#10
        self.mass               = cfg['mass']#1.0
        self.J                  = cfg['I']#单位矩阵
        self.g                  = torch.tensor([0., 0., -cfg['g']])#0，0，-10
        self.body_extent        = cfg['body']
        self.body_nbins         = cfg['nbins']#10，10，5

        self.CHURCH = False

        self.dt = self.T_final / self.steps#2/20 = 0.1

        self.start_state = start_state
        self.end_state   = end_state
        #去头去尾，增加一个维度 ,shape=(18,1)
        slider = torch.linspace(0, 1, self.steps)[1:-1, None]#
        #start_state+(end_state-start_state)*slider-不同的比例因子
        states = (1-slider) * self.full_to_reduced_state(start_state) + \
                    slider  * self.full_to_reduced_state(end_state)
        #states的最后一维是相机坐标x轴与世界坐标x轴夹角？
        self.states = states.clone().detach().requires_grad_(True)
        self.initial_accel = torch.tensor([cfg['g'], cfg['g']]).requires_grad_(True)#初始2个加速度

        #PARAM this sets the shape of the robot body point cloud
        #设置机器人身体点云的形状
        body = torch.stack( torch.meshgrid( torch.linspace(self.body_extent[0, 0], self.body_extent[0, 1], self.body_nbins[0]),
                                            torch.linspace(self.body_extent[1, 0], self.body_extent[1, 1], self.body_nbins[1]),
                                            torch.linspace(self.body_extent[2, 0], self.body_extent[2, 1], self.body_nbins[2])), dim=-1)
        self.robot_body = body.reshape(-1, 3)

        if self.CHURCH:
            self.robot_body = self.robot_body/2

        self.epoch = 0

    def load_ply(self,path):
        plydata = PlyData.read(path)
        #global xyz, opacities
        self.xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        self.opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    def full_to_reduced_state(self, state):
        #state.shape=(18,)
        pos = state[:3]
        R = state[6:15].reshape((3,3))#这里是blender里设置的R

        x,y,_ = R @ torch.tensor( [1.0, 0, 0 ] )
        angle = torch.atan2(y, x)#archtan()求角度

        return torch.cat( [pos, torch.tensor([angle]) ], dim = -1).detach()

    # def query_which_cube(self,x,y,z):
    #     loc = (torch.tensor([x, y, z]) - torch.tensor([x_min, y_min, z_min])) / inter_val
    #     return int(loc[0]),int(loc[1]),int(loc[2])
    def potential_field(self):
        xyz = self.gaussian.get_xyz
        scales = self.gaussian.get_scaling
        density = self.gaussian.get_opacity
        density = density.detach().cpu().numpy()
        xyz = xyz.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        eta = 15  # 引力增益系数
        zeta = 25  # 斥力增益系数
        rho0 = 0.08  # 障碍物影响范围
        goal_tol = 0.05  # 目标点容差
        max_iter = 500  # 最大迭代次数
        lr = 0.05
        start=self.start_state[:3].detach().cpu().numpy()
        end= self.end_state[:3].detach().cpu().numpy()
        path=path_searching(zeta,rho0,xyz,eta,lr,max_iter,goal_tol,start,end,scales,density)
        squares=torch.tensor(np.array(path),device='cuda',dtype=torch.float32)

        #squares = torch.cat([torch.tensor(i) for i in path],dim=0)
        states = torch.cat([squares, torch.zeros((squares.shape[0], 1))], dim=-1)
        randomness = torch.normal(mean=0, std=0.001 * torch.ones(states.shape))
        states += randomness
        self.states = states.clone().detach().requires_grad_(True)




    def a_star_init(self):
        side = 100 #PARAM grid size
        #生成区域大小
        if self.CHURCH:
            x_linspace = torch.linspace(-2,-1, side)
            y_linspace = torch.linspace(-1,0, side)
            z_linspace = torch.linspace(0,1, side)
            x_min, x_max = -2,-1
            y_min, y_max = -1,0
            z_min, z_max = 0,1
            coods = torch.stack( torch.meshgrid( x_linspace, y_linspace, z_linspace), dim=-1)
        else:
            linspace = torch.linspace(-1,1, side) #PARAM extends of the thing
            x_min, x_max = -1,1
            y_min, y_max = -1,1
            z_min, z_max = -1,1
            # side, side, side, 3
            coods = torch.stack( torch.meshgrid( linspace, linspace, linspace ), dim=-1)

        inter_val = (torch.tensor([x_max, y_max, z_max]) - torch.tensor([x_min, y_min, z_min])) / (side - 1)

        kernel_size = 5 # 100/5 = 20. scene size of 2 gives a box size of 2/20 = 0.1 = drone size
        if self.gaussian:
            #self.xyz = self.gaussian._xyz.detach().cpu().numpy()
            self.load_ply('../gaussian-splatting/data/output/point_cloud/iteration_10000/point_cloud.ply')
            occupy = torch.zeros([i - 1 for i in coods.shape[:-1]] + [2])
            sigam = torch.sigmoid(torch.tensor(self.opacities))
            #sigam = self.gaussian.get_opacity

            for i in range(self.xyz.shape[0]):
                #xx, yy, zz = query_which_cube(xyz[i][0], xyz[i][1], xyz[i][2])
                #选择场景内高斯
                if (self.xyz[i] >= -1).all() and (self.xyz[i]<=1).all():
                    loc = (torch.tensor([self.xyz[i][0], self.xyz[i][1], self.xyz[i][2]]) - torch.tensor([x_min, y_min, z_min])) / inter_val
                    xx, yy, zz = int(loc[0]),int(loc[1]),int(loc[2])
                    occupy[xx][yy][zz][0] += sigam[i][0]  # 高斯密度
                    occupy[xx][yy][zz][1] += 1  # 高斯数量

            occupied = (occupy[..., 0] / (occupy[..., 1] + 0.0001)) > 0.1

        else:
            output = self.nerf(coods)#预测区域中每个点的不透明度
            maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
            #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

            # occupied.shape = 20, 20, 20  ,  计算每个点的是否被占据,阈值为0.3
            occupied = maxpool(output[None,None,...])[0,0,...] > 0.3

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )#取整数
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )#取整数

        print(start, end)
        path = astar(occupied, start, end)#A*寻路,返回List

        # convert from index cooredinates,搜索到的路径变回世界坐标
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

        #adding way,states = (19,4)
        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

        #prevents weird zero derivative issues
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness

        # smooth path (diagram of which states are averaged)
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
        next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
        states = (prev_smooth + next_smooth + states)/3

        self.states = states.clone().detach().requires_grad_(True)

    def params(self):
        return [self.initial_accel, self.states]

    def calc_everything(self):
        #这里增加了第一维1
        start_pos   = self.start_state[None, 0:3]#位置
        start_v     = self.start_state[None, 3:6]#速度(vx,vy,vz)
        start_R     = self.start_state[6:15].reshape((1, 3, 3))
        start_omega = self.start_state[None, 15:]

        end_pos   = self.end_state[None, 0:3]
        end_v     = self.end_state[None, 3:6]
        end_R     = self.end_state[6:15].reshape((1, 3, 3))
        end_omega = self.end_state[None, 15:]
        #下一时刻的rotation
        next_R = next_rotation(start_R, start_omega, self.dt)#先左乘角速度的旋转矩阵再乘start_R，得到在世界坐标下的rotation

        # start, next, decision_states, last, end
        #计算重力和驱动力的合力加速度，start_R @ torch.tensor([0,0,1.0])-在世界坐标系下的表达
        start_accel = start_R @ torch.tensor([0,0,1.0]) * self.initial_accel[0] + self.g
        next_accel = next_R @ torch.tensor([0,0,1.0]) * self.initial_accel[1] + self.g
        #下2时刻速度
        next_vel = start_v + start_accel * self.dt
        after_next_vel = next_vel + next_accel * self.dt
        #下3时刻位置
        next_pos = start_pos + start_v * self.dt
        after_next_pos = next_pos + next_vel * self.dt
        after2_next_pos = after_next_pos + after_next_vel * self.dt
    
        # position 2 and 3 are unused - but the atached roations are
        #把计算到的pos和搜索到的Waypoint拼接，头部加入start_pos，尾部加入end_pos
        current_pos = torch.cat( [start_pos, next_pos, after_next_pos, after2_next_pos, self.states[2:, :3], end_pos], dim=0)

        prev_pos = current_pos[:-1, :]
        next_pos = current_pos[1: , :]
        #计算相邻2个pos间的速度
        current_vel = (next_pos - prev_pos)/self.dt
        current_vel = torch.cat( [ current_vel, end_v], dim=0)#加上终止速度

        prev_vel = current_vel[:-1, :]
        next_vel = current_vel[1: , :]
        # 计算相邻2个速度间的加速度
        current_accel = (next_vel - prev_vel)/self.dt - self.g

        #重复最后一个加速度  duplicate last accceleration - its not actaully used for anything (there is no action at last state)
        current_accel = torch.cat( [ current_accel, current_accel[-1,None,:] ], dim=0)
        #计算每个加速度的模长
        accel_mag     = torch.norm(current_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration,加速度方向的单位向量
        z_axis_body = current_accel/accel_mag

        # remove states with rotations already constrained
        z_axis_body = z_axis_body[2:-1, :]
        #机体x坐标在z=0平面投影和world_x夹角
        z_angle = self.states[:,3]
        #??????????
        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)
        #   z_axis_body叉乘in_plane_heading
        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        rot_matrix = torch.cat( [start_R, next_R, rot_matrix, end_R], dim=0)
        #前一矩阵的逆乘以后一矩阵，，计算世界坐标下相邻2个pose的旋转向量
        current_omega = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt
        current_omega = torch.cat( [ current_omega, end_omega], dim=0)

        prev_omega = current_omega[:-1, :]
        next_omega = current_omega[1:, :]

        angular_accel = (next_omega - prev_omega)/self.dt
        # duplicate last ang_accceleration - its not actaully used for anything (there is no action at last state)
        angular_accel = torch.cat( [ angular_accel, angular_accel[-1,None,:] ], dim=0)

        # S, 3    3,3      S, 3, 1
        torques = (self.J @ angular_accel[...,None])[...,0]
        actions =  torch.cat([ accel_mag*self.mass, torques ], dim=-1)

        return current_pos, current_vel, current_accel, rot_matrix, current_omega, angular_accel, actions

    def get_full_states(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def get_actions(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        if not torch.allclose( actions[:2, 0], self.initial_accel ):
            print(actions)
            print(self.initial_accel)
        return actions

    def get_next_action(self):
        actions = self.get_actions()
        # fz, tx, ty, tz
        return actions[0, :]

    def body_to_world(self, points):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def get_state_cost(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        fz = actions[:, 0].to(device)
        torques = torch.norm(actions[:, 1:], dim=-1).to(device)

        # S, B, 3  =  S, _, 3 +      _, B, 3   X    S, _,  3
        #self.robot_body 是机器人身体的离散化点云 【500，3】
        B_body, B_omega = torch.broadcast_tensors(self.robot_body, omega[:,None,:])
        #v = v0 + w*r
        point_vels = vel[:,None,:] + torch.cross(B_body, B_omega, dim=-1)

        # S, B
        distance = torch.sum( vel**2 + 1e-5, dim = -1)**0.5
        # S, B
        density = self.nerf( self.body_to_world(self.robot_body) )**2

        # multiplied by distance to prevent it from just speed tunnelling
        # S =   S,B * S,_
        #碰撞概率
        #
        colision_prob = torch.mean(density * distance[:,None], dim = -1) 

        if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t)).to(device)
            colision_prob = colision_prob * mask

        #PARAM cost function shaping
        return 1000 * fz ** 2 + 0.01 * torques ** 4 , colision_prob * 1e6
        #return 1000*fz**2 + 0.01*torques**4 + colision_prob * 1e6, colision_prob*1e6


    def total_cost(self):
        total_cost, colision_loss  = self.get_state_cost()
        return torch.mean(total_cost)

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)

        try:
            for it in range(self.epochs_init):
                opt.zero_grad()
                self.epoch = it#记录当前epoch
                loss = self.total_cost()#start to end的力的控制代价和碰撞代价
                print(it, loss)
                loss.backward()
                opt.step()

                save_step = 50
                if it%save_step == 0:
                    if hasattr(self, "basefolder"):
                        self.save_poses(self.basefolder / "init_poses" / (str(it//save_step)+".json"))
                        self.save_costs(self.basefolder / "init_costs" / (str(it//save_step)+".json"))
                    else:
                        print("Warning: data not saved!")

        except KeyboardInterrupt:
            print("finishing early")

    def learn_update(self, iteration):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)

        for it in range(self.epochs_update):
            opt.zero_grad()
            self.epoch = it
            loss = self.total_cost()###start_state 到 end_state 总控制，和碰撞代价
            print(it, loss)
            loss.backward()
            opt.step()
            # it += 1

            # if (it > self.epochs_update and self.max_residual < 1e-3):
            #     break

            save_step = 50
            if it%save_step == 0:
                if hasattr(self, "basefolder"):
                    self.save_poses(self.basefolder / "replan_poses" / (str(it//save_step)+ f"_time{iteration}.json"))
                    self.save_costs(self.basefolder / "replan_costs" / (str(it//save_step)+ f"_time{iteration}.json"))
                else:
                    print("Warning: data not saved!")

    def update_state(self, measured_state):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        ###把故估计出的状态进行更新
        self.start_state = measured_state
        self.states = self.states[1:, :].detach().requires_grad_(True)##移除首个状态
        self.initial_accel = actions[1:3, 0].detach().requires_grad_(True)##取得2个加速度
        # print(self.initial_accel.shape)

    def plot(self, quadplot):
        quadplot.trajectory( self, "g" )
        ax = quadplot.ax_graph

        pos, vel, accel, _, omega, _, actions = self.calc_everything()
        actions = actions.cpu().detach().numpy()
        pos = pos.cpu().detach().numpy()
        vel = vel.cpu().detach().numpy()
        omega = omega.cpu().detach().numpy()

        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        ax.plot(actions[...,3], label="tz")

        ax.plot(pos[...,0], label="px")
        # ax.plot(pos[...,1], label="py")
        # ax.plot(pos[...,2], label="pz")

        ax.plot(vel[...,0], label="vx")
        # ax.plot(vel[...,1], label="vy")
        ax.plot(vel[...,2], label="vz")

        # ax.plot(omega[...,0], label="omx")
        ax.plot(omega[...,1], label="omy")
        # ax.plot(omega[...,2], label="omz")

        ax_right = quadplot.ax_graph_right

        total_cost, colision_loss = self.get_state_cost()
        ax_right.plot(total_cost.detach().numpy(), 'black', label="cost")
        ax_right.plot(colision_loss.detach().numpy(), 'cyan', label="colision")
        ax.legend()

    def save_poses(self, filename):
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        poses = []
        pose_dict = {}
        with open(filename,"w+") as f:
            for pos, rot in zip(positions, rot_matrix):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.cpu().detach().numpy()
                pose[:3, 3]  = pos.cpu().detach().numpy()
                pose[3,3] = 1

                poses.append(pose.tolist())
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

    def save_costs(self, filename):
        positions, vel, _, rot_matrix, omega, _, actions = self.calc_everything()
        total_cost, colision_loss  = self.get_state_cost()

        output = {"colision_loss": colision_loss.cpu().detach().numpy().tolist(),
                  "pos": positions.cpu().detach().numpy().tolist(),
                  "actions": actions.cpu().detach().numpy().tolist(),
                  "total_cost": total_cost.cpu().detach().numpy().tolist()}

        with open(filename,"w+") as f:
            json.dump( output,  f, indent=4)

    def save_progress(self, filename):
        if hasattr(self.renderer, "config_filename"):
            config_filename = self.renderer.config_filename
        else:
            config_filename = None

        to_save = {"cfg": self.cfg,
                    "start_state": self.start_state,
                    "end_state": self.end_state,
                    "states": self.states,
                    "initial_accel":self.initial_accel,
                    "config_filename": config_filename,
                    }
        torch.save(to_save, filename)


    # def load_progress(cls, filename, renderer=None):
    #     # a note about loading: it won't load the optimiser learned step sizes
    #     # so the first couple gradient steps can be quite bad

    #     loaded_dict = torch.load(filename)
    #     print(loaded_dict)

    #     if renderer == None:
    #         assert loaded_dict['config_filename'] is not None
    #         renderer = load_nerf(loaded_dict['config_filename'])

    #     obj = cls(renderer, loaded_dict['start_state'], loaded_dict['end_state'], loaded_dict['cfg'])
    #     obj.states = loaded_dict['states'].requires_grad_(True)
    #     obj.initial_accel = loaded_dict['initial_accel'].requires_grad_(True)

    #     return obj

'''
def main():

    # violin - astar
    # renderer = get_nerf('configs/violin.txt')
    # start_state = torch.tensor([0.44, -0.23, 0.2, 0])
    # end_state = torch.tensor([-0.58, 0.66, 0.15, 0])

    #playground
    experiment_name = "playground_slide"
    renderer = get_nerf('configs/playground.txt')

    # under slide
    start_pos = torch.tensor([-0.3, -0.27, 0.06])
    end_pos = torch.tensor([0.02, 0.58, 0.65])

    # around slide
    # start_pos = torch.tensor([-0.3, -0.27, 0.06])
    # end_pos = torch.tensor([-0.14, 0.6, 0.78])


    #stonehenge
    # renderer = get_nerf('configs/stonehenge.txt')
    # start_state = torch.tensor([-0.06, -0.79, 0.2, 0])
    # end_state = torch.tensor([-0.46, 0.55, 0.16, 0])

    # start_pos = torch.tensor([-0.05,-0.9, 0.2])
    # end_pos   = torch.tensor([-1 , 0.7, 0.35])
    # start_pos = torch.tensor([-1, 0, 0.2])
    # end_pos   = torch.tensor([ 1, 0, 0.5])


    start_R = vec_to_rot_matrix( torch.tensor([0.0,0.0,0]))
    start_state = torch.cat( [start_pos, torch.tensor([0,0,0]), start_R.reshape(-1), torch.zeros(3)], dim=0 )
    end_state   = torch.cat( [end_pos,   torch.zeros(3), torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )

    # experiment_name = "test" 
    # filename = "line.plan"
    # renderer = get_manual_nerf("empty")
    # renderer = get_manual_nerf("cylinder")

    cfg = {"T_final": 2,
            "steps": 20,
            "lr": 0.01,
            "epochs_init": 2500,
            "fade_out_epoch": 0,
            "fade_out_sharpness": 10,
            "epochs_update": 250,
            }


    basefolder = "experiments" / pathlib.Path(experiment_name)
    if basefolder.exists():
        print(basefolder, "already exists!")
        if input("Clear it before continuing? [y/N]:").lower() == "y":
            shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "train_poses").mkdir()
    (basefolder / "train_graph").mkdir()

    print("created", basefolder)

    traj = System(renderer, start_state, end_state, cfg)
    # traj = System.load_progress(filename, renderer); traj.epochs_update = 250 #change depending on noise

    traj.basefolder = basefolder

    traj.a_star_init()

    # quadplot = QuadPlot()
    # traj.plot(quadplot)
    # quadplot.show()

    traj.learn_init()

    traj.save_progress(basefolder / "trajectory.pt")

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.show()


    save = Simulator(start_state)
    save.copy_states(traj.get_full_states())

    if False: # for mpc control
        sim = Simulator(start_state)
        sim.dt = traj.dt #Sim time step changes best on number of steps

        for step in range(cfg['steps']):
            action = traj.get_next_action().clone().detach()
            print(action)

            state_noise = torch.normal(mean= 0, std=torch.tensor( [0.01]*3 + [0.01]*3 + [0]*9 + [0.005]*3 ))
            # state_noise[3] += 0.0 #crosswind

            # sim.advance(action) # no noise
            sim.advance(action, state_noise) #add noise
            measured_state = sim.get_current_state().clone().detach()

            measurement_noise = torch.normal(mean= 0, std=torch.tensor( [0.01]*3 + [0.02]*3 + [0]*9 + [0.005]*3 ))
            measured_state += measurement_noise
            traj.update_state(measured_state) 

            traj.learn_update()

            print("sim step", step)
            if step % 5 !=0 or step == 0:
                continue

            quadplot = QuadPlot()
            traj.plot(quadplot)
            quadplot.trajectory( sim, "r" )
            quadplot.trajectory( save, "b", show_cloud=False )
            quadplot.show()


        quadplot = QuadPlot()
        traj.plot(quadplot)
        quadplot.trajectory( sim, "r" )
        quadplot.trajectory( save, "b", show_cloud=False )
        quadplot.show()

def OPEN_LOOP(traj):
    sim = Simulator(traj.start_state)
    sim.dt = traj.dt #Sim time step changes best on number of steps

    for step in range(cfg['steps']):
        action = traj.get_actions()[step,:].detach()
        print(action)
        sim.advance(action)

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.trajectory( sim, "r" )
    quadplot.trajectory( save, "b", show_cloud=False )
    quadplot.show()

if __name__ == "__main__":
    main()
'''