import numpy as np
import shutil
import pathlib
import subprocess
from tqdm import trange
import argparse
from demos.render import *
from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate(planner_cfg, agent_cfg, filter_cfg, extra_cfg, gaussian, pipeline):
    start_state = planner_cfg['start_state']
    end_state = planner_cfg['end_state']
    # Creates a workspace to hold all the trajectory data
    basefolder = "paths" / pathlib.Path(planner_cfg['exp_name'])
    if basefolder.exists():
        print(basefolder, "already exists!")
        if input("Clear it before continuing? [y/N]:").lower() == "y":
            shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "init_poses").mkdir()
    (basefolder / "init_costs").mkdir()
    (basefolder / "estimator_data").mkdir()
    print("created", basefolder)

    # Initialize Planner，
    traj = Planner(start_state, end_state, planner_cfg, gaussian=gaussian)
    traj.basefolder = basefolder

    traj.potential_field()
    # Change start state from 18-vector (with rotation as a rotation matrix) to 12 vector (with rotation as a rotation vector)
    start_state = torch.cat([start_state[:6], rot_matrix_to_vec(start_state[6:15].reshape((3, 3))), start_state[15:]],
                            dim=-1).cuda()

    agent_cfg['x0'] = start_state

    # Initialize the agent. Evolves the agent with time and interacts with the simulator (Blender) to get observations.
    agent = Agent(agent_cfg, camera_cfg, blender_cfg)

    # State estimator. Takes the observations from Agent class and performs filtering to get a state estimate (12-vector)
    filter = Estimator(filter_cfg, agent, start_state, gaussian=gaussian, pipeline=pipeline, white_background=True)
    filter.basefolder = basefolder

    true_states = start_state.cpu().detach().numpy()
    # 有多少个动作action
    steps = traj.get_actions().shape[0]

    noise_std = extra_cfg['mpc_noise_std']
    noise_mean = extra_cfg['mpc_noise_mean']

    try:
        for iter in trange(steps):
            # In MPC style, take the next action recommended from the planner
            if iter < steps - 5:
                action = traj.get_next_action().clone().detach()  # get_next_action()-取首个actions
            else:
                action = traj.get_actions()[iter - steps + 5, :]  # get_actions()-多个actions

            noise = torch.normal(noise_mean, noise_std)

            # Have the agent perform the recommended action, subject to noise. true_pose, true_state are here
            # for simulation purposes in order to benchmark performance. They are the true state of the agent
            # subjected to noise. gt_img is the observation.

            # 根据action计算得新的state,pose,新pose下相机拍到的图片,这里的state是带noise的（都是真实值！！！）
            true_pose, true_state, gt_img = agent.step(action, noise=noise)
            true_states = np.vstack((true_states, true_state))  # 不断拼接true_state

            # Given the planner's recommended action and the observation, perform state estimation. true_pose
            # is here only to benchmark performance. 估计状态

            # 这里gt_img是action后相机的观测，true_pose是动力学计算出的（带噪声）
            state_est = filter.estimate_state(gt_img, true_pose, action)

            if iter < steps - 5:
                # state estimate is 12-vector. Transform to 18-vector
                #####state_est = torch.cat([state_est[:6], vec_to_rot_matrix(state_est[6:9]).reshape(-1), state_est[9:]], dim=-1)
                # ！！！使用true-state
                true_state = torch.tensor(true_state, dtype=torch.float32, device='cuda')
                state_est = torch.cat([true_state[:6], vec_to_rot_matrix(true_state[6:9]).reshape(-1), true_state[9:]],
                                      dim=-1)

                # Let the planner know where the agent is estimated to be
                # state_est作为start-state, 去除首个self.states
                traj.update_state(state_est)

                # Replan from the state estimate，优化self.state(基于控制代价和碰撞代价)
                traj.learn_update(iter)
        return

    except KeyboardInterrupt:
        return


#########################

if __name__ == "__main__":
    ### 3DGS SPECIFIC###
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='stonetest')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    # 场景的大小
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--ply_path', type=str,
                        default='../gaussian-splatting/data/output/point_cloud/iteration_10000/point_cloud.ply')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    seed_everything(args.seed)

    gaussians = GaussianModel(1)
    scene = Scene(model.extract(args), gaussians, load_iteration=-1, shuffle=False)  # 加载场景对应的高斯,对gaussians进行初始化

    ### ESTIMATOR CONFIGS
    dil_iter = 3  # Number of times to dilate mask around features in observed image
    kernel_size = 5  # Kernel of dilation
    batch_size = 1024  # How many rays to sample in dilated mask
    lrate_relative_pose_estimation = 1e-3  # State estimator learning rate
    N_iter = 300  # Number of times to perform gradient descent in state estimator

    sig0 = 1 * np.eye(12)  # Initial state covariance
    Q = 1 * np.eye(12)  # Process noise covariance

    body_lims = np.array([
        [-0.05, 0.05],
        [-0.05, 0.05],
        [-0.02, 0.02]
    ])

    # Discretizations of sample points in x,y,z direction
    body_nbins = [10, 10, 5]

    mass = 1.  # mass of drone
    g = 10.  # gravitational constant
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # inertia tensor
    path = 'sim_img_cache/'  # Directory where pose and images are exchanged
    blend_file = '../blender_scenes/stonehenge.blend'  # Blend file of your scene

    ### PLANNER CONFIGS
    # X, Y, Z
    # STONEHENGE
    start_pos = [0.487, -0.047, 0.354]  # Starting position [x,y,z]
    # start_pos = [-0.25, 0.46, 0.2]
    # end_pos = [-0.54, -0.76, 0.17]
    end_pos = [-0.796, -0.191, 0.222]  # Goal position

    # start_pos = [-0.09999999999999926,
    #             -0.8000000000010297,
    #             0.0999999999999695]
    # end_pos = [0.10000000000000231,
    #             0.4999999999996554,
    #             0.09999999999986946]

    # Rotation vector
    start_R = [0., 0., 0.0]  # Starting orientation (Euler angles)
    end_R = [0., 0., 0.0]  # Goal orientation

    # Angular and linear velocities
    init_rates = torch.zeros(3)  # All rates

    T_final = 2.  # Final time of simulation
    steps = 20  # Number of time steps to run simulation

    planner_lr = 0.001  # Learning rate when learning a plan
    epochs_init = 2500  # Num. Gradient descent steps to perform during initial plan
    fade_out_epoch = 0
    fade_out_sharpness = 10
    epochs_update = 250  # Num. grad descent steps to perform when replanning

    ### MPC CONFIGS
    mpc_noise_mean = [0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0,
                      0]  # Mean of process noise [positions, lin. vel, angles, ang. rates]
    mpc_noise_std = [2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2]  # standard dev. of noise

    ### Integration
    start_pos = torch.tensor(start_pos).float()
    end_pos = torch.tensor(end_pos).float()

    # Change rotation vector to rotation matrix 3x3
    start_R = vec_to_rot_matrix(torch.tensor(start_R))
    end_R = vec_to_rot_matrix(torch.tensor(end_R))

    # Convert 12 dimensional to 18 dimensional vec ,3,3,9,3
    start_state = torch.cat([start_pos, init_rates, start_R.reshape(-1), init_rates], dim=0)
    end_state = torch.cat([end_pos, init_rates, end_R.reshape(-1), init_rates], dim=0)

    # Store configs in dictionary
    planner_cfg = {
        "T_final": T_final,
        "steps": steps,
        "lr": planner_lr,
        "epochs_init": epochs_init,
        "fade_out_epoch": fade_out_epoch,
        "fade_out_sharpness": fade_out_sharpness,
        "epochs_update": epochs_update,
        'start_state': start_state.to(device),
        'end_state': end_state.to(device),
        'exp_name': args.workspace,  # Experiment name
        'I': torch.tensor(I).float().to(device),
        'g': g,
        'mass': mass,
        'body': body_lims,
        'nbins': body_nbins
    }

    agent_cfg = {
        'dt': T_final / steps,  # Duration of each time step
        'mass': mass,
        'g': g,
        'I': torch.tensor(I).float().to(device)
    }

    camera_cfg = {
        'half_res': False,  # Half resolution
        'white_bg': False,  # White background,默认是True的!
        'path': path,  # Directory where pose and images are stored
        'res_x': 800,  # x resolution (BEFORE HALF RES IS APPLIED!)
        'res_y': 800,  # y resolution
        'trans': True,  # Boolean    (Transparency)
        'mode': 'RGB'  # Can be RGB-Alpha, or just RGB,默认是RGBA的!
    }

    blender_cfg = {
        'blend_path': blend_file,
        'script_path': 'viz_func.py'  # Path to Blender script
    }

    filter_cfg = {
        'dil_iter': dil_iter,
        'batch_size': batch_size,
        'kernel_size': kernel_size,
        'lrate': lrate_relative_pose_estimation,
        'N_iter': N_iter,
        'sig0': torch.tensor(sig0).float().to(device),
        'Q': torch.tensor(Q).float().to(device),
        'render_viz': True,
        'show_rate': [20, 100]
    }

    extra_cfg = {
        'mpc_noise_std': torch.tensor(mpc_noise_std),
        'mpc_noise_mean': torch.tensor(mpc_noise_mean)
    }

    # Main loop
    simulate(planner_cfg, agent_cfg, filter_cfg, extra_cfg, gaussian=gaussians, pipeline=pipeline.extract(args))

    # Visualize trajectories in Blender
    bevel_depth = 0.02  # Size of the curve visualized in blender
    subprocess.run(['blender', blend_file, '-P', 'viz_data_blend.py', '--', args.workspace, str(bevel_depth)])

    end_text = 'End of simulation'
    print(f'{end_text:.^20}')
