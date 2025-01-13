import numpy as np
import json
import time
import cv2
import matplotlib.pyplot as plt

import imageio
from utils.system_utils import searchForMaxIteration

from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from demos.render import *
from nav.math_utils import vec_to_rot_matrix, rot_x, nerf_matrix_to_ngp_torch, nearestPD, calcSE3Err


# 要修改的有measurement_fn，render_from_pose


def find_POI(img_rgb, render=False):  # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    """
    检测图像中SIFT关键点
    """
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    # Initiate ORB detector
    # orb = cv2.ORB_create()
    # find the keypoints with ORB
    # keypoints2 = orb.detect(img_gray,None)

    if render:
        feat_img = cv2.drawKeypoints(img_gray, keypoints, img)
    else:
        feat_img = None

    # keypoints = keypoints + keypoints2
    # keypoints = keypoints2

    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)

    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)

    extras = {
        'features': feat_img
    }

    return xy, extras  # pixel coordinates


# def nearestPD(A):
#     """Find the nearest positive-definite matrix to input
#     A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
#     credits [2].
#     [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
#     [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
#     matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
#     """

#     B = (A + A.T) / 2
#     _, s, V = la.svd(B)

#     H = np.dot(V.T, np.dot(np.diag(s), V))

#     A2 = (B + H) / 2

#     A3 = (A2 + A2.T) / 2

#     if isPD(A3):
#         return A3

#     spacing = np.spacing(la.norm(A))
#     # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
#     # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
#     # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
#     # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
#     # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
#     # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
#     # `spacing` will, for Gaussian random matrixes of small dimension, be on
#     # othe order of 1e-16. In practice, both ways converge, as the unit test
#     # below suggests.
#     I = np.eye(A.shape[0])
#     k = 1
#     while not isPD(A3):
#         mineig = np.min(np.real(la.eigvals(A3)))
#         A3 += I * (-mineig * k**2 + spacing)
#         k += 1

#     return A3

# def isPD(B):
#     """Returns true when input is positive-definite, via Cholesky"""
#     try:
#         _ = la.cholesky(B)
#         return True
#     except la.LinAlgError:
#         return False

# def calcSO3Err(R_gt, R_est):
#     ''' angle between two rotation matrices (in degrees) '''
#     rotDiff = np.dot(R_gt, np.transpose(R_est))
#     trace = np.trace(rotDiff) 
#     if trace < -1 and (-1 - trace) < 0.0001:
#         return np.rad2deg(np.arccos(-1))
#     if trace > 3 and (trace - 3) < 0.0001:
#         return np.rad2deg(np.arccos(1))
#     return np.rad2deg(np.arccos((trace-1.0)/2.0))

# def calcSE3Err(T_gt, T_est):
#     ''' translation err & angle between two rotation matrices (in degrees) '''
#     ang_err_deg = calcSO3Err(T_gt[0:3, 0:3], T_est[0:3, 0:3])
#     t_err = np.linalg.norm(T_gt[0:3, 3] - T_est[0:3, 3])
#     return t_err, ang_err_deg

class Pose_to_view:
    def __init__(self, pose):
        # c2w = pose#.copy()
        # c2w[:3, 1:3] *= -1
        # w2c = np.linalg.inv(c2w)
        # self.R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        # self.T = w2c[:3, 3]

        c2w = pose
        c2w[:3, 1:3] *= -1
        w2c = torch.inverse(c2w)
        self.R = torch.transpose(w2c[:3, :3], 0, 1)
        self.T = w2c[:3, 3]
        self.image_height = 800
        self.image_width = 800
        self.FoVx = 0.6911112070083618
        self.FoVy = 0.6911112070083618
        self.world_view_transform = getWorld2View2(self.R, self.T).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=self.FoVx, fovY=self.FoVy).transpose(0,
                                                                                                                     1).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)


# pipeline.extract(args)

# parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)
#
#     # Initialize system state (RNG)
#     safe_state(args.quiet)


class Estimator():
    def __init__(self, filter_cfg, agent, start_state, filter=True, get_rays_fn=None, render_fn=None, gaussian=None,
                 white_background=None, pipeline=None) -> None:

        # Parameters
        self.batch_size = filter_cfg['batch_size']  # 1024
        self.kernel_size = filter_cfg['kernel_size']  # 5
        self.dil_iter = filter_cfg['dil_iter']  # 3
        # add
        self.gaussian = gaussian
        self.white_background = white_background
        self.pipeline = pipeline

        self.lrate = filter_cfg['lrate']  # 0.001

        self.agent = agent
        self.is_filter = filter
        # 可视化渲染过程
        self.render_viz = filter_cfg['render_viz']
        if self.render_viz:
            self.f, self.axarr = plt.subplots(1, 3, figsize=(15, 50))

        self.show_rate = filter_cfg['show_rate']
        self.error_print_rate, self.render_rate = self.show_rate  # 误差打印频率-20，渲染频率-100

        # State initial estimate at time t=0
        self.xt = start_state  # Size 12,包括pos,R（旋转向量），角速度，线速度,各3维
        self.sig = filter_cfg['sig0']  # State covariance 12x12 状态协方差矩阵，这里默认是单位阵
        self.Q = filter_cfg['Q']  # Process noise covariance 12x12 ，过程噪声协方差矩阵，这里默认是单位阵
        self.iter = filter_cfg['N_iter']

        # NERF SPECIFIC CONFIGS  修改！！
        self.get_rays = get_rays_fn
        self.render_fn = render_fn

        # Storage for plots
        self.losses = None
        self.covariance = None
        self.state_estimate = None
        self.states = None
        self.action = None

        self.iteration = 0

    def estimate_relative_pose(self, sensor_image, start_state, sig, obs_img_pose=None):
        # start-state is 12-vector
        ##注意实际中obs_img_pose是未知的，这里仅仅为了测试性能！！！！
        obs_img_noised = sensor_image
        W_obs = sensor_image.shape[0]
        H_obs = sensor_image.shape[1]

        # find points of interest of the observed image
        # POI-关键的坐标数组，extras-图像
        POI, extras = find_POI(obs_img_noised,
                               render=self.render_viz)  # xy pixel coordinates of points of interest (N x 2)
        # 找到多少个关键点
        print(f'Found {POI.shape[0]} features')
        ### IF FEATURE DETECTION CANT FIND POINTS, RETURN INITIAL
        if len(POI.shape) == 1:
            self.losses = []
            self.states = []
            error_text = 'Feature Detection Failed.'
            print(f'{error_text:.^20}')
            return start_state.clone().detach(), False

        obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)
        obs_img_noised = torch.tensor(obs_img_noised).cuda()

        # create meshgrid from the observed image
        coords = np.asarray(
            np.stack(np.meshgrid(np.linspace(0, H_obs - 1, H_obs), np.linspace(0, W_obs - 1, W_obs)), -1), dtype=int)

        # create sampling mask for interest region sampling strategy
        interest_regions = np.zeros((H_obs, W_obs,), dtype=np.uint8)
        interest_regions[POI[:, 0], POI[:, 1]] = 1
        I = self.dil_iter
        interest_regions = cv2.dilate(interest_regions, np.ones((self.kernel_size, self.kernel_size), np.uint8),
                                      iterations=I)
        interest_regions = np.array(interest_regions, dtype=bool)
        interest_regions = coords[interest_regions]

        # Optimzied state is 12 vector initialized as the starting state to be optimized. Add small epsilon to avoid singularities
        optimized_state = torch.tensor([0.1, 0, 0.5, 0.3, 0, 0], dtype=torch.float32, device='cuda', requires_grad=True)
        # optimized_state.requires_grad_(True)

        # Add velocities, omegas, and pose object to optimizer
        if self.is_filter is True:
            optimizer = torch.optim.Adam(params=[optimized_state], lr=self.lrate, betas=(0.9, 0.999))
        else:
            raise ('Not implemented')

        # calculate initial angles and translation error from observed image's pose
        if obs_img_pose is not None:
            pose = torch.eye(4)
            pose[:3, :3] = vec_to_rot_matrix(optimized_state[6:9])
            pose[:3, 3] = optimized_state[:3]
            print('initial error', calcSE3Err(pose.detach().cpu().numpy(), obs_img_pose))

        # Store data
        losses = []
        states = []

        for k in range(self.iter):
            optimizer.zero_grad()
            print('interest_regions.shape[0] is :', interest_regions.shape[0])
            print('self.batch_size is :', self.batch_size)
            rand_inds = np.random.choice(interest_regions.shape[0], size=self.batch_size, replace=False)
            batch = interest_regions[rand_inds]

            # pix_losses.append(loss.clone().cpu().detach().numpy().tolist())
            # Add dynamics loss
            ##start_state-是根据前一时刻计算出的state ，obs_img_noised是相机观测
            loss = self.measurement_fn(optimized_state, start_state, sig, obs_img_noised, batch)

            losses.append(loss.item())  # nerf在这里的Loss是0.08，0.07
            states.append(optimized_state.clone().cpu().detach().numpy().tolist())

            loss.backward()
            optimizer.step()

            print('Loss: ', loss)
            print('State', optimized_state)

            # NOT IMPLEMENTED: EXPONENTIAL DECAY OF LEARNING RATE
            # new_lrate = self.lrate * (0.8 ** ((k + 1) / 100))
            # new_lrate = extra_arg_dict['lrate'] * np.exp(-(k)/1000)
            # for param_group in optimizer.param_groups:
            #    param_group['lr'] = new_lrate

            # print results periodically
            if obs_img_pose is not None and ((k + 1) % self.error_print_rate == 0 or k == 0):
                print('Step: ', k)
                print('Loss: ', loss)
                print('State', optimized_state)

                with torch.no_grad():
                    pose = torch.eye(4)
                    pose[:3, :3] = vec_to_rot_matrix(optimized_state[6:9])  # 旋转向量到旋转矩阵
                    pose[:3, 3] = optimized_state[:3]
                    pose_error = calcSE3Err(pose.detach().cpu().numpy(), obs_img_pose)  # 待优化的pose和图像观察的pose求error
                    print('error', pose_error)
                    print('-----------------------------------')

                    if (k + 1) % self.render_rate == 0 and self.render_viz:
                        # rgb = self.render_from_pose(pose)#从待优化的pose下渲染得到的图像
                        # rgb 1个tensor,1个np
                        if self.gaussian:
                            # pose_test = [
                            #     [0.890663743019104, -0.026803040876984596, 0.4538719058036804, 1.134679731443595],
                            #     [0.45466262102127075, 0.05250595882534981, -0.8891147375106812, -2.2227868575679617],
                            #     [0.0, 0.9982608556747437, 0.058951497077941895, 0.14737873955980724],
                            #     [0.0, 0.0, 0.0, 1.0]]
                            # rgb = self.render_from_pose_gaussian(pose.detach().cpu().numpy())
                            rgb = self.render_from_pose_gaussian(pose)
                            # torchvision.utils.save_image(img,
                            # os.path.join('/home/k/图片', '{0:05d}'.format(10086) + ".png"))
                        else:

                            rgb = self.render_from_pose(pose)
                        rgb = torch.squeeze(rgb).cpu().detach().numpy()

                        # Add keypoint visualization ，把关键点可视化出来
                        render = rgb.reshape((obs_img_noised.shape[0], obs_img_noised.shape[1], -1))
                        gt_img = obs_img_noised.cpu().numpy()  # 传感器图像作为ground_truth
                        render[batch[:, 0], batch[:, 1]] = np.array([0., 1., 0.])
                        gt_img[batch[:, 0], batch[:, 1]] = np.array([0., 1., 0.])

                        self.f.suptitle(
                            f'Time step: {self.iteration}. Grad step: {k + 1}. Trans. error: {pose_error[0]} m. Rotate. error: {pose_error[1]} deg.')
                        self.axarr[0].imshow(gt_img)
                        self.axarr[0].set_title('Ground Truth')

                        self.axarr[1].imshow(extras['features'])
                        self.axarr[1].set_title('Features')

                        self.axarr[2].imshow(render)
                        if self.gaussian:
                            self.axarr[2].set_title('3DGS Render')
                        else:
                            self.axarr[2].set_title('NeRF Render')

                        plt.pause(1)

        print("Done with main relative_pose_estimation loop")
        self.target = obs_img_noised
        self.batch = batch

        self.losses = losses
        self.states = states
        return optimized_state.clone().detach(), True

    def measurement_fn(self, state, start_state, sig, target, batch):
        # Process loss. !!!!!目标图像target和state下的渲染图在感兴趣的区域比较求loss!!!!!
        # loss_dyn = mahalanobis(state, start_state, sig)#sig-协方差矩阵,这里计算的是state和start_state的马氏距离

        H, W, _ = target.shape

        # Assuming the camera frustrum is oriented in the body y-axis. The camera frustrum is in the -z axis
        # in its own frame, so we need a 90 degree rotation about the x-axis to transform 
        # TODO: Check this, doesn't look right. Should be camera to world
        # R = vec_to_rot_matrix(state[6:9])
        # rot = rot_x(torch.tensor(np.pi/2)) @ R[:3, :3]#绕x轴旋90度

        # pose, trans = nerf_matrix_to_ngp_torch(rot, state[:3])#state[:3]是世界坐标下平移向量

        # new_pose = torch.eye(4)
        # new_pose[:3, :3] = pose
        # new_pose[:3, 3] = trans
        if self.gaussian:
            pose = torch.eye(4)
            pose[:3, :3] = vec_to_rot_matrix(state[3:6])
            pose[:3, 3] = state[:3]
            bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]  # 是否白色背景，命令行参数
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            # 改为在感兴趣区域求loss
            # pose_np = pose.detach().numpy()
            view = Pose_to_view(pose)
            img = render(view, self.gaussian, self.pipeline, background)["render"]
            tep = img.permute(1, 2, 0)  # shape变为（H，W，3）
            rgb = tep[batch[:, 0], batch[:, 1]]  # 选择特征点区域

        else:
            rays = self.get_rays(new_pose.reshape((1, 4, 4)))
            # 这里是在特定像素处求Loss,这里选择了1024个像素点
            rays_o = rays["rays_o"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]
            rays_d = rays["rays_d"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]

            output = self.render_fn(rays_o.reshape((1, -1, 3)), rays_d.reshape((1, -1, 3)))  # 得到一张图
            # output also contains a depth channel for use with depth data if one chooses
            # gaussian渲染的output_shape和这里好像不一样？？
            rgb = output['image'].reshape((-1, 3))  # (1024,3)
        # target变成(1024,3)
        target = target[batch[:, 0], batch[:, 1]]  # TODO: Make sure target size is [H, W, 3]

        loss_rgb = torch.nn.functional.mse_loss(rgb, target)

        loss = loss_rgb  # + loss_dyn

        return loss * 100

    def render_from_pose_gaussian(self, pose):
        """
        从指定pose得到渲染rgb图,pose是世界坐标下4*4变换矩阵
        """
        # img.permute(1, 2, 0).reshape((1, -1, 3))
        bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]  # 是否白色背景，命令行参数
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # gs render!
        view = Pose_to_view(pose)
        img = render(view, self.gaussian, self.pipeline, background)["render"]
        # !!!转化shape!
        rgb = img.permute(1, 2, 0).reshape((-1, 3))
        # 直接在这显示出来和nerf对比,返回tensor
        return rgb

    def render_from_pose(self, pose):
        """
        从指定pose得到渲染rgb图
        """

        rot = rot_x(torch.tensor(np.pi / 2)) @ pose[:3, :3]
        trans = pose[:3, 3]
        pose, trans = nerf_matrix_to_ngp_torch(rot, trans)

        new_pose = torch.eye(4)
        new_pose[:3, :3] = pose
        new_pose[:3, 3] = trans

        rays = self.get_rays(new_pose.reshape((1, 4, 4)))

        output = self.render_fn(rays["rays_o"], rays["rays_d"])
        # output also contains a depth channel for use with depth data if one chooses

        rgb = torch.squeeze(output['image'])
        # (640000,3) ,范围从0.0005 到1
        return rgb

    def estimate_state(self, sensor_img, obs_img_pose, action):
        # img = imageio.imread(img_path)
        # img = (np.array(img) / 255.0).astype(np.float32)
        # Computes Jacobian w.r.t dynamics are time t-1. Then update state covariance Sig_{t|t-1}.
        # Perform grad. descent on J = measurement loss + process loss
        # Compute state covariance Sig_{t} by hessian at state at time t.
        # 输入应用action后观测图像，以其相应pose,动作，估计系统状态
        # with torch.no_grad():
        # Propagated dynamics. x t|t-1
        # xt should be 12-vector
        ####self.xt = self.agent.drone_dynamics(self.xt, action)
        # self.action = action.clone().cpu().numpy().tolist() !!!!bug!!!
        ####self.action = action.clone().cpu().detach().numpy().tolist()

        # State estimate at t-1 is self.xt. Find jacobian wrt dynamics
        t1 = time.time()

        # A should be 12 x 12
        ####A = torch.autograd.functional.jacobian(lambda x: self.agent.drone_dynamics(x, action), self.xt)

        t2 = time.time()
        print('Elapsed time for Jacobian', t2 - t1)

        # Propagate covariance
        # Covariance should be 12 x 12,卡尔曼中先验误差协方差矩阵
        sig_prop = self.Q

        # Argmin of total cost. Encapsulate this argmin optimization as a function call
        then = time.time()
        # xt is 12-vector
        xt, success_flag = self.estimate_relative_pose(sensor_img, self.xt.clone().detach(), sig_prop,
                                                       obs_img_pose=obs_img_pose)

        print('Optimization step for filter', time.time() - then)

        # Hessian to get updated covariance
        t3 = time.time()

        if self.is_filter is True and success_flag is True:
            # xt is 12-vector
            # Hessian is 12x12
            hess = torch.autograd.functional.hessian(
                lambda x: self.measurement_fn(x, self.xt.clone().detach(), sig_prop, self.target, self.batch),
                xt.clone().detach())

            # Turn covariance into positive definite
            hess_np = hess.cpu().detach().numpy()
            hess = nearestPD(hess_np)  # 找最近正定矩阵

            t4 = time.time()
            print('Elapsed time for hessian', t4 - t3)

            # Update state covariance
            self.sig = torch.inverse(torch.tensor(hess))

        self.xt = xt

        self.covariance = self.sig.clone().cpu().detach().numpy().tolist()
        self.state_estimate = self.xt.clone().cpu().detach().numpy().tolist()

        # save_path = self.basefolder / "estimator_data" / f"step{self.iteration}.json"
        # self.save_data(save_path)

        self.iteration += 1

        return self.xt.clone().detach()

    def save_data(self, filename):
        data = {}

        data['loss'] = self.losses
        data['covariance'] = self.covariance
        data['state_estimate'] = self.state_estimate
        data['grad_states'] = self.states
        data['action'] = self.action

        with open(filename, "w+") as f:
            json.dump(data, f, indent=4)
        return


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
    pipeline = pipeline.extract(args)
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # "cuda" if torch.cuda.is_available() else
    # device = torch.device( "cpu")

    sig0 = 1 * np.eye(12)
    Q = 1 * np.eye(12)
    filter_cfg = {
        'dil_iter': 3,
        'batch_size': 1024,
        'kernel_size': 5,
        'lrate': 0.1,
        'N_iter': 300,
        'sig0': torch.tensor(sig0).float().to(device),
        'Q': torch.tensor(Q).float().to(device),
        'render_viz': True,
        'show_rate': [20, 40]
    }
    img_path = '//test.png'
    img = imageio.v2.imread(img_path)
    img = (np.array(img) / 255.0).astype(np.float32)
    img = (np.array(img) * 255.).astype(np.uint8)

    test = Estimator(filter_cfg, None, torch.randn(6, device=device, dtype=torch.float32), gaussian=gaussians,
                     pipeline=pipeline)
    print(test.xt)
    est_state = test.estimate_state(img, None, torch.tensor([1, 1, 1, 1], dtype=torch.float32))
    print(est_state)
