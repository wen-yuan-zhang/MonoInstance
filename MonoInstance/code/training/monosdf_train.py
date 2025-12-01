import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import trimesh
import cv2
import shutil
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth
import torch.distributed as dist
from process import process_confidence


class MonoSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.stage = kwargs['stage']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id']
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        is_continue = kwargs['is_continue']

        if self.GPU_INDEX == 0:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            self.methodname = kwargs['methodname']
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.methodname))

            self.plots_dir = os.path.join(self.expdir, self.methodname, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.methodname, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.methodname, 'runconf.conf')))
            os.makedirs(os.path.join(self.expdir, self.methodname, 'records'), exist_ok=True)
            os.system("""cp -r {0} "{1}/" """.format('model', os.path.join(self.expdir, self.methodname, 'records')))
            os.system("""cp -r {0} "{1}/" """.format('training', os.path.join(self.expdir, self.methodname, 'records')))
            os.system("""cp -r {0} "{1}/" """.format('utils', os.path.join(self.expdir, self.methodname, 'records')))
            os.system("""cp -r {0} "{1}/" """.format('datasets', os.path.join(self.expdir, self.methodname, 'records')))
            os.system("""cp -r {0} "{1}/" """.format('confs', os.path.join(self.expdir, self.methodname, 'records')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        dataset_conf['stage'] = kwargs['stage']
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        # if preprocess: load scene_dataset_simple for calculating monocular confidence and background segmentation
        if kwargs['preprocess']:
            dataset_conf['exp_dir'] = os.path.join(self.expdir, self.methodname)
            train_dataset = utils.get_class("datasets.scene_dataset_simple.SceneDatasetPreprocess")(**dataset_conf)
            process_confidence.analyze_monocular_confidence(train_dataset,uv_down=4)
            process_confidence.detach_background(train_dataset)
            exit()

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8)  # TODO: debug 0/train 8
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        
        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            checkpoint_name = kwargs['checkpoint_name']
            old_checkpnts_dir = os.path.join(self.expdir, checkpoint_name, 'checkpoints')
            checkpoint_epoch = kwargs['checkpoint_epoch']
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', checkpoint_epoch+".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']
            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', checkpoint_epoch+".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, checkpoint_epoch+".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        # self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):

        print("training...")
        if self.GPU_INDEX == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        self.iter_step = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0 and epoch != self.start_epoch:   # don't eval at 1 epoch
                self.model.eval()

                self.train_dataset.eval()
                plot_data = None

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                # do not render imgs for efficiency
                # model_input["intrinsics"] = model_input["intrinsics"].cuda()
                # model_input["uv"] = model_input["uv"].cuda()
                # model_input['pose'] = model_input['pose'].cuda()
                #
                # split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels*2)
                # res = []
                # for s in tqdm(split):
                #     out = self.model(s, indices)
                #     d = {'rgb_values': out['rgb_values'].detach(),
                #          'normal_map': out['normal_map'].detach(),
                #          'depth_values': out['depth_values'].detach(),
                #          'cam_locs': out['cam_loc'].detach(),
                #          'ray_dirs': out['ray_dirs'].detach(), }
                #     if 'rgb_un_values' in out:
                #         d['rgb_un_values'] = out['rgb_un_values'].detach()
                #     res.append(d)
                #     del out
                #
                # batch_size = ground_truth['rgb'].shape[0]
                # model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                # plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'],
                #                                ground_truth['normal'], ground_truth['depth'])

                plt.plot(self.model.module.implicit_network,
                         indices,
                         plot_data,
                         self.plots_dir,
                         epoch,
                         self.img_res,
                         **self.plot_conf
                         )
                torch.cuda.empty_cache()
                self.model.train()
                print('eval completed.')

            # switch to stage 2 at 1/3 epochs with mono confidence: importance sampling; prior weight; rgb warp
            if epoch > self.nepochs - 10:
                self.train_dataset.train_instance = True
                self.loss.warp_weight = 0.      # remove warp for finetune, but with mono weight+importance sampling
                self.plot_freq = 1
                self.loss.depth_weight = 0.1
                self.loss.normal_cos_weight = 0.005
                self.loss.normal_l1_weight = 0.005
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-5
            elif epoch > self.nepochs // 4:
                self.train_dataset.train_instance = True
                if self.stage == 1:     # stage 1 training ends. save checkpoints and render low-res depths
                    self.save_checkpoints(epoch)
                    print('stage 1 training finished. rendering depth...')
                    self.render_all_frame_depth()
                    exit()
            else:
                self.train_dataset.train_instance = False     # first 1/3 epochs

            # reduce normal prior weight to enhance details for 1-10 epochs
            if epoch < 10:
                self.loss.normal_l1_weight = self.conf.get_float('loss.normal_l1_weight') / 5
                self.loss.normal_cos_weight = self.conf.get_float('loss.normal_cos_weight') / 5
            else:
                self.loss.normal_l1_weight = self.conf.get_float('loss.normal_l1_weight')
                self.loss.normal_cos_weight = self.conf.get_float('loss.normal_cos_weight')

            self.train_dataset.train()
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                for key in model_input.keys():      # send data to cuda
                    model_input[key] = model_input[key].cuda()

                self.optimizer.zero_grad()

                model_outputs = self.model(model_input, indices)

                loss_output = self.loss(model_outputs, ground_truth, model_input)
                loss = loss_output['loss']
                loss.backward()
                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))

                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss.item(), self.iter_step)
                self.writer.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), self.iter_step)
                self.writer.add_scalar('Loss/smooth_loss', loss_output['smooth_loss'].item(), self.iter_step)
                self.writer.add_scalar('Loss/depth_loss', loss_output['depth_loss'].item(), self.iter_step)
                self.writer.add_scalar('Loss/normal_l1_loss', loss_output['normal_l1'].item(), self.iter_step)
                self.writer.add_scalar('Loss/normal_cos_loss', loss_output['normal_cos'].item(), self.iter_step)

                self.writer.add_scalar('Statistics/beta', self.model.module.density.get_beta().item(), self.iter_step)
                self.writer.add_scalar('Statistics/alpha', 1. / self.model.module.density.get_beta().item(), self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)

                if self.Grid_MLP:
                    self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                    self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                    self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)

                self.train_dataset.train()
                self.scheduler.step()

                if self.GPU_INDEX == 0 and self.iter_step % 20 == 0:
                    print('{} [{}] : loss={:.3f}, rgb_loss={:.3f}, warp_loss={:.3f}, depth={:.3f}, norm={:.3f}'
                          .format(self.methodname, epoch, loss.item(), loss_output['rgb_loss'].item(),
                                  loss_output['rgb_warp_loss'].item(), loss_output['depth_loss'].item(), loss_output['normal_l1'].item()))

        if self.GPU_INDEX == 0:
            self.save_checkpoints(epoch)

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        # depth_map = depth_map * scale + shift     # no scale shift
        depth_map = depth_map
        
        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()

    def get_point_cloud_ours(self, depth, model_input, model_outputs):
        intrinsics = model_input["intrinsics"]
        uv = model_input["uv"]
        pose = model_input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        depth = depth.reshape(-1,1)
        depth_points = cam_loc + ray_dirs * depth / depth_scale

        return depth_points.detach().cpu().numpy()

    def visual_pred_gt_depth(self, model_input, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb']
        depth_gt = ground_truth['depth']
        batch_size, num_samples, _ = rgb_gt.shape

        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_gt, depth_map[..., None], depth_gt > 0.)
        rescaled_depth_gt = depth_gt * scale + shift
        # rescaled_depth_gt = depth_gt    # no scale-shift align for metric3d?

        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud_ours(depth, model_input, model_outputs)
        pred_points = pred_points[:,:3]

        gt_depth = rescaled_depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud_ours(gt_depth, model_input, model_outputs)
        gt_points = gt_points[:, :3]

        # # calculate depth-normal
        # depth_points = gt_points.reshape(self.img_res[0], self.img_res[1], 3)
        # depth_points = torch.tensor(depth_points, dtype=torch.float32)
        # output = torch.zeros_like(depth_points)
        # dx = torch.cat([depth_points[2:, 1:-1] - depth_points[:-2, 1:-1]], dim=0)
        # dy = torch.cat([depth_points[1:-1, 2:] - depth_points[1:-1, :-2]], dim=1)
        # normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        # output[1:-1, 1:-1, :] = normal_map

        if not os.path.exists('debug'):
            os.makedirs('debug')
        # cv2.imwrite('debug/gtdepth_normal.jpg', ((output+1)/2*255).int().detach().cpu().numpy()[...,[2,1,0]])
        trimesh.Trimesh(pred_points).export('debug/pred_points.ply')
        trimesh.Trimesh(gt_points).export('debug/gt_points.ply')

    def render_all_frame_depth(self, uv_down=4):
        self.train_dataset.eval()
        plot_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                      shuffle=False, collate_fn=self.train_dataset.collate_fn)
        self.train_dataset.uv_down = uv_down
        iterator = iter(plot_dataloader)
        for i in tqdm(range(len(plot_dataloader))):
            indices, model_input, ground_truth = next(iterator)
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            with torch.no_grad():
                split = utils.split_input(model_input, self.total_pixels//(uv_down**2), n_pixels=20000)    # 100000 will oom?
                res = []
                for s in split:
                    out = self.model(s, indices, only_return_depth=True)
                    d = {'depth_values': out['depth_values'].detach().cpu()}
                    res.append(d)

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels//(uv_down**2), batch_size)
            # debug
            # # model_outputs['depth_values'] = torch.from_numpy(np.load(self.train_dataset.depth_paths[indices.item()]).reshape(-1, 1)).float().cuda()
            # self.visual_pred_gt_depth(model_input, model_outputs, ground_truth)
            depth_map = model_outputs['depth_values'].reshape(self.train_dataset.img_res[0]//uv_down, self.train_dataset.img_res[1]//uv_down)
            os.makedirs(os.path.join(self.expdir, self.methodname, 'render_depth'),exist_ok=True)
            np.save(os.path.join(self.expdir, self.methodname, 'render_depth', f'{indices.item():06d}.npy'), depth_map.cpu().numpy())