import os
import torch
import torch.nn.functional as F
import numpy as np
import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
from tqdm import tqdm
from collections import defaultdict
from utils import visualization
import trimesh


class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 n_src=3,
                 num_pixels=1024,
                 stage=1,
                 ):

        self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(scan_id))
        self.train_instance = False
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.n_src = n_src
        # self.refview_valid_pixnum_thres = self.total_pixels * 0.0478
        self.refview_valid_pixnum_thres = self.total_pixels * 0.001        # select appropriate reference views

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.sampling_size = num_pixels
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
        self.depth_paths = depth_paths
        self.mask_path = os.path.join(self.instance_dir, 'instance_masks')
        confidence_path = os.path.join(self.instance_dir, 'monocular_convincing')

        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        self.pose_all = torch.stack(self.pose_all, 0)

        print('Loading rgb images...')
        self.rgb_images = []
        for path in tqdm(image_paths):
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        self.rgb_images = torch.stack(self.rgb_images, 0)

        print('Loading depth and normal images...')
        self.depth_images = []
        self.normal_images = []
        for i in tqdm(range(len(depth_paths))):
            dpath, npath = depth_paths[i], normal_paths[i]
            depth = np.load(dpath)
            border=8
            depth[:border,:]=0
            depth[-border:,:]=0
            depth[:,:border]=0
            depth[:,-border:]=0
            depth = depth.reshape(-1)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())
        self.depth_images = torch.stack(self.depth_images, 0)
        self.normal_images = torch.stack(self.normal_images, 0)

        if stage == 2:  # only at the second training stage
            self.interest_instance_labels = np.loadtxt(os.path.join(self.instance_dir, 'interest_instances', 'interest_instance_labels.txt')).astype(bool)
            print(self.interest_instance_labels)
            print('Loading mask images...')
            all_labels = os.listdir(self.mask_path)  # label starts from 0
            all_image_pix_idx = np.arange(self.img_res[0] * self.img_res[1])
            self.mask_images = [{} for i in range(self.n_images)]       # n_images, [{'inst_i': np.ndarray, 'inst_j': np.ndarry, ...}, {}, {}]
            self.perframe_instances = [{} for i in range(self.n_images)]    # n_images, [{'label1': (obj_pix,dilated_pix), 'label2': (obj_pix,dilated_pix), '-1': pix},...] -1表示背景区域
            self.instances_refview_valid = [{} for i in range(len(all_labels))]    # whether is appropriate ref view. n_instances, [['frame_i':True,'frame_j':True,'frame_k':False,...], [...],...]
            self.monocular_confidence_imgs = np.ones((self.rgb_images.shape[0], self.img_res[0] * self.img_res[1]), dtype=np.float32)
            self.near_views_idx = []    # n_instances, [{'frame_i': [near_idx1,near_idx2,...],'frame_j':[near_idx1,near_idx2,...]},{},{},...]
            # per-label mask
            for i in tqdm(range(len(all_labels))):
                mask_path_i = os.path.join(self.mask_path, str(i))
                mask_frame_names = sorted(os.listdir(mask_path_i), key=lambda x: int(x[:-4]))   # sorted as frames
                instance_pix_num = []
                all_valid_frame_id_i = []
                # read all frames of the i-th instance
                for mask_frame_name in mask_frame_names:
                    frame_id = int(mask_frame_name[:-4])     # '2.jpg' -> 2
                    _mask = cv2.imread(os.path.join(mask_path_i, mask_frame_name), cv2.IMREAD_GRAYSCALE)
                    _, _mask = cv2.threshold(_mask, 127, 255, cv2.THRESH_BINARY)
                    if _mask.shape[0] != self.img_res[0]:
                        _mask = cv2.resize(_mask, (self.img_res[1], self.img_res[0]), cv2.INTER_NEAREST)
                    _mask = _mask / 255
                    mask_tensor = torch.from_numpy(_mask.reshape(-1, 1)).float()
                    # split nonzero/zero of object bbx. mask area is dilated.
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    tmp = cv2.erode(_mask, kernel, iterations=1)
                    tmp = (tmp > 0.5)
                    object_pix = np.nonzero(tmp.reshape(-1))[0]
                    # object_pix = np.nonzero(_mask.reshape(-1))[0]
                    if len(object_pix) < 100:
                        print('object pixels are too few. skip', mask_frame_name)
                        continue
                    # nonzero
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                    tmp = cv2.dilate(_mask, kernel, iterations=1)
                    tmp = (tmp > 0.5)
                    dilated_object_pix = np.nonzero(tmp.reshape(-1))[0]
                    if self.interest_instance_labels[i]:    # we only focus on foreground objects.
                        self.perframe_instances[frame_id].update({i: (object_pix, dilated_object_pix)})
                    instance_pix_num.append(len(object_pix))
                    self.mask_images[frame_id].update({i: mask_tensor})

                    # read monocular confidence map
                    confidence_path_i = os.path.join(confidence_path, str(i))
                    try:
                        confidence_data = np.load(os.path.join(confidence_path_i, str(frame_id)+'.npy')).reshape(-1)    # size is same as nonzero_idx
                        _object_pix = np.nonzero(_mask.reshape(-1))[0]
                        if self.interest_instance_labels[i]:
                            self.monocular_confidence_imgs[frame_id, _object_pix] = confidence_data
                        else:
                            self.monocular_confidence_imgs[frame_id, _object_pix] = np.ones_like(confidence_data)
                    except:
                        print('monocular confidence map not found.')
                    all_valid_frame_id_i.append(frame_id)
                # find an appropriate src view for warping
                if len(instance_pix_num) != 0:
                    for _i, _frameid in enumerate(all_valid_frame_id_i):
                        if len(all_valid_frame_id_i) < self.n_src+1:  # if available view num < self.n_src, do not warp
                            self.instances_refview_valid[i].update({_frameid: False})
                        else:
                            self.instances_refview_valid[i].update({_frameid: instance_pix_num[_i]>self.refview_valid_pixnum_thres})    # according to number of pixels

                # calculate nearby views according to camera poses
                framelist = np.array(all_valid_frame_id_i)    # all views containing this instance
                warp_views = self.cal_warp_views(i, framelist, min_angle=10, max_angle=60, max_dist=0.04)
                self.near_views_idx.append(warp_views)

            assert not np.isnan(self.monocular_confidence_imgs).any()
             # add background pixels
            for i in range(self.n_images):
                bg_pix = set(all_image_pix_idx)
                fg_pix_border = []
                for key, value in self.perframe_instances[i].items():
                    fg_pix_border += list(value[1])
                bg_pix = np.array(list(bg_pix-set(fg_pix_border)))
                self.perframe_instances[i].update({-1: bg_pix})

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        self.uv = uv
        self.uv_down = 1

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = self.uv
        if self.uv_down != 1:
            uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv[:, ::self.uv_down, ::self.uv_down]
            uv = uv.reshape(2, -1).transpose(1, 0)
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "idx": torch.tensor([idx])
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "normal": self.normal_images[idx],
            # "gt_depth": self.gt_depth_images[idx],
        }

        if not self.return_full_img:
            if self.train_instance:
                use_warp_in_this_view = False       # =False: this view contains only background, or all objects are inappropriate
                if len(self.perframe_instances[idx])>1:
                    for k,v in self.perframe_instances[idx].items():
                        if self.instances_refview_valid[k][idx]:
                            use_warp_in_this_view = True
                            break
                if not use_warp_in_this_view:
                    sampling_idx = np.random.choice(self.img_res[0]*self.img_res[1], [self.sampling_size], replace=False)
                else:
                    split_size = self.sampling_size // 4 * 3
                    # determine ray num according to the pixel proportion of each instance
                    instance_labels = []
                    instance_sampling_idx = []
                    total_instance_pix = sum([len(v[0]) for k,v in self.perframe_instances[idx].items() if k != -1])
                    if total_instance_pix / (self.img_res[0]*self.img_res[1]) > 0.7:
                        split_size = int(self.sampling_size / 10 * 9.5)   # if background pixels are too few: upsample instance pixel nums
                    for k,v in self.perframe_instances[idx].items():
                        if k != -1:
                            # sample according to mono confidence
                            instance_labels.append(k)
                            instance_samp_pixnum = int(len(v[0]) / total_instance_pix * split_size)
                            p = self.monocular_confidence_imgs[idx][v[0]]
                            p = 1 - p + 0.05     # ensure sampling on every pixel
                            p = p / sum(p)
                            _sample = np.random.choice(v[0], instance_samp_pixnum, p=p)
                            instance_sampling_idx.append(_sample)
                            # instance_sampling_idx.append(np.random.choice(v[0],instance_samp_pixnum))
                    instance_labels.append(-1)
                    if len(self.perframe_instances[idx][-1]) != 0:
                        instance_sampling_idx.append(np.random.choice(self.perframe_instances[idx][-1],self.sampling_size-split_size))
                    else:  # if no pixels in background
                        instance_sampling_idx.append(np.random.choice(self.img_res[0]*self.img_res[1], self.sampling_size-split_size))
                    sampling_idx = np.concatenate(instance_sampling_idx)
                sampling_idx = torch.LongTensor(sampling_idx)


                use_warp_idx_mask = torch.zeros(len(sampling_idx)).bool()
                # for accurate monocular: use monocular; for inaccurate monocular: use warp
                prior_thres1 = 0.2      # 0.3
                prior_thres2 = 0.2
                depthprior_weight = torch.from_numpy(self.monocular_confidence_imgs[idx][sampling_idx])
                normalprior_weight = torch.from_numpy(self.monocular_confidence_imgs[idx][sampling_idx])
                if use_warp_in_this_view:
                    depthprior_weight[:split_size] = torch.where(depthprior_weight<prior_thres1, torch.tensor([0.]), depthprior_weight)[:split_size]
                    normalprior_weight[:split_size] = torch.where(normalprior_weight<prior_thres2, torch.tensor([0.]), normalprior_weight)[:split_size]

                    # pixels that needs to warp for each instance
                    # [2,1,3] -> [[T,T,F,F,F,F], [F,F,T,F,F,F], [F,F,F,T,T,T]]
                    numbers = torch.tensor([len(idxs) for idxs in instance_sampling_idx[:-1]])
                    cumsum = numbers.cumsum(0)
                    indices = torch.arange(split_size).unsqueeze(0)
                    start_indices = torch.cat((torch.tensor([0]), cumsum[:-1]))
                    use_warp_idx_mask = (indices >= start_indices.unsqueeze(1)) & (indices < cumsum.unsqueeze(1))
                    use_warp_idx_mask = torch.cat([use_warp_idx_mask, torch.zeros(use_warp_idx_mask.shape[0], len(sampling_idx)-split_size).bool()], 1)

                sample["use_warp_idx_mask"] = use_warp_idx_mask
                sample["depthprior_weight"] = depthprior_weight
                sample["normalprior_weight"] = normalprior_weight
                # ground_truth["mask"] = mask_img[sampling_idx, :]
                # ground_truth["full_mask"] = mask_img
            else:
                sampling_idx = torch.randperm(self.total_pixels)[:self.sampling_size]
                sample["depthprior_weight"] = torch.ones(self.sampling_size)    # not train instance mode: apply depth & normal to all pixels
                sample["normalprior_weight"] = torch.ones(self.sampling_size)
            ground_truth["rgb"] = self.rgb_images[idx][sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            # warp data
            if self.train_instance and use_warp_in_this_view:
                sample["full_rgb_near"] = []
                sample["full_mask_near"] = []
                sample["pose_near"] = []
                sample["idx_near"] = []
                sample["warp_valid"] = []
                for label in instance_labels:
                    if label != -1:     # background
                        candidate_idxs_near = self.near_views_idx[label][idx]
                        if len(candidate_idxs_near) == 0:   # no nearby views
                            idxs_near = np.random.choice(np.array([idx]), [self.n_src])
                            self.instances_refview_valid[label][idx] = False
                        elif len(candidate_idxs_near) < self.n_src:
                            idxs_near = np.random.choice(candidate_idxs_near, [self.n_src])
                        else:
                            idxs_near = np.random.choice(candidate_idxs_near, [self.n_src], replace=False)
                        sample["full_rgb_near"].append(self.rgb_images[idxs_near])
                        sample["full_mask_near"].append(torch.stack([self.mask_images[_idx][label] for _idx in idxs_near], 0))
                        sample["pose_near"].append(self.pose_all[idxs_near])
                        sample["idx_near"].append(torch.from_numpy(idxs_near))
                        sample["warp_valid"].append(self.instances_refview_valid[label][idx])

                sample["full_rgb_near"] = torch.stack(sample["full_rgb_near"])
                sample["full_mask_near"] = torch.stack(sample["full_mask_near"])
                sample["pose_near"] = torch.stack(sample["pose_near"])
                sample["idx_near"] = torch.stack(sample["idx_near"])
                sample["warp_valid"] = torch.tensor(sample["warp_valid"]).bool()

            sample["uv"] = uv[sampling_idx, :]

        return idx, sample, ground_truth

    def cal_warp_views(self, inst_id, framelist, min_angle=10, max_angle=60, max_dist=0.04, ray_center='instance'):
        # calculate nearby views according to center ray angles and distances
        # inst_id: instance label, framelist: all frames containing inst_id, invalid_idxs: views not appropriate to warp, ray_center='instance'/'image': where to emit rays
        # filter that: min_angle<angle<max_angle, dist<max_dist
        cam_locs = torch.zeros(len(framelist), 3).cuda()
        ray_dirs = torch.zeros(len(framelist), 3).cuda()
        for _i, frame_id in enumerate(framelist):
            inst_mask = self.mask_images[frame_id][inst_id].reshape(self.img_res[0], self.img_res[1])
            nonzero = torch.nonzero(inst_mask)
            # use instance center is better than image center
            if ray_center=='instance':
                min_x, max_x = nonzero[:,1].min(), nonzero[:,1].max()
                min_y, max_y = nonzero[:,0].min(), nonzero[:,0].max()
            elif ray_center=='image':
                min_x, max_x = 0, self.img_res[1]
                min_y, max_y = 0, self.img_res[0]
            center_x, center_y = (min_x+max_x)/2, (min_y+max_y)/2
            uv = torch.tensor([[[center_x,center_y]]]).cuda()
            ray_dir, cam_loc = rend_util.get_camera_params(uv, self.pose_all[None,frame_id].cuda(), self.intrinsics_all[0].unsqueeze(0).cuda())
            ray_dirs[_i] = ray_dir.squeeze()
            cam_locs[_i] = cam_loc.squeeze()

        p1 = cam_locs.unsqueeze(1)  # Shape: (N, 1, 3)
        d1 = ray_dirs.unsqueeze(1)  # Shape: (N, 1, 3)
        p2 = cam_locs.unsqueeze(0)  # Shape: (1, N, 3)
        d2 = ray_dirs.unsqueeze(0)  # Shape: (1, N, 3)
        # Compute cross products and dot products
        cross_prod = torch.cross(d1, d2, dim=2)
        dot_prod = torch.sum(d1 * d2, dim=2)
        # Calculate the norm of the cross product
        cross_norm = torch.norm(cross_prod, dim=2)
        # Calculate distances
        # Distance = |(p2 - p1) dot n| / ||n||
        dp = p2 - p1
        numerator = torch.abs(torch.sum(dp * cross_prod, dim=2))
        distances = numerator / cross_norm
        # Replace infinities and NaNs (parallel lines) with zero distance
        distances = torch.where(torch.isfinite(distances), distances, torch.zeros_like(distances))
        # Calculate angles in degrees
        angles = torch.atan2(cross_norm, dot_prod) * 180.0 / torch.pi

        near_view_selected = (angles > min_angle) & (angles < max_angle) & (distances < max_dist)
        near_view_idx = {framelist[i]: framelist[near_view_selected[i].cpu().numpy()] for i in range(near_view_selected.shape[0])}
        return near_view_idx

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def eval(self):
        self.return_full_img = True

    def train(self):
        self.return_full_img = False

