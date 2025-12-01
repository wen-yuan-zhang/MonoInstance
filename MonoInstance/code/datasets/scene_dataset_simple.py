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


# A simple dataset for preprocess monocular convincing
class SceneDatasetPreprocess(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 num_views=-1,
                 num_pixels=1024,
                 stage=1,
                 exp_dir='exps',
                 ):
        self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(scan_id))
        self.train_instance = False
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        self.use_warp = True

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
        self.mask_path = mask_path = os.path.join(self.instance_dir, 'instance_masks')
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
            self.depth_images.append(torch.from_numpy(depth).float())

            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())
        self.depth_images = torch.stack(self.depth_images, 0)
        self.normal_images = torch.stack(self.normal_images, 0)

        scene_scale = np.loadtxt(os.path.join(self.instance_dir, 'scene_scale/scene_scale.txt'), dtype=np.float32)
        self.gt_depth_images = []
        for i in tqdm(range(len(depth_paths))):
            # #
            # gt_depth_path = os.path.join(self.instance_dir, 'depth', '{:06d}.npy'.format(i))
            # depth = np.load(gt_depth_path).reshape(-1) * 0.32518612027593996      # scan1
            # depth = np.load(gt_depth_path).reshape(-1) * 0.2889620123314539       # scan2
            # depth = np.load(gt_depth_path).reshape(-1) * 0.3650161912057014       # scan3
            # depth = np.load(gt_depth_path).reshape(-1) * 0.28016234847769583      # scan4
            gt_depth_path = os.path.join(exp_dir, 'render_depth', '{:06d}.npy'.format(i))
            gt_depth = np.load(gt_depth_path)

            self.gt_depth_images.append(torch.from_numpy(gt_depth).float())

        print('Loading mask images...')

        all_labels = os.listdir(mask_path)
        self.mask_images = [{} for i in range(
            self.n_images)]  # n_images, [{'inst_i': np.ndarray, 'inst_j': np.ndarry, ...}, {}, {}]
        self.instances_refview_valid = [{} for i in range(len(
            all_labels))]
        self.perframe_instances = [{} for i in range(self.n_images)]
        # per-label mask
        for i in tqdm(range(len(all_labels))):
            mask_path_i = os.path.join(mask_path, str(i))
            mask_frame_names = sorted(os.listdir(mask_path_i), key=lambda x: int(x[:-4]))  # 按帧数排
            instance_pix_num = []
            all_valid_frame_id_i = []
            for mask_frame_name in mask_frame_names:
                frame_id = int(mask_frame_name[:-4])  # '2.jpg' -> 2
                _mask = cv2.imread(os.path.join(mask_path_i, mask_frame_name), cv2.IMREAD_GRAYSCALE)
                _, _mask = cv2.threshold(_mask, 127, 255, cv2.THRESH_BINARY)
                if _mask.shape[0] != self.img_res[0]:
                    _mask = cv2.resize(_mask, (self.img_res[1], self.img_res[0]), cv2.INTER_NEAREST)
                _mask = _mask / 255
                mask_tensor = torch.from_numpy(_mask.reshape(-1, 1)).float()
                # calculate for self.perframe_instances
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                tmp = cv2.erode(_mask, kernel, iterations=1)
                tmp = (tmp > 0.5)
                object_pix = np.nonzero(tmp.reshape(-1))[0]
                self.perframe_instances[frame_id].update({i: (object_pix, object_pix)})
                self.mask_images[frame_id].update({i: mask_tensor})
                all_valid_frame_id_i.append(frame_id)
            for _i, _frameid in enumerate(all_valid_frame_id_i):
                self.instances_refview_valid[i].update({_frameid: True})


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = self.uv
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

        return idx, sample, ground_truth

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

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def eval(self):
        self.return_full_img = True

    def train(self):
        self.return_full_img = False
