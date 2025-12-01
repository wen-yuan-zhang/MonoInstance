import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T


class BlendSwapDataset(Dataset):
    def __init__(self, conf, split='train', N_vis=-1):
        self.device = torch.device('cuda')
        self.N_vis = N_vis
        self.root_dir = conf.get_string('data_dir')
        scene = conf.get_string('scene')
        self.root_dir = os.path.join(self.root_dir, scene)
        
        self.split = split
        self.is_stack = False
        self.downsample = 1.0
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        # self.define_proj_mat()

        self.white_bg = True

    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = int(self.meta['w'] / self.downsample), int(self.meta['h'] / self.downsample)
        self.img_wh = [w, h]
        self.focal_x = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal_y = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y'])  # original focal length
        self.cx, self.cy = self.meta['cx'], self.meta['cy']

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.intrinsics = torch.tensor([[self.focal_x, 0, self.cx], [0, self.focal_y, self.cy], [0, 0, 1]]).float()

        self.poses = []
        self.masks = []
        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#
            frame = self.meta['frames'][i]
            img_path = frame['file_path'].replace('rgb', 'mask')
            img_path = os.path.join(self.root_dir, img_path)
            if os.path.exists(img_path):
                mask_img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
                mask = torch.tensor(mask_img[:,:,0] > 128)
                mask = self.mask_shrink(mask)
                self.masks.append(mask)
                raw_img = cv2.imread(os.path.join(self.root_dir, frame['file_path']))
                paint_img = np.zeros([raw_img.shape[0], raw_img.shape[1], 3])
                paint_img[mask] = raw_img[mask]
                cv2.imwrite('debug/raw_img_mask.jpg', paint_img)
                pose = np.array(frame['transform_matrix'])
                pose = pose @ self.blender2opencv
                c2w = torch.FloatTensor(pose)
                self.poses.append(c2w)

        self.poses = torch.stack(self.poses)    # (N, 4, 4)
        self.masks = torch.stack(self.masks)     # (N, H, W)
        self.h, self.w = h, w

    def define_transforms(self):
        self.transform = T.ToTensor()

    # def define_proj_mat(self):
    #     self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def __len__(self):
        return self.poses.shape[0]

    def mask_shrink(self, mask):
        # 把mask整体往内收缩一圈（上下左右只要有False就删掉）
        # 中、上、下、左、右取最小值
        # h, w = mask.shape
        # mask = mask.int()
        # mask_up = torch.cat([mask[0:1, :], mask[:-1, :]], 0)
        # mask_left = torch.cat([mask[:, 0:1], mask[:, :-1]], 1)
        # mask_right = torch.cat([mask[:, 1:], mask[:, -1:]], 1)
        # mask_down = torch.cat([mask[1:, :], mask[-1:, :]], 0)
        # mask_alldir = torch.stack([mask, mask_up, mask_left, mask_right, mask_down], -1)
        # new_mask, _ = torch.min(mask_alldir, -1)
        # return new_mask.bool()

        kernel = np.ones((5,5)) / 25
        mask = cv2.filter2D(np.array(mask, dtype=np.uint8) * 255, -1, kernel)
        new_mask = np.where(mask == 255, 1, 0)
        return torch.tensor(new_mask, dtype=torch.bool)



    # def gen_random_rays_at(self, img_idx, batch_size):
    #     pixels_x = torch.randint(low=0, high=self.w, size=[batch_size])
    #     pixels_y = torch.randint(low=0, high=self.h, size=[batch_size])
    #     color = self.all_rgbs[img_idx] # [h, w, 3]
    #     color = color[(pixels_y, pixels_x)]  # [batch_size, 3]
    #     mask = torch.ones_like(color, dtype=torch.float)
    #     all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6) # [h, w, 6]
    #     rand_rays = all_rays[(pixels_y, pixels_x)] # [batch_size, 6]
    #     return torch.cat([rand_rays, color, mask[:, :1]], dim=-1).to(self.device)

    def gen_random_rays_at(self, img_idx, batch_size, mode):
        if mode == 'batch':
            pixels_x = torch.randint(low=0, high=self.w, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.h, size=[batch_size])

            color = self.all_rgbs[img_idx] # [h, w, 3]
            color = color[(pixels_y, pixels_x)]  # [batch_size, 3]
            mask = torch.ones_like(color, dtype=torch.float)
            all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6) # [h, w, 6]
            rand_rays = all_rays[(pixels_y, pixels_x)] # [batch_size, 6]
            return torch.cat([rand_rays, color, mask[:, :1]], dim=-1).to(self.device)
        elif mode == 'patch':
            pixels_x = torch.randint(low=0, high=self.w, size=[batch_size // 9])
            pixels_y = torch.randint(low=0, high=self.h, size=[batch_size // 9])
            pixel_idx = pixels_y * self.h + pixels_x
            rand_idx = self.all_neighbor_idx[pixel_idx].reshape(-1)
            pixels_x = rand_idx % self.w
            pixels_y = rand_idx // self.w
            patch_rgb_std = self.all_rgb_std[img_idx].reshape(-1, 1)[rand_idx]
            color = self.all_rgbs[img_idx]  # [h, w, 3]
            color = color[(pixels_y, pixels_x)]  # [batch_size, 3]
            mask = torch.ones_like(color, dtype=torch.float)
            all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6)  # [h, w, 6]
            rand_rays = all_rays[(pixels_y, pixels_x)]  # [batch_size, 6]
            return torch.cat([rand_rays, color, mask[:, :1], patch_rgb_std], dim=-1).to(self.device)


    def near_far_from_sphere(self, rays_o, rays_d):
        # copied from dataset.py
        # a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        # b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        # mid = 0.5 * (-b) / a
        # near = mid - 1.0
        # far = mid + 1.0

        near = torch.zeros(rays_o.shape[0], 1).cuda()
        far = torch.ones(rays_o.shape[0], 1).cuda() * 3
        return near, far

    def gen_rays_at(self, img_idx, resolution_level=1):
        all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6) # [h, w, 6]
        rays_o = all_rays[:, :, :3].to(self.device)
        rays_d = all_rays[:, :, 3:].to(self.device)
        return rays_o, rays_d

    def image_at(self, idx, resolution_level):
        img = cv2.imread(self.image_paths[idx])
        return (cv2.resize(img, (self.w // resolution_level, self.h // resolution_level))).clip(0, 255)

    
    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        # only used in novel view synthesis
        raise NotImplementedError()