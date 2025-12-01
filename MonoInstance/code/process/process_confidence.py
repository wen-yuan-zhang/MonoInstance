from tqdm import tqdm
import shutil
import torch
import numpy as np
import os
import cv2
from collections import defaultdict, Counter
from utils import visualization, ours_util
from pytorch3d.ops import knn_points, ball_query
import trimesh
import open3d as o3d
import math


def analyze_monocular_confidence(dataset, uv_down=4): # normalize across all instances
    def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))
    per_instance_max_density = []
    all_frame_densities = []
    shifted_pred_depths = []
    for i in range(len(dataset.depth_images)):
        depth = dataset.depth_images[i].cuda()
        border = 8
        downsample_depth = depth.clone()
        downsample_depth[:border, :] = 0
        downsample_depth[-border:, :] = 0
        downsample_depth[:, :border] = 0
        downsample_depth[:, -border:] = 0
        downsample_depth = downsample_depth[::uv_down,::uv_down]
        gt_depth = dataset.gt_depth_images[i].cuda()    # GT depth means rendering depth here
        scale, shift = ours_util.compute_scale(downsample_depth[None, ...], gt_depth[None, ...],((gt_depth>0)&(downsample_depth>0))[None, ...])
        shifted_pred_depth = depth * scale + shift
        shifted_pred_depths.append(shifted_pred_depth.reshape(-1))
    for instance_label in tqdm(range(len(dataset.instances_refview_valid))):
        candidiate_idx = list(dataset.instances_refview_valid[instance_label].keys())
        per_frame_depth_points = []
        fuse_gt_points = []
        max_density = -1
        for i in candidiate_idx:
            shifted_pred_depth = shifted_pred_depths[i]
            mask_img_i = dataset.mask_images[i][instance_label].reshape(dataset.img_res[0], dataset.img_res[1])
            uv = torch.nonzero(mask_img_i).flip(-1).unsqueeze(0).cuda()
            uv_mask = dataset.mask_images[i][instance_label].squeeze().bool()
            K = dataset.intrinsics_all[0].unsqueeze(0).cuda()
            pose_i = dataset.pose_all[i].unsqueeze(0).cuda()
            depth_points = ours_util.get_depth_point_cloud(uv, K, pose_i, shifted_pred_depth[uv_mask])
            # gt_depth_points = ours_util.get_depth_point_cloud(uv, K, pose_i, gt_depth[uv_mask])
            per_frame_depth_points.append(depth_points)
            # fuse_gt_points.append(gt_depth_points)
        # fuse_gt_points = torch.cat(fuse_gt_points, 0)
        fuse_points = torch.cat(per_frame_depth_points, 0)

        downsamp_points = 20000 if fuse_points.shape[0] > 20000 else fuse_points.shape[0]
        fuse_points = fuse_points[np.random.choice(fuse_points.shape[0], [downsamp_points], replace=False)]
        # remove noise points for calculating bounding box
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fuse_points.detach().cpu().numpy())
        labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=4)) + 1  # -1 for noise
        mask = np.ones(len(labels), dtype=bool)
        count = np.bincount(labels)
        # remove component with less than 20% points
        for i in range(len(count)):
            if count[i] < 0.1 * len(labels):
                mask[labels == i] = False
        remain_index = np.where(mask)[0]
        pcd = pcd.select_by_index(remain_index)
        pcd, index = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)     # TODO: scannet=2.0, replica=0.1
        remain_index = remain_index[index]
        fuse_points_after = np.asarray(pcd.points)
        fuse_points_after = torch.FloatTensor(fuse_points_after).cuda()
        obb = pcd.get_oriented_bounding_box()
        obb_volume = np.prod(obb.extent)
        radius = clamp(obb_volume + 0.01, 0.005, 0.07)  # volume=0.01->r=0.02; volume=0.040->r=0.05

        # all frames' density of an instance
        instance_frame_densities = []      # list: n elements of [n_frame_point]
        for i in range(len(per_frame_depth_points)):
            depth_points = per_frame_depth_points[i]
            density = []
            # split to avoid oom
            for batch in depth_points.split(50000):
                query = ball_query(batch[None],fuse_points_after[None],K=2000, radius=radius)
                dist = query.dists[0]
                idx = query.idx[0]
                batch_density = (idx!=-1).sum(-1)
                density.append(batch_density)
            density = torch.cat(density, 0).cpu().numpy()
            instance_frame_densities.append(density)
            frame_max_density = density.max()
            if frame_max_density > max_density:
                max_density = frame_max_density
        all_frame_densities.append(instance_frame_densities)
        per_instance_max_density.append(max_density*0.8)    # >max_density*80% -> regard as 1

    # write to file
    print('Writting to files...')
    for instance_label in range(len(dataset.instances_refview_valid)):
        out_dir = os.path.join(dataset.instance_dir, 'monocular_convincing', str(instance_label))
        os.makedirs(out_dir,exist_ok=True)
        candidiate_idx = list(dataset.instances_refview_valid[instance_label].keys())
        for i, idx in enumerate(candidiate_idx):
            frame_convincing = all_frame_densities[instance_label][i]
            frame_convincing = frame_convincing / (per_instance_max_density[instance_label]+1e-6)
            frame_convincing = np.clip(frame_convincing, 0., 1.0)
            np.save(os.path.join(out_dir, str(idx)), frame_convincing)

# detect all background instance ids according to groundedsam mask
# use scene_dataset.py instead of scene_dataset_simple.py
def detach_background(dataset):
    groundedsam_mask_path = os.path.join(dataset.instance_dir, '../../../../Grounded-Segment-Anything/outputs/bg_masks/')
    # groundedsam_mask_path = "/home/zhangwenyuan/enhancing-mono/Grounded-Segment-Anything/outputs/bg_masks/"
    print(groundedsam_mask_path)
    sam_mask_img_names = os.listdir(groundedsam_mask_path)
    sam_mask_imgs = {}      # {frame_i: nonzero_pix, frame_j: nonzero_pix, ...}
    for name in sam_mask_img_names:
        frame_id = int(name[:-4])
        img = cv2.imread(os.path.join(groundedsam_mask_path, name), cv2.IMREAD_GRAYSCALE)
        nonzero_pix = np.nonzero((img>128).reshape(-1))[0]
        sam_mask_imgs.update({frame_id: nonzero_pix})
    mask_path = dataset.mask_path
    all_labels = os.listdir(mask_path)
    instances_in_bgmask = [[] for i in all_labels]      # n_instances, [[True,False,False,...],[True,True,False,...],...]
    instances_in_bgmask_prop = [[] for i in all_labels]
    FRAME_IN_BG_THRES=0.8   # pixel percentage > FRAME_IN_BG_THRES -> this frame is bg
    INSTANCE_IS_BG_THRES=0.5    # pixel percentage > INSTANCE_IS_BG_THRES -> this instance is bg
    for frame_id, instance_info in enumerate(dataset.perframe_instances):
        if frame_id in sam_mask_imgs:
            sam_mask_pix = sam_mask_imgs[frame_id]
            for instance_id, pix_info in instance_info.items():
                if instance_id != -1:   # not belong to any instance id
                    instance_pix = pix_info[0]
                    intersect_pix = np.intersect1d(sam_mask_pix, instance_pix, assume_unique=True)
                    if len(intersect_pix) / len(instance_pix) > FRAME_IN_BG_THRES:
                        instances_in_bgmask[instance_id].append(True)
                    else:
                        instances_in_bgmask[instance_id].append(False)
                    instances_in_bgmask_prop[instance_id].append((frame_id, len(intersect_pix) / len(instance_pix)))
    interest_instance_labels = [False for i in all_labels]
    for i, info in enumerate(instances_in_bgmask):
        if len(info) == 0:
            interest_instance_labels[i] = False
            continue
        if sum(info) / len(info) > INSTANCE_IS_BG_THRES:
            interest_instance_labels[i] = False
        else:
            interest_instance_labels[i] = True
    interest_instance_label_dir = os.path.join(dataset.instance_dir, 'interest_instances')
    os.makedirs(interest_instance_label_dir, exist_ok=True)
    np.savetxt(os.path.join(interest_instance_label_dir, 'interest_instance_labels.txt'), interest_instance_labels)
    print([(i,t) for i,t in enumerate(interest_instance_labels)])


if __name__ == '__main__':
    print('main')