# scale monodepth, then put them into maskclustering
import numpy as np
import cv2
import os
import glob
from PIL import Image
import argparse
from tqdm import tqdm
import torch


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def compute_scale(prediction, target, mask):
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    x_0 = torch.zeros_like(b_0)
    det = a_00
    valid = det.nonzero()

    # omit shift
    x_0[valid] = b_0[valid] / det[valid]

    return x_0, 0

parser = argparse.ArgumentParser()
parser.add_argument('--mono_path', type=str)
parser.add_argument('--gt_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--dataset', type=str)  # scannet or replica
# parser.add_argument('--scene_scale', type=float)
opt = parser.parse_args()


# load depth
depth_paths = sorted(glob.glob(os.path.join(opt.mono_path, '*_depth.npy')))
# print(depth_paths)
scene_scale = np.loadtxt(os.path.join(opt.mono_path, 'scene_scale/scene_scale.txt'), dtype=np.float32).item()

for depth_path in tqdm(depth_paths):
    depth_path = os.path.basename(depth_path)
    mono_depth = np.load(os.path.join(opt.mono_path, depth_path))
    mono_depth = torch.tensor(mono_depth, dtype=torch.float32).cuda()
    border = 9
    mono_depth[:border, :] = 0
    mono_depth[-border:, :] = 0
    mono_depth[:, :border] = 0
    mono_depth[:, -border:] = 0
    # render depth. NOTE: downscale
    render_depth = np.load(os.path.join(opt.gt_path, 'render_depth', depth_path[:6] + '.npy'))
    render_depth = torch.tensor(render_depth, dtype=torch.float32).cuda() / scene_scale     # return to GT scale. NOTE: maskclustering is GT scale; mono confidence is [-1,1]
    downsampled_mono_depth = mono_depth[::4, ::4]
    # scale, shift = compute_scale_and_shift(downsampled_mono_depth[None, ...], render_depth[None, ...], ((render_depth>0)&(downsampled_mono_depth>0))[None, ...])
    scale, shift = compute_scale(downsampled_mono_depth[None, ...], render_depth[None, ...], ((render_depth>0) & (downsampled_mono_depth>0))[None, ...])
    print(scale, shift)

    shifted_pred_mono_depth = mono_depth * scale + shift
    shifted_pred_mono_depth = shifted_pred_mono_depth.cpu().numpy()
    # # save for monocular confidence calculating
    # os.makedirs(os.path.join(opt.input_path, 'mono_depth_aligned'),exist_ok=True)
    # np.save(os.path.join(opt.input_path, 'mono_depth_aligned',depth_path), shifted_pred_mono_depth*scene_scale)

    shifted_pred_mono_depth = shifted_pred_mono_depth
    shifted_pred_mono_depth = np.array(shifted_pred_mono_depth*1000, dtype=np.uint16)
    if opt.dataset == 'scannet':
        shifted_pred_mono_depth = cv2.resize(shifted_pred_mono_depth, (640, 480), cv2.INTER_NEAREST)
    elif opt.dataset == 'replica':
        shifted_pred_mono_depth = shifted_pred_mono_depth
    else:
        raise NotImplementedError
    image = Image.fromarray(shifted_pred_mono_depth, 'I;16')
    image.save(os.path.join(opt.output_path, depth_path.replace('npy','png')))
