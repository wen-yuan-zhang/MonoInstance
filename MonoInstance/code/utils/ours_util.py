import numpy as np
import torch
from utils import rend_util
import trimesh
import os
from colour import Color


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

    x_0[valid] = b_0[valid] / det[valid]

    return x_0, 0

def project_uv_to_another_view(uv, K, T1, T2, depth):
    # uv: [b,N,2], K, T1, T2: [1,4,4], depth: [N,1]

    ray_dirs, cam_loc = rend_util.get_camera_params(uv, T1, K)
    ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(T1.device)[None], K)
    depth_scale = ray_dirs_tmp[0, :, 2:]
    batch_size, num_pixels, _ = ray_dirs.shape
    cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
    ray_dirs = ray_dirs.reshape(-1, 3)
    depth = depth.reshape(-1, 1)
    depth_points = cam_loc + ray_dirs * depth / depth_scale     # [N,3]
    # pix_loc = K * T^-1 * xyz
    xyz_h = torch.cat([depth_points, torch.ones_like(depth_points[...,:1])],-1)
    cam_pos = torch.matmul(xyz_h, T2[0].transpose(0,1).inverse())[...,:3]   # [N,3]
    projections = torch.matmul(cam_pos, K[0,:3,:3].transpose(0,1))
    pixel_locations = projections[..., :2] / projections[..., 2:3]
    front_mask = projections[...,2] > 0
    return pixel_locations, front_mask

def project_points_to_another_view(points, K, T):
    # points: [N,3], K,T: [1,4,4]; just for single src view
    xyz_h = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    cam_pos = torch.matmul(xyz_h, T[0].transpose(0, 1).inverse())[..., :3]  # [N,3]
    projections = torch.matmul(cam_pos, K[0, :3, :3].transpose(0, 1))
    pixel_locations = projections[..., :2] / projections[..., 2:3]
    front_mask = projections[..., 2] > 0
    return pixel_locations, front_mask

# same as "project_points_to_another_view" under single src view
def project_points_to_another_view_neuralwarp(points, K, T):
    # points: [N,3], K,T:[n_src, 4,4]
    pose = torch.inverse(T)[:,:3]
    intr = K[:,:3,:3]
    xyz = (intr.unsqueeze(1) @ pose.unsqueeze(1) @ add_hom(points).unsqueeze(-1))[..., :3, 0]
    in_front = xyz[..., 2] > 0
    grid = xyz[..., :2] / torch.clamp(xyz[..., 2:], 1e-8)
    return grid, in_front

def get_depth_point_cloud(uv, K, T, depth):
    # uv: [b,N,2], K, T: [1,4,4], depth: [N,1]

    ray_dirs, cam_loc = rend_util.get_camera_params(uv, T, K)
    ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(T.device)[None], K)
    depth_scale = ray_dirs_tmp[0, :, 2:]
    batch_size, num_pixels, _ = ray_dirs.shape
    cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
    ray_dirs = ray_dirs.reshape(-1, 3)
    depth = depth.reshape(-1, 1)
    depth_points = cam_loc + ray_dirs * depth / depth_scale  # [N,3]
    return depth_points


def normalize(flow, h, w, clamp=None):
    # either h and w are simple float or N torch.tensor where N batch size
    try:
        h.device

    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)
    elif len(flow.shape) == 5:
        w = w.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        h = h.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    res = torch.empty_like(flow)
    if res.shape[-1] == 3:
        res[..., 2] = 1

    # for grid_sample with align_corners=True
    # https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = 2 * flow[..., 0] / (w - 1) - 1
    res[..., 1] = 2 * flow[..., 1] / (h - 1) - 1

    if clamp:
        return torch.clamp(res, -clamp, clamp)
    else:
        return res


def unnormalize(flow, h, w):
    try:
        h.device
    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)

    res = torch.empty_like(flow)

    if res.shape[-1] == 3:
        res[..., 2] = 1

    # idem: https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = ((flow[..., 0] + 1) / 2) * (w - 1)
    res[..., 1] = ((flow[..., 1] + 1) / 2) * (h - 1)

    return res

def add_hom(pts):
    try:
        dev = pts.device
        ones = torch.ones(pts.shape[:-1], device=dev).unsqueeze(-1)
        return torch.cat((pts, ones), dim=-1)

    except AttributeError:
        ones = np.ones((pts.shape[0], 1))
        return np.concatenate((pts, ones), axis=1)