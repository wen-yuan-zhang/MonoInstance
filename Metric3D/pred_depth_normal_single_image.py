# import pydevd_pycharm
# pydevd_pycharm.settrace('166.111.81.13', port=33778, stdoutToServer=True, stderrToServer=True)
# CUDA_VISIBLE_DEVICES=1 python pred_depth_normal_dtu.py

dependencies = ['torch', 'torchvision']

import os
import torch
from tqdm import tqdm
import numpy as np
import shutil
import argparse
# try:
#     from mmcv.utils import Config, DictAction
# except:
from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
metric3d_dir = os.path.dirname(__file__)

MODEL_TYPE = {
    'ConvNeXt-Tiny': {
        # TODO
    },
    'ConvNeXt-Large': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
    },
    'ViT-Small': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
    },
    'ViT-Large': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
        # 'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
        'ckpt_file': 'checkpoints/metric_depth_vit_large_800k.pth'
    },
    'ViT-giant2': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
    },
}



def metric3d_convnext_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model

def metric3d_vit_small(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model

def metric3d_vit_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            # torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            torch.load(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model

def metric3d_vit_giant2(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'],
            strict=False,
        )
    return model

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose



if __name__ == '__main__':
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/zhangwenyuan/xiangmu/zhongdianwanwei/data/DJI_20240704_process/images/', type=str, help='input data path')
    parser.add_argument('--output_dir', default='/home/zhangwenyuan/xiangmu/zhongdianwanwei/data/DJI_20240704_process/priors/', type=str, help='input data path')
    opt = parser.parse_args()

    #### prepare data
    filenames = os.listdir(opt.data_dir)
    filenames = [file for file in filenames if file.endswith('.JPG')]
    filenames = sorted(filenames)
    rgbs = []

    K = np.array([[744.7004643058907,0.0,507.0],
                [0.0,746.4471635251881,380.0],
                [0.0, 0.0,1.0]],dtype=np.float32)
    img_rescale = 1.0
    raw_intrinsic = [K[0][0]*img_rescale, K[1][1]*img_rescale, K[0][2]*img_rescale, K[1][2]*img_rescale]

    print('reading images...')
    for rgb_filename in filenames:
        # intrinsic = [1170.187988/2, 1170.187988/2, 647.75/2, 483.75/2]  # scan0050_00
        # intrinsic = [1169.621094/2, 1169.621094/2, 646.295044/2, 489.927032/2]  # scan0084_00
        # intrinsic = [1170.187988/2, 1170.187988/2, 647.75/2, 483.75/2]  # scan0580_00
        gt_depth_scale = 256.0
        rgb_origin = cv2.imread(os.path.join(opt.data_dir, rgb_filename))[:, :, ::-1]

        #### ajust input size to fit pretrained model
        # keep ratio resize
        input_size = (616, 1064) # for vit model
        # input_size = (544, 1216) # for convnext model
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # remember to scale intrinsic, hold depth
        intrinsic = [raw_intrinsic[0] * scale, raw_intrinsic[1] * scale, raw_intrinsic[2] * scale, raw_intrinsic[3] * scale]
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()
        rgbs.append(rgb)

    ###################### canonical camera space ######################
    # inference
    # model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model = metric3d_vit_large(pretrain=True)
    model.cuda().eval()

    for i in tqdm(range(len(rgbs))):
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': rgbs[i]})

            # un pad
            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
            # upsample to original size
            pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
            # un pad
            pred_normal = output_dict['prediction_normal'].squeeze().permute(1,2,0)    # [h,w,4]
            pred_normal = pred_normal[pad_info[0]: pred_normal.shape[0] - pad_info[1], pad_info[2]: pred_normal.shape[1] - pad_info[3]]
            # upsample to original size
            pred_normal = torch.nn.functional.interpolate(pred_normal.permute(2,0,1)[None,...], rgb_origin.shape[:2],mode='bilinear').squeeze()
            pred_normal = pred_normal.squeeze()[:3, ...]
            pred_normal = (pred_normal + 1) / 2     # [3,h,w]
            ###################### canonical camera space ######################
            #### de-canonical transform
            canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
            # pred_depth = torch.clamp(pred_depth, 0, 6.)

            # output
            img_idx = filenames[i][:-4]
            os.makedirs(opt.output_dir, exist_ok=True)
            np.save(os.path.join(opt.output_dir, img_idx+'_depth.npy'), pred_depth.float().cpu().numpy())
            np.save(os.path.join(opt.output_dir, img_idx+'_normal.npy'), pred_normal.float().cpu().numpy())
            cv2.imwrite(os.path.join(opt.output_dir, img_idx+'_depth.png'), (pred_depth*(255/pred_depth.max())).int().cpu().numpy())
            cv2.imwrite(os.path.join(opt.output_dir, img_idx+'_normal.png'), (pred_normal.permute(1,2,0)*255).int().cpu().numpy()[...,[2,1,0]])
            # cv2.imwrite(os.path.join(opt.data_dir, img_idx+'_rgb_test.png'), rgb_origin[...,[2,1,0]])     # just for test
            # shutil.copy(os.path.join('data/own_data', filenames[i]), 'outputs/'+img_idx+'_rgb.png')



    # #### you can now do anything with the metric depth
    # # such as evaluate predicted depth
    # if depth_file is not None:
    #   gt_depth = cv2.imread(depth_file, -1)
    #   gt_depth = gt_depth / gt_depth_scale
    #   gt_depth = torch.from_numpy(gt_depth).float().cuda()
    #   assert gt_depth.shape == pred_depth.shape
    #
    #   mask = (gt_depth > 1e-8)
    #   abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
    #   print('abs_rel_err:', abs_rel_err.item())