# import pydevd_pycharm
# pydevd_pycharm.settrace('166.111.81.13', port=44002, stdoutToServer=True, stderrToServer=True)

# 在vis_mask.py基础上进行改进，针对任意场景，给定segmentation_dir就做visualize
# script:
# python vis_mask_anyscene.py --segmentation_dir xxx

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def create_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap[1:]

def main(instance_dir, vis_dir):
    instanceid_list = os.listdir(instance_dir)
    all_images = np.zeros((3,378,504,3),dtype=np.uint8)
    for instid in instanceid_list:
        segmentation_path = os.path.join(instance_dir, instid)
        frames = os.listdir(segmentation_path)
        for frame in frames:
            mask_img = cv2.imread(os.path.join(segmentation_path, frame), cv2.IMREAD_UNCHANGED)
            frameid = int(frame[:-4])
            all_images[frameid,mask_img>128] = colormap[int(instid)]
    for i in range(3):
        cv2.imwrite(os.path.join(vis_dir, f'{i:04d}.jpg'), all_images[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instancemask_dir', type=str)

    args = parser.parse_args()
    colormap = create_colormap()
    vis_dir = os.path.join(args.instancemask_dir, '..', 'debug/vis_mask')
    os.makedirs(vis_dir, exist_ok=True)
    main(args.instancemask_dir, vis_dir)