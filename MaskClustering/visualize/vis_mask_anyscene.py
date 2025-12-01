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

def main(segmentation_dir, vis_dir):
    frame_list = os.listdir(segmentation_dir)
    for filename in frame_list:
        segmentation_path = os.path.join(segmentation_dir, filename)
        segmentation_image = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        color_segmentation = np.zeros((segmentation_image.shape[0], segmentation_image.shape[1], 3), dtype=np.uint8)
        mask_ids = np.unique(segmentation_image)
        mask_ids.sort()

        text_list, text_center_list = [], []
        for mask_id in mask_ids:
            if mask_id == 0:
                continue
            color_segmentation[segmentation_image == mask_id] = colormap[mask_id]
            mask_pos = np.where(segmentation_image == mask_id)
            mask_center = (int(np.mean(mask_pos[1])), int(np.mean(mask_pos[0])))
            text_list.append(str(mask_id))
            text_center_list.append(mask_center)

        # for text, text_center in zip(text_list, text_center_list):
        #     cv2.putText(color_segmentation, text, text_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imwrite(os.path.join(vis_dir, filename), color_segmentation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_dir', type=str)

    args = parser.parse_args()
    colormap = create_colormap()
    vis_dir = os.path.join(args.segment_dir, '..', 'vis_mask')
    os.makedirs(vis_dir, exist_ok=True)
    main(args.segment_dir, vis_dir)