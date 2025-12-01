import torch
from utils.config import get_dataset, get_args
from utils.post_process import post_process, export_object_list, export_object_dict
from graph.construction import mask_graph_construction, construct_scene_points
from graph.iterative_clustering import iterative_clustering
from tqdm import tqdm
import os

# CUDA_VISIBLE_DEVICES=1 python main.py --config demo --debug --seq_name_list scene0050_00
# CUDA_VISIBLE_DEVICES=0 python main.py --config scannet --debug --seq_name_list scene0050_00_ours

def main(args):
    dataset = get_dataset(args)
    # scene_points = dataset.get_scene_points()
    frame_list = dataset.get_frame_list(1)  # args.step
    # if os.path.exists(os.path.join(dataset.object_dict_dir, args.config, f'object_dict.npy')):
    #     return

    with torch.no_grad():
        scene_points = construct_scene_points(dataset, frame_list)
        nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = mask_graph_construction(args, scene_points, frame_list, dataset)

        object_list = iterative_clustering(nodes, observer_num_thresholds, args.view_consensus_threshold, args.debug)
        # export once
        export_object_list(dataset, object_list=object_list, vis_dir=os.path.join(dataset.segmentation_dir, '..', 'mask_instances'))

        object_dict = post_process(dataset, object_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args)
        # export twice
        export_object_dict(dataset, object_dict=object_dict, vis_dir=os.path.join(dataset.segmentation_dir, '..', 'mask_instances_postprocess'))


if __name__ == '__main__':
    args = get_args()
    seq_name_list = args.seq_name_list.split('+')

    for seq_name in tqdm(seq_name_list):
        args.seq_name = seq_name
        main(args)