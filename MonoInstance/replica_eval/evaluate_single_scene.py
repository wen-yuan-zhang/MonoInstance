# CUDA_VISIBLE_DEVICES=4 python evaluate_single_scene.py --input_mesh ../exps/replica_mlp_room1/full_mono0.2_warp0.4/plots/surface_360.ply --scan room1 --output_dir ../exps/replica_mlp_room1/full_mono0.2_warp0.4/plots/evaluation

import os
from pathlib import Path
import subprocess
import argparse


scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--scan', type=str, default='1', help='scan id of the input mesh')
    parser.add_argument('--output_dir', type=str, default='outputs', help='path to the output folder')
    args = parser.parse_args()

    

    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scan = args.scan

    ply_file = args.input_mesh

    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")

    # cumesh
    cull_mesh_out = os.path.join(out_dir, f"cull_{scan}.ply")
    cmd = f"python cull_mesh.py --input_mesh {ply_file} --input_scalemat ../data/replica/{scan}/cameras.npz --traj ../data/replica/{scan}/traj.txt --output_mesh {cull_mesh_out}"
    print(cmd)
    os.system(cmd)

    cmd = f"python eval_recon.py --rec_mesh {cull_mesh_out} --gt_mesh ../data/replica/cull_GTmesh/{scan}.ply"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",")
    print(output)