# running script: (default master_port is 29500
# CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29501 training/exp_runner.py --conf confs/scannet_mlp_ours.conf --scan_id 1 --method_name vanilla
# if continue:
# CUDA_VISIBLE_DEVICES=7 torchrun --master_port 29501 training/exp_runner.py --conf confs/scannet_mlp_ours.conf --scan_id 1 --method_name test --is_continue --ckpt_name 0.5res+metric3d_depthalign --ckpt_epoch 200
import sys
sys.path.append('../code')
import argparse
import torch

import os
from training.monosdf_train import MonoSDFTrainRunner
import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    #parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--scan_id', type=str, default='scene0050_00', help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--ckpt_name', default='', type=str,
                        help='The checkpoint experiment name to be used in case of continuing from a previous run.')
    parser.add_argument('--ckpt_epoch', default='latest', type=str,
                        help='The epoch of the run to be used in case of continuing from a previous run.')  # can be regarded as method name
    parser.add_argument('--method_name', type=str, default='0.5res+metric3d', help='exps_folder/expname/methodname.')
    parser.add_argument('--preprocess', action='store_true', help='--preprocess does not start training, just for preprocessing.')
    parser.add_argument('--stage', type=int,
                        help='1 for training 1/4 epochs with full pixels; 2 for training other 3/4 epochs with mono confidence and instance sampling')

    opt = parser.parse_args()

    '''
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    '''
    local_rank = int(os.environ["LOCAL_RANK"])
    gpu = local_rank

    # set distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        # print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
    torch.distributed.barrier()


    trainrunner = MonoSDFTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    checkpoint_name=opt.ckpt_name,
                                    checkpoint_epoch=opt.ckpt_epoch,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis,
                                    methodname=opt.method_name,
                                    preprocess=opt.preprocess,
                                    stage=opt.stage,
                                    )

    trainrunner.run()
