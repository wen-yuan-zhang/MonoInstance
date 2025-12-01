#!/bin/bash
# chmod +x full_train_scannet.sh
# CUDA_VISIBLE_DEVICES=1 ./full_train_scannet.sh


SCENE_NAME="scene0050_00"
METHOD_NAME='full_mono0.2_warp0.4'
MASTER_PORT=29500
SCANNET_PATH='/home/zhangwenyuan/data/ScanNet/scans'
DATA_PATH="/home/zhangwenyuan/MonoInstance/MonoInstance/data/scannet/$SCENE_NAME"
EXP_PATH="/home/zhangwenyuan/MonoInstance/MonoInstance/exps/scannet_mlp_$SCENE_NAME/$METHOD_NAME"
MASKCLUSTERING_PATH="/home/zhangwenyuan/MonoInstance/MaskClustering"

set -e  # exit if meets any error


#### transform scannet dataset into monosdf format
#echo 'prepare scannet raw data...'
#cd preprocess
#python process_scannet.py --scene_name $SCENE_NAME --input_path $SCANNET_PATH --output_path $DATA_PATH


#### image deblurring. just for scannet dataset
#echo 'image deblur...'
#cd ../../HI-Diff
#cp -r $DATA_PATH/image datasets/test/RealBlur_J/input
#cp -r $DATA_PATH/image datasets/test/RealBlur_J/target
#python test.py -opt options/test/RealBlur_J_owndata.yml
#cp results/test_HI_Diff_RealBlur_J/visualization/RealBlur_J/* $DATA_PATH
#rm -r datasets/test/RealBlur_J/input
#rm -r datasets/test/RealBlur_J/target


#### estimate depth & normal using metric3d
#echo 'monocular depth and normal estimation...'
#cd ../Metric3D
#python pred_depth_normal_scannet.py --data_dir $DATA_PATH


### Train for first 1/4 epochs；Render depth maps for MaskClustering at the end(1/2 or lower reslution for efficiency)
echo 'stage 1 training...'
cd ../MonoInstance/code
torchrun --master_port $MASTER_PORT training/exp_runner.py --conf confs/scannet_mlp_ours.conf --scan_id $SCENE_NAME --method_name $METHOD_NAME --stage 1


### MaskClustering segment multi-view semantics
echo 'instance segmentation...'
cd ../../MaskClustering-ours
mkdir -p data/scannet/processed/$SCENE_NAME
mkdir -p data/scannet/processed/$SCENE_NAME/color
cp $DATA_PATH/*_rgb.png data/scannet/processed/$SCENE_NAME/color/
mkdir -p data/scannet/processed/$SCENE_NAME/depth
python ../MonoInstance/code/process/construct_monodepth_maskclustering.py --mono_path $DATA_PATH --gt_path $EXP_PATH --output_path $MASKCLUSTERING_PATH/data/scannet/processed/$SCENE_NAME/depth
cp -r $DATA_PATH/intrinsic/ data/scannet/processed/$SCENE_NAME/intrinsic
cp -r $DATA_PATH/pose/ data/scannet/processed/$SCENE_NAME/pose
python third_party/detectron2/projects/CropFormer/demo_cropformer/mask_predict.py --config-file third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml --root data/scannet/processed/ --image_path_pattern color/*.png --dataset scannet --seq_name_list $SCENE_NAME --opts MODEL.WEIGHTS checkpoints/Mask2Former_hornet_3x_576d0b.pth
python main.py --config scannet --debug --seq_name_list $SCENE_NAME
cp -r data/scannet/processed/$SCENE_NAME/output/mask_instances_postprocess $DATA_PATH/instance_masks


### GroundedSAM filter out backgrounds
echo 'background segmentation...'
cd ../Grounded-Segment-Anything
source ~/anaconda3/etc/profile.d/conda.sh
conda activate groundedsam
rm -r assets/*
cp $DATA_PATH/*_rgb.png assets/
rm -r outputs/*
python grounded_sam_demo_ours.py --input_path assets --output_dir outputs/bg_masks


### calculate monocular confidence
echo 'calculate monocular confidence...'
source ~/anaconda3/etc/profile.d/conda.sh
conda activate monoinstance
cd ../MonoInstance/code
torchrun --master_port $MASTER_PORT training/exp_runner.py --conf confs/scannet_mlp_ours.conf --scan_id $SCENE_NAME --method_name $METHOD_NAME --preprocess


### Continue training into the end
echo 'stage 2 training...'
torchrun --master_port $MASTER_PORT training/exp_runner.py --conf confs/scannet_mlp_ours.conf --scan_id $SCENE_NAME --method_name $METHOD_NAME --stage 2 --is_continue --ckpt_name $METHOD_NAME