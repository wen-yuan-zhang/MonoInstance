# python process_scannet.py --scene_name $SCENE_NAME --input_path $SCANNET_PATH --output_path $DATA_PATH
# need to ensure that raw scannet data contain color, intrinsic, pose
import numpy as np
import os
import glob
import PIL
from PIL import Image
from torchvision import transforms
import shutil


H, W = 968, 1296
resize_factor = 0.5

# scannet format to monosdf format
def scannet_to_monosdf(scene, data_dir, out_dir, skip=8):
    trans_totensor = transforms.Compose([transforms.Resize([int(H*resize_factor), int(W*resize_factor)], interpolation=PIL.Image.BILINEAR),])

    out_path = out_dir
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    # folders = ["image", "depth"]
    # for folder in folders:
    #     out_folder = os.path.join(out_path, folder)
    #     os.makedirs(out_folder, exist_ok=True)

    # load color
    color_path = os.path.join(data_dir, scene, 'color')
    color_paths = sorted(glob.glob(os.path.join(color_path, '*.jpg')),
                         key=lambda x: int(os.path.basename(x)[:-4]))
    print(color_paths)

    # # load depth
    # depth_path = os.path.join(data_dir, scene, 'depth')
    # depth_paths = sorted(glob.glob(os.path.join(depth_path, '*.png')),
    #                      key=lambda x: int(os.path.basename(x)[:-4]))
    # print(depth_paths)

    # load intrinsic
    intrinsic_path = os.path.join(data_dir, scene, 'intrinsic', 'intrinsic_color.txt')
    camera_intrinsic = np.loadtxt(intrinsic_path)
    if not os.path.exists(os.path.join(out_path, 'intrinsic')):
        shutil.copytree(os.path.join(data_dir, scene, 'intrinsic'), os.path.join(out_path, 'intrinsic'))    # copy intrinsics for maskclustering
    print(camera_intrinsic)

    # load pose
    pose_path = os.path.join(data_dir, scene, 'pose')
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for _pose_path in pose_paths:
        c2w = np.loadtxt(_pose_path)
        poses.append(c2w)
    poses = np.array(poses)

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print('center', center, 'scale', scale)
    os.makedirs(os.path.join(out_path, 'scene_scale'),exist_ok=True)
    with open(os.path.join(out_path,'scene_scale/scene_scale.txt'),'w') as f:
        f.write(str(scale))

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3] *= scale
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    out_index = 0
    cameras = {}

    camera_intrinsic[:2, :] *= resize_factor
    K = camera_intrinsic

    for idx, (valid, pose, image_path) in enumerate(zip(valid_poses, poses, color_paths)):
        if idx % skip != 0: continue
        if not valid: continue
        print(idx, valid)

        target_image = os.path.join(out_path, "image/%06d_rgb.png" % (out_index))
        print(target_image)
        os.makedirs(os.path.join(out_path, 'image'), exist_ok=True)
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        # # load depth
        # target_image = os.path.join(out_path, "depth/%06d.png" % (out_index))
        # depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        # depth_PIL = Image.fromarray(depth)
        # new_depth = depth_trans_totensor(depth_PIL)
        # new_depth = np.asarray(new_depth)
        # plt.imsave(target_image, new_depth, cmap='viridis')
        # np.save(target_image.replace(".png", ".npy"), new_depth)

        # save pose for maskclustering
        pose = K @ np.linalg.inv(pose)
        os.makedirs(os.path.join(out_path, 'pose'), exist_ok=True)
        shutil.copy(os.path.join(pose_path, str(idx)+'.txt'), os.path.join(out_path, 'pose', str(out_index)+'.txt'))

        # cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d" % (out_index)] = scale_mat
        cameras["world_mat_%d" % (out_index)] = pose

        out_index += 1

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)

# copy images into hidiff dataset for running hidiff
def image_deblur(data_path, hidiff_path):
    shutil.copytree(os.path.join(data_path, 'image'), os.path.join(hidiff_path, 'dataset/test/RealBlur_J/input'))
    shutil.copytree(os.path.join(data_path, 'image'), os.path.join(hidiff_path, 'dataset/test/RealBlur_J/target'))
    os.system('python test.py -opt options/test/RealBlur_J_owndata.yml')
    os.system('cp {}/results/test_HI_Diff_RealBlur_J/visualization/RealBlur_J/* {}/'.format(hidiff_path, data_path))

# def monocular_estimation(data_path, metric3d_path):



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    opt = parser.parse_args()
    scannet_to_monosdf(opt.scene_name, opt.input_path, opt.output_path)
