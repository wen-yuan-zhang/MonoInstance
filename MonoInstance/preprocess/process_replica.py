
import numpy as np
import cv2
import os
import shutil
import trimesh

out_path_prefix = 'D:/code/MonoInstance/MonoInstance/data/replica'
data_root = 'D:/data/Replica'
scenes = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']

# scenes = ['room0']

for scene in scenes:
    out_path = os.path.join(out_path_prefix, scene)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    # folders = ["train", "val", "test", "val2"]
    folders = ["image", "mask"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    pose_file = os.path.join(data_root, scene, 'traj.txt')
    shutil.copy(pose_file, os.path.join(out_path, 'traj.txt'))
    images_dir = os.path.join(data_root, scene, "results")
    mesh_file = os.path.join(data_root, "%s_mesh.ply" % (scene))

    poses = np.loadtxt(pose_file)
    poses = poses.reshape(-1, 4, 4)

    mesh = trimesh.load(mesh_file)

    min_vertices = mesh.vertices.min(axis=0)
    max_vertices = mesh.vertices.max(axis=0)

    # import pdb; pdb.set_trace()

    center = (min_vertices + max_vertices) / 2.

    scale = 2. / (np.max(max_vertices - min_vertices) * 1.1)
    print('center', center, 'scale', scale)
    os.makedirs(os.path.join(out_path, 'scene_scale'),exist_ok=True)
    with open(os.path.join(out_path,'scene_scale/scene_scale.txt'),'w') as f:
        f.write(str(scale))
    # poses[:, :3, 3] -= center
    # poses[:, :3, 3] *= scale

    # we should normalized to unit cube

    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3] *= scale
    scale_mat = np.linalg.inv(scale_mat)

    # load image
    c2w = poses
    num_image = c2w.shape[0]

    train_frames = []
    test_frames = []

    # copy image
    out_index = 0
    cameras = {}
    K = np.eye(4)
    K[0, 0] = 600.0
    K[1, 1] = 600.0
    K[0, 2] = 599.5
    K[1, 2] = 339.5
    # save intrinsic for maskclustering
    intrinsic_path = os.path.join(out_path, 'intrinsic')
    os.makedirs(intrinsic_path, exist_ok=True)
    np.savetxt(os.path.join(intrinsic_path, 'intrinsic_depth.txt'), K)

    for i in range(num_image):
        if i % 8 != 0:
            continue
        # if i >= 100:
        #    continue

        # copy image file 
        current_frame = os.path.join(images_dir, 'frame%06d.jpg' % (i))
        target_image = os.path.join(out_path, "%06d_rgb.png" % (out_index))
        # print(target_image)
        shutil.copy(current_frame, target_image)
        # os.system("cp %s %s" % (current_frame, target_image))

        # # write mask
        # mask = (np.ones((680, 1200, 3)) * 255.).astype(np.uint8)
        #
        # target_image = os.path.join(out_path, "mask/%03d.png" % (out_index))
        # cv2.imwrite(target_image, mask)

        # save pose
        pose = c2w[i].copy()
        pose = K @ np.linalg.inv(pose)
        # save pose for maskclustering
        os.makedirs(os.path.join(out_path, 'pose'), exist_ok=True)
        np.savetxt(os.path.join(out_path, 'pose', str(out_index)+'.txt'), c2w[i])

        # cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d" % (out_index)] = scale_mat
        cameras["world_mat_%d" % (out_index)] = pose

        out_index += 1

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)