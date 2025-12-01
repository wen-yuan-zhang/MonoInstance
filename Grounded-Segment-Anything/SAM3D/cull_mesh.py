# import pydevd_pycharm
# pydevd_pycharm.settrace('166.111.81.124', port=32797, stdoutToServer=True, stderrToServer=True)

import numpy as np
import torch
import trimesh
from pyhocon import ConfigFactory
from tqdm import tqdm
import os
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterizer
from pytorch3d.renderer.cameras import PerspectiveCameras
import json


H = 680
W = 1200



translation_dict = {
    # BlendSwap
    "breakfast_room": [0.0, -1.42, 0.0],
    "kitchen": [0.3, -3.42, -0.12],
    "green_room": [-1, -0.38, 0.3],
    "complete_kitchen": [1.5, -2.25, 0.0],
    "grey_white_room": [-0.12, -1.94, -0.69],
    "morning_apartment": [-0.22, -0.43, 0.0],
    "staircase": [0.0, -2.42, 0.0],
    "whiteroom": [0.3, -3.42, -0.12],
    # Replica
    'office0': [-0.1944, 0.6488, -0.3271],
    'office1': [-0.585, -0.4703, -0.3507],
    'office2': [0.1909, -1.2262, -0.1574],
    'office3': [0.7893, 1.3371, -0.3305],
    'office4': [-2.0684, -0.9268, -0.1993],
    # 'room0': [-3.00, -1.1631, 0.1235],
    'room0': [-3.00291, -1.16314, 0.12348],     # monosdf config
    'room1': [2.0795, 0.1747, 0.0314],
    'room2': [-2.5681, 0.7727, 1.1110],
}

scale_dict = {
    # BlendSwap
    "breakfast_room": 0.4,
    "kitchen": 0.25,
    "green_room": 0.25,
    "complete_kitchen": 0.20,
    "grey_white_room": 0.25,
    "morning_apartment": 0.5,
    "staircase": 0.25,
    "whiteroom": 0.25,
    # Replica
    'office0': 0.4,
    'office1': 0.41,
    'office2': 0.24,
    'office3': 0.21,
    'office4': 0.30,
    # 'room0': 0.25,
    'room0': 0.234163,      # monosdf config
    'room1': 0.30,
    'room2': 0.29,
}

scene_bounds_dict = {
    # BlendSwap
    "whiteroom": np.array([[-2.46, -0.1, 0.36],
                             [3.06, 3.3, 8.2]]),
    "kitchen": np.array([[-3.12, -0.1, -3.18],
                             [3.75, 3.3, 5.45]]),
    "breakfast_room": np.array([[-2.23, -0.5, -1.7],
                             [1.85, 2.77, 3.0]]),
    "staircase":np.array([[-4.14, -0.1, -5.25],
                             [2.52, 3.43, 1.08]]),
    "complete_kitchen":np.array([[-5.55, 0.0, -6.45],
                             [3.65, 3.1, 3.5]]),
    "green_room":np.array([[-2.5, -0.1, 0.4],
                             [5.4, 2.8, 5.0]]),
    "grey_white_room":np.array([[-0.55, -0.1, -3.75],
                             [5.3, 3.0, 0.65]]),
    "morning_apartment":np.array([[-1.38, -0.1, -2.2],
                             [2.1, 2.1, 1.75]]),
    "thin_geometry":np.array([[-2.15, 0.0, 0.0],
                             [0.77, 0.75, 3.53]]),
    # Replica
    'office0': np.array([[-2.0056, -3.1537, -1.1689],
                             [2.3944, 1.8561, 1.8230]]),
    'office1': np.array([[-1.8204, -1.5824, -1.0477],
                             [2.9904, 2.5231, 1.7491]]),
    'office2': np.array([[-3.4272, -2.8455, -1.2265],
                             [3.0453, 5.2980, 1.5414]]),
    'office3': np.array([[-5.1116, -5.9395, -1.2207],
                             [3.5329, 3.2652, 1.8816]]),
    'office4': np.array([[-1.2047, -2.3258, -1.2093],
                             [5.3415, 4.1794, 1.6078]]),
    'room0': np.array([[-1.1494, -1.4560, -1.7974],
                             [7.1052, 3.7323, 1.5004]]),
    'room1': np.array([[-5.4027, -3.0385, -1.4080],
                             [1.2436, 2.6891, 1.3452]]),
    'room2': np.array([[-0.8171, -3.2454, -2.9081],
                             [5.9533, 1.7000, 0.6861]]),
}



def clean_invisible_vertices(mesh, train_dataset):

    poses = train_dataset.poses
    n_imgs = train_dataset.__len__()
    pc = mesh.vertices
    faces = mesh.faces
    xyz = torch.Tensor(pc)
    xyz = xyz.reshape(1, -1, 3)
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)

    # delete mesh vertices that are not inside any camera's viewing frustum
    whole_mask = np.ones(pc.shape[0]).astype(np.bool)
    for i in tqdm(range(0, n_imgs, 1), desc='clean_vertices'):
        intrinsics = train_dataset.intrinsics
        pose = poses[i]
        # *根据blender数据改写投影逻辑
        camera_pos = torch.einsum('abj,ij->abi', xyz_h, pose.inverse())
        projections = torch.einsum('ij, abj->abi', intrinsics, camera_pos[..., :3])  # [W, H, 3]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8) - 0.5
        pixel_locations = pixel_locations[:, :, [1, 0]]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        uv = pixel_locations.reshape(-1, 2)
        z = pixel_locations[..., -1:] + 1e-5
        z = z.reshape(-1)
        edge = 0
        mask = (0 <= z) & (uv[:, 0] < H - edge) & (uv[:, 0] > edge) & (uv[:, 1] < W-edge) & (uv[:, 1] > edge)
        whole_mask &= ~mask.cpu().numpy()

    pc = mesh.vertices
    faces = mesh.faces
    face_mask = whole_mask[mesh.faces].all(axis=1)
    mesh.update_faces(~face_mask)

    return mesh

# correction from pytorch3d (v0.5.0)
def corrected_cameras_from_opencv_projection( R, tvec, camera_matrix, image_size):
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Get the PyTorch3D focal length and principal point.
    s = (image_size_wh).min(dim=1).values

    focal_pytorch3d = focal_length / (0.5 * s)
    p0_pytorch3d = -(principal_point - image_size_wh / 2) * 2 / s

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    # R_pytorch3d = R.clone()
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1
    # T_pytorch3d[:, 0] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
    )


def clean_triangle_faces(mesh, train_dataset):
    # returns a mask of triangles that reprojects on at least nb_visible images
    num_view = train_dataset.__len__()
    K = train_dataset.intrinsics[:3, :3].unsqueeze(0).repeat([num_view, 1, 1])
    # 这里跟neuralwarp的代码保持一致就好了，我也不知道为啥t需要做一次乘法得到……
    R = train_dataset.poses[:, :3, :3].transpose(2, 1)
    t = - train_dataset.poses[:, :3, :3].transpose(2, 1) @ train_dataset.poses[:, :3, 3:]
    sizes = torch.Tensor([[train_dataset.w, train_dataset.h]]).repeat([num_view, 1])
    cams = [K, R, t, sizes]
    num_faces = len(mesh.faces)
    nb_visible = 1
    count = torch.zeros(num_faces, device="cuda")
    K, R, t, sizes = cams[:4]

    n = len(K)
    with torch.no_grad():
        for i in tqdm(range(n), desc="clean_faces"):
            intr = torch.zeros(1, 4, 4).cuda()  #
            intr[:, :3, :3] = K[i:i + 1]
            intr[:, 3, 3] = 1
            vertices = torch.from_numpy(mesh.vertices).cuda().float()  #
            faces = torch.from_numpy(mesh.faces).cuda().long()  #
            meshes = Meshes(verts=[vertices],
                            faces=[faces])

            cam = corrected_cameras_from_opencv_projection(camera_matrix=intr, R=R[i:i + 1].cuda(),  #
                                                           tvec=t[i:i + 1].squeeze(2).cuda(),  #
                                                           image_size=sizes[i:i + 1, [1, 0]].cuda())  #
            cam = cam.cuda()  #
            raster_settings = rasterizer.RasterizationSettings(image_size=tuple(sizes[i, [1, 0]].long().tolist()),
                                                               faces_per_pixel=1)
            meshRasterizer = rasterizer.MeshRasterizer(cam, raster_settings)

            with torch.no_grad():
                ret = meshRasterizer(meshes)
                pix_to_face = ret.pix_to_face
                mask = train_dataset.masks[i]
                seg_face = pix_to_face.squeeze()[mask]
                # seg_face = pix_to_face.squeeze()

            visible_faces = seg_face.view(-1).unique()
            count[visible_faces[visible_faces > -1]] += 1

    pred_visible_mask = (count >= nb_visible).cpu()

    mesh.update_faces(pred_visible_mask)
    return mesh

def cull_by_bounds(points, scene_bounds):
    eps = 0.02
    inside_mask = np.all(points >= (scene_bounds[0] - eps), axis=1) & np.all(points <= (scene_bounds[1] + eps), axis=1)
    return inside_mask



def crop_mesh(scene, mesh, subdivide=True, max_edge=0.015):
    vertices = mesh.vertices
    triangles = mesh.faces

    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, max_edge=max_edge, max_iter=10)

    # Cull with the bounding box first
    inside_mask = None
    scene_bounds = scene_bounds_dict[scene]
    if scene_bounds is not None:
        inside_mask = cull_by_bounds(vertices, scene_bounds)

    inside_mask = inside_mask[triangles[:, 0]] | inside_mask[triangles[:, 1]] | inside_mask[triangles[:, 2]]
    triangles = triangles[inside_mask, :]
    print("Processed culling by bound")
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # we don't need subdivided mesh to render depth
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def transform_mesh(scene, mesh):
    pc = mesh.vertices
    faces = mesh.faces
    pc = (pc / scale_dict[scene]) - np.array([translation_dict[scene]])
    mesh = trimesh.Trimesh(pc, faces, process=False)
    return mesh


def detransform_mesh(scene, mesh):
    pc = mesh.vertices
    faces = mesh.faces
    pc = (pc + np.array([translation_dict[scene]])) * scale_dict[scene]
    mesh = trimesh.Trimesh(pc, faces, process=False)
    return mesh

def mono_to_ours_normed(mesh):
    # 我们的测评代码和数据集是针对我们的replica设计的，跟mono有一些偏差，需要先把mono的mesh转成我们的
    # 输入是mono归一化到[-1, 1]的mesh，返回的是我们的归一化到[-1, 1]的mesh，跟我们测评NeRFPrior时输入的mesh一致（坐标轴翻转过）
    pc = mesh.vertices
    faces = mesh.faces
    # monosdf to gt scale
    pc = (pc / 0.234163) - np.array([[-3.00291, -1.16314, 0.12348]])
    pc = (pc + np.array([[-3.00, -1.1631, 0.1235]])) * 0.25
    pc = (pc * np.array([[1, -1, 1]]))[:, [0, 2, 1]]
    mesh = trimesh.Trimesh(pc, faces, process=False)
    return mesh


if __name__ == '__main__':
    from blender_swap import BlendSwapDataset

    out_dir_pat = 'meshes/%s'
    scene = 'room0'

    conf = """type=Blender
              data_dir=data/Replica
              scene=room0"""
    conf = ConfigFactory.parse_string(conf)
    train_dataset = BlendSwapDataset(conf)

    exp_name = 'room0'
    mesh_name = 'monosdf'
    dir_pth = out_dir_pat % (exp_name)
    mesh_pth = os.path.join(dir_pth, mesh_name+'.ply')
    print(dir_pth, mesh_pth)
    mesh = trimesh.load_mesh(mesh_pth)

    # yz轴翻转，然后y轴反向：这个是用nerfprior的blender格式数据跑出来的结果向monosdf/replica原始的opencv格式转换需要的
    mesh = mono_to_ours_normed(mesh)
    mesh.export(os.path.join(dir_pth, '%s_test.ply' % mesh_name))
    mesh.vertices = mesh.vertices[:, [0,2,1]] * np.array([[1, -1, 1]])
    mesh = transform_mesh(scene, mesh)
    mesh = crop_mesh(scene, mesh)
    mesh = detransform_mesh(scene, mesh)
    mesh.vertices = (mesh.vertices * np.array([[1, -1, 1]]))[:, [0, 2, 1]]
    mesh.export(os.path.join(dir_pth, '%s_cropped.ply' % mesh_name))
    mesh = clean_invisible_vertices(mesh, train_dataset)
    mesh = clean_triangle_faces(mesh, train_dataset)
    # mesh.fill_holes()
    mesh.export(os.path.join(dir_pth, '%s_cropped_culled.ply' % mesh_name))
    # mesh.vertices = mesh.vertices[:, [0, 2, 1]] * np.array([[1, -1, 1]])
    # mesh = transform_mesh(scene, mesh)
    # mesh.export(os.path.join(dir_pth, '%s_cropped_culled_transformed.ply' % mesh_name))
