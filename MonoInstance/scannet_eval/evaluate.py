
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import glob
import os
import pyrender
import os
import argparse
from tqdm import tqdm
from pathlib import Path

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)
    # debug: output verts_pred error map
    # trimesh.Trimesh(vertices=verts_pred, vertex_colors=np.asarray(np.ones_like(verts_pred)*(255 * dist2[:, None]/dist2.max()),dtype=np.uint8)).export('errormap.ply')

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics

# hard-coded image size
H, W = 968, 1296

# load pose
def load_poses(scan):
    pose_path = os.path.join(f'../data/scannet/{scan}', 'pose')
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    # for pose_path in pose_paths[::8]:
    for pose_path in pose_paths:
        c2w = np.loadtxt(pose_path)
        if np.isfinite(c2w).any():
            poses.append(c2w)
    poses = np.array(poses)
    
    return poses


class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        

def refuse(mesh, poses, K):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="../exps/")
    parser.add_argument("--exp_name", type=str, default="scannet_mlp")
    parser.add_argument("--out_dir", type=str, default="evaluation/scannet_mlp")
    parser.add_argument("--scan", type=str, default="scene0050_00")
    parser.add_argument(
        "--ply_file",
        type=str,
        default="/home/zhangwenyuan/MonoInstance/MonoInstance/exps/scannet_mlp_scene0050_00/full_mono0.2_warp0.4/plots/surface_189.ply"
    )
    args = parser.parse_args()

    root_dir = "../exps/"
    exp_name = "scannet_mlp"
    out_dir = "evaluation/scannet_mlp"
    Path(out_dir).mkdir(parents=True, exist_ok=True)


    # scenes = ["scene0050_00", "scene0084_00", "scene0580_00", "scene0616_00"]
    scenes = [args.scan]
    all_results = []
    for idx, scan in enumerate(scenes):
        ply_file = args.ply_file

        mesh = trimesh.load(ply_file)
        print(mesh)

        # transform to world coordinate
        cam_file = f"../data/scannet/{scan}/cameras.npz"
        scale_mat = np.load(cam_file)['scale_mat_0']
        mesh.vertices = (scale_mat[:3, :3] @ mesh.vertices.T + scale_mat[:3, 3:]).T

        # load pose and intrinsic for render depth
        poses = load_poses(scan)

        intrinsic_path = os.path.join(f'../data/scannet/{scan}/intrinsic/intrinsic_color.txt')
        K = np.loadtxt(intrinsic_path)[:3, :3]
        mesh.faces = mesh.faces[:,[0,2,1]]  # TODO: the face orientation of mcubes is opposite from that of skimage
        mesh = refuse(mesh, poses, K)

        # save mesh
        out_mesh_path = os.path.join(out_dir, f"{exp_name}_{scan}.ply")
        o3d.io.write_triangle_mesh(out_mesh_path, mesh)
        mesh = trimesh.load(out_mesh_path)


        #gt_mesh = os.path.join("../data/scannet/GTmesh", f"{scan}_vh_clean_2.ply")
        gt_mesh = os.path.join("../data/scannet/GTmesh_lowres/", f"{scan[5:]}.obj")

        gt_mesh = trimesh.load(gt_mesh)

        metrics = evaluate(mesh, gt_mesh)
        print(metrics)
        all_results.append(metrics)

    # print all results
    for scan, metric in zip(scenes, all_results):
        values = [scan] + [str(metric[k]) for k in metric.keys()]
        out = ",".join(values)
        print(out)
