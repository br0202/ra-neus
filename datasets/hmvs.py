import os
import math
import numpy as np
from PIL import Image
from termcolor import colored

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.ray_utils import get_ray_directions
from utils.misc import get_rank

import os
import copy
import logging
import numpy as np
import argparse
import R3DParser
import pyransac3d as pyrsc


def bin2camera(work_space, bin_file):
    """convert r3d bin data to cmaera ex/intrinsics"""
    cam_intrinsics, cam_rotations, cam_centers, resolutions, fns = R3DParser.LoadR3DBinDataset(
        work_space, bin_file)
    cam_intrinsics = cam_intrinsics.reshape(-1, 3, 3)
    resolutions = resolutions.reshape(-1, 2)
    cam_rotations = cam_rotations.reshape(-1, 3, 3)
    cam_centers = cam_centers.reshape(-1, 3)
    extrinsics = np.zeros((len(fns), 4, 4)).astype("float32")
    extrinsics[:, :3, :3] = cam_rotations
    extrinsics[:, :3, 3] = cam_centers
    extrinsics[:, 3, 3] = 1

    intrinsics = []
    for i in range(len(fns)):
        intrinsic = {
            'width': 0,
            'height': 0,
            'f': 0,
            'cx': 0,
            'cy': 0,
            'b1': 0,
            'b2': 0,
            'k1': 0,
            'k2': 0,
            'k3': 0,
            'k4': 0,
            'p1': 0,
            'p2': 0,
            'p3': 0,
            'p4': 0,
        }
        cam_intrinsic = cam_intrinsics[i]
        intrinsic["width"] = resolutions[i][0]
        intrinsic["height"] = resolutions[i][1]
        intrinsic["cx"] = cam_intrinsic[0, 2]  # - resolutions[i][0] / 2
        intrinsic["cy"] = cam_intrinsic[1, 2]  # - resolutions[i][1] / 2
        intrinsic["f"] = cam_intrinsic[0, 0]
        intrinsics.append(copy.deepcopy(intrinsic))
    return extrinsics, intrinsics, fns


# get center
def get_center(pts):
    center = pts.mean(0)
    """
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    """
    return center


# get rotation
def get_rot(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def normalize_poses(poses,
                    pts,
                    up_est_method,
                    center_est_method,
                    cam_downscale=None):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[..., 3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[..., 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0., 0., -1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) *
                  t[:, None, :] +
                  torch.stack([cams_ori, cams_ori.roll(1, 0)], dim=-1)).mean(
                      (0, 2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = get_center(pts)
    else:
        raise NotImplementedError(
            f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(
            pts.numpy(),
            thresh=0.01)  # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(
            plane_eq)  # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1)  # plane normal as up direction
        signed_distance = (
            torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) *
            plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z  # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    elif up_est_method == 'z-axis':
        # center pose
        poses[:, :3, 1:3] *= -1.  # OpenGL => COLMAP
        # full 4x4 poses
        onehot = torch.tile(torch.tensor([0., 0., 0., 1.0]),
                            (poses.size()[0], 1, 1))
        poses = torch.cat((poses, onehot), axis=1).cpu().numpy()
        # normalization
        z = poses[:, :3, 1].mean(0) / (
            np.linalg.norm(poses[:, :3, 1].mean(0)) + 1e-10)
        # rotate averaged camera up direction to [0,0,1]
        R_z = get_rot(z, [0, 0, 1])
        R_z = torch.tensor(np.pad(R_z, [0, 1])).float()
        R_z[-1, -1] = 1
        poses = torch.as_tensor(poses[:, :3]).float()
    else:
        raise NotImplementedError(
            f'Unknown up estimation method: {up_est_method}')

    if not up_est_method == 'z-axis':
        # new axis
        y_ = torch.as_tensor([z[1], -z[0], 0.])
        x = F.normalize(y_.cross(z), dim=0)
        y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = R_z[:3, :3].T if up_est_method == 'z-axis' else torch.stack(
            [x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([
            poses,
            torch.as_tensor([[[0., 0., 0., 1.]]]).expand(
                poses.shape[0], -1, -1)
        ],
                               dim=1)
        inv_trans = torch.cat([
            torch.cat([R, torch.as_tensor([[0., 0., 0.]]).T], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ],
                              dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])],
                                     dim=-1)[..., None])[:, :3, 0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,
                                          3].min(0)[0], poses_norm[...,
                                                                   3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:, 0]) & (pts[:, 0] < poses_max[0]) &
                     (poses_min[1] < pts[:, 1]) & (pts[:, 1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([
            poses_norm,
            torch.as_tensor([[[0., 0., 0., 1.]]]).expand(
                poses_norm.shape[0], -1, -1)
        ],
                               dim=1)
        inv_trans = torch.cat([
            torch.cat([torch.eye(3), t], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ],
                              dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]

        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])],
                                     dim=-1)[..., None])[:, :3, 0]
        if up_est_method == 'z-axis':
            # rectify convention
            poses_norm[:, :3, 1:3] *= -1  # COLMAP => OpenGL
            poses_norm = poses_norm[:, [1, 0, 2], :]
            poses_norm[:, 2] *= -1
            pts = pts[:, [1, 0, 2]]
            pts[:, 2] *= -1

        # rescaling
        if cam_downscale:
            scale = cam_downscale
        else:
            # auto-scale with camera positions
            scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale
        pts = pts / scale

    else:
        # rotation and translation
        Rc = R_z[:3, :3].T if up_est_method == 'z-axis' else torch.stack(
            [x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([
            poses,
            torch.as_tensor([[[0., 0., 0., 1.]]]).expand(
                poses.shape[0], -1, -1)
        ],
                               dim=1)
        inv_trans = torch.cat(
            [torch.cat([R, t], dim=1),
             torch.as_tensor([[0., 0., 0., 1.]])],
            dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]  # (N_images, 3, 4)

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])],
                                     dim=-1)[..., None])[:, :3, 0]
        if up_est_method == 'z-axis':
            # rectify convention
            poses_norm[:, :3, 1:3] *= -1  # COLMAP => OpenGL
            poses_norm = poses_norm[:, [1, 0, 2], :]
            poses_norm[:, 2] *= -1
            pts = pts[:, [1, 0, 2]]
            pts[:, 2] *= -1

        # rescaling
        if cam_downscale:
            scale = cam_downscale
        else:
            # auto-scale with camera positions
            scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale
        pts = pts / scale

    return poses_norm, pts, R, t, scale


def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0., 0., 0.],
                             dtype=cameras.dtype,
                             device=cameras.device)
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.],
                         dtype=center.dtype,
                         device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:, None]],
                        axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)

    return all_c2w


class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        if not ColmapDatasetBase.initialized:
            assert os.path.isfile(self.config.pose_path)
            # NOTE: self.config.root_dir determines the 'WorkSpace' used in parser
            all_c2w, intrinsics, fns = bin2camera(self.config.root_dir,
                                                  self.config.pose_path)
            mask_ori_dir = os.path.join(self.config.root_dir, 'Mask')
            has_mask = os.path.exists(
                mask_ori_dir)  # TODO: support partial masks
            self.apply_mask = has_mask and self.config.apply_mask
            self.apply_depth = self.config.apply_depth
            mask_dir = os.path.join(self.config.root_dir,
                                    f'Mask_{self.config.img_downscale}')

            all_images, all_fg_masks, all_depths, all_depths_mask, directions = [], [], [], [], []

            for i, d in enumerate(all_c2w):
                intrinsic = intrinsics[i]
                H = int(intrinsic["height"])
                W = int(intrinsic["width"])

                if 'img_wh' in self.config:
                    w, h = self.config.img_wh
                    assert round(W / w * h) == H
                elif 'img_downscale' in self.config:
                    w, h = int(W / self.config.img_downscale +
                               0.5), int(H / self.config.img_downscale + 0.5)
                else:
                    raise KeyError(
                        "Either img_wh or img_downscale should be specified.")

                img_wh = (w, h)
                factor = w / W

                fx = fy = intrinsic[
                    "f"] * factor  # camdata[1].params[0] * factor
                cx = intrinsic["cx"] * factor
                cy = intrinsic["cy"] * factor

                direction = get_ray_directions(w, h, fx, fy, cx, cy).to(
                    self.rank
                ) if self.config.load_data_on_gpu else get_ray_directions(
                    w, h, fx, fy, cx, cy).cpu()
                directions.append(direction)
                if self.split in ['train', 'val']:
                    idx_tmp = fns[i].replace(self.config.root_dir,
                                             '')[1:].rfind('/')
                    img_folder = fns[i].replace(self.config.root_dir,
                                                '')[0:idx_tmp + 1]
                    img_path = fns[i].replace(
                        f"{img_folder}",
                        f"{img_folder}_{self.config.img_downscale}")
                    if not os.path.exists(img_path):
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        img_path = fns[i]
                    img = Image.open(img_path)
                    img_path = fns[i].replace(
                        img_folder,
                        f"{img_folder}_{self.config.img_downscale}")
                    if img.size[0] != w or img.size[1] != h:
                        img = img.resize(img_wh, Image.BICUBIC)
                        img.save(img_path)
                    img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]
                    img = img.to(
                        self.rank
                    ) if self.config.load_data_on_gpu else img.cpu()
                    if self.apply_mask:
                        mask_path = os.path.join(
                            mask_dir, fns[i].split("/")[-1][:-3] + 'png')
                        # mask_paths = list(filter(os.path.exists, mask_paths))
                        # assert len(mask_paths) == 1
                        if not os.path.exists(mask_path):
                            os.makedirs(os.path.join(
                                self.config.root_dir,
                                f"Mask_{self.config.img_downscale}"),
                                        exist_ok=True)
                            mask_path = os.path.join(
                                mask_ori_dir,
                                fns[i].split("/")[-1][:-3] + 'png')
                        mask = Image.open(mask_path)  # (H, W, 1)
                        if mask.size[0] != w or mask.size[1] != h:
                            mask = mask.resize(img_wh, Image.BICUBIC)
                            mask.save(
                                os.path.join(
                                    self.config.root_dir,
                                    f"Mask_{self.config.img_downscale}",
                                    fns[i].split("/")[-1][:-3] + 'png'))
                        mask = TF.to_tensor(mask)[0] * 255
                        mask_fg = torch.ones_like(mask, device=img.device)
                        mask_fg[mask == 20] = 0
                        mask_fg[mask == 80] = 0
                        mask_fg[mask == 83] = 0
                        mask_fg[mask == 102] = 0
                        mask = mask_fg
                    else:
                        mask = torch.ones_like(img[..., 0], device=img.device)
                    if self.apply_depth:
                        depth_folder = 'Depths'
                        depth_path = fns[i].replace(
                            f"{img_folder}",
                            f"/{depth_folder}_{self.config.img_downscale}"
                        ).replace(".jpg", ".tif")
                        if not os.path.exists(depth_path):
                            os.makedirs(os.path.dirname(depth_path),
                                        exist_ok=True)
                            depth_path = fns[i].replace(
                                f"{img_folder}",
                                f"/{depth_folder}").replace(".jpg", ".tif")
                        depth = Image.open(depth_path)
                        depth_path = fns[i].replace(
                            f"{img_folder}",
                            f"/{depth_folder}_{self.config.img_downscale}"
                        ).replace(".jpg", ".tif")
                        if depth.size[0] != w or depth.size[1] != h:
                            depth = depth.resize(img_wh, Image.BICUBIC)
                            depth.save(depth_path)
                        depth = TF.to_tensor(depth).permute(
                            1, 2, 0) / self.config.cam_downscale
                        depth = depth.to(
                            self.rank
                        ) if self.config.load_data_on_gpu else depth.cpu()
                        depth_mask = (depth > 0.0).to(bool)
                    else:
                        depth = torch.zeros_like(img[..., 0],
                                                 device=img.device)  # (h, w)
                        depth_mask = (depth > 0.0).to(bool)
                    all_fg_masks.append(mask)  # (h, w)
                    all_images.append(img)
                    all_depths.append(depth)  # (h, w)
                    all_depths_mask.append(depth_mask)

            directions = torch.stack(directions, dim=0)
            all_c2w = torch.tensor(all_c2w)[:, :3]
            all_c2w[:, :, 1:3] *= -1.  # COLMAP => OpenGL
            if self.config.pcd_path is not None:
                assert os.path.isfile(self.config.pcd_path)
                import open3d as o3d
                pts3d = np.asarray(
                    o3d.io.read_point_cloud(self.config.pcd_path).points)
                pts3d = torch.from_numpy(pts3d).float()
            else:
                print(colored('sparse point cloud not given', 'red'))
                pts3d = []
            if self.config.repose:
                if self.config.center_est_method == 'point':
                    print(
                        colored(
                            'scene centered on ' + os.path.join(
                                self.config.root_dir, self.config.pcd_path),
                            'blue'))
                all_c2w, pts3d, R, t, scale = normalize_poses(
                    all_c2w,
                    pts3d,
                    up_est_method=self.config.up_est_method,
                    center_est_method=self.config.center_est_method,
                    cam_downscale=self.config.cam_downscale)
            else:
                print(
                    colored(
                        'scene centered on ' + os.path.join(
                            self.config.root_dir, self.config.pcd_path),
                        'blue'))
                poses_min, poses_max = all_c2w[..., 3].min(0)[0], all_c2w[
                    ..., 3].max(0)[0]
                pts_fg = pts3d[(poses_min[0] < pts3d[:, 0])
                               & (pts3d[:, 0] < poses_max[0]) &
                               (poses_min[1] < pts3d[:, 1]) &
                               (pts3d[:, 1] < poses_max[1])]
                print(colored(get_center(pts3d), 'blue'))
                all_c2w[:, :, 3] -= get_center(pts3d)
                pts3d -= get_center(pts3d)

            ColmapDatasetBase.properties = {
                'w': w,
                'h': h,
                'img_wh': img_wh,
                'factor': factor,
                'has_mask': has_mask,
                'apply_mask': self.apply_mask,
                'apply_depth': self.apply_depth,
                'directions': directions,
                'pts3d': pts3d,
                'all_c2w': all_c2w,
                'all_images': all_images,
                'all_fg_masks': all_fg_masks,
                'all_depths': all_depths,
                'all_depths_mask': all_depths_mask,
                'transform_R': R,
                'transform_t': t,
                'transform_s': scale
            }

            ColmapDatasetBase.initialized = True

        for k, v in ColmapDatasetBase.properties.items():
            setattr(self, k, v)

        if self.split == 'test':
            self.all_c2w = create_spheric_poses(
                self.all_c2w[:, :, 3], n_steps=self.config.n_test_traj_steps)
            # NOTE: 300 is a hyper-parameter which determines the zoom-in scale
            self.all_c2w[:, :, 3] /= (300 / self.config.cam_downscale)
            self.all_images = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w, 3),
                dtype=torch.float32)
            self.all_fg_masks = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w),
                dtype=torch.float32)
            self.all_depths = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w),
                dtype=torch.float32)
            self.all_depths_mask = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w),
                dtype=torch.bool)
        else:
            self.all_images, self.all_fg_masks, self.all_depths, self.all_depths_mask = torch.stack(
                self.all_images, dim=0).float(), torch.stack(
                    self.all_fg_masks, dim=0).float(), torch.stack(
                        self.all_depths,
                        dim=0).float(), torch.stack(self.all_depths_mask,
                                                    dim=0).bool()
        """
        # for debug use
        from models.ray_utils import get_rays
        rays_o, rays_d = get_rays(self.directions.cpu(), self.all_c2w, keepdim=True)
        pts_out = []
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 0.0 0.0' for l in rays_o[:,0,0].reshape(-1, 3).tolist()]))

        t_vals = torch.linspace(0, 1, 8)
        z_vals = 0.05 * (1 - t_vals) + 0.5 * t_vals

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 0.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))
        
        open('cameras.txt', 'w').write('\n'.join(pts_out))
        open('scene.txt', 'w').write('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 0.0' for l in self.pts3d.view(-1, 3).tolist()]))

        exit(1)
        """

        self.all_c2w = self.all_c2w.float().to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank)
            self.all_fg_masks = self.all_fg_masks.to(self.rank)


class ColmapDataset(Dataset, ColmapDatasetBase):

    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {'index': index}


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):

    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('hmvs')
class ColmapDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(
                self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(
                self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(dataset,
                          num_workers=os.cpu_count(),
                          batch_size=batch_size,
                          pin_memory=True,
                          sampler=sampler)

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
