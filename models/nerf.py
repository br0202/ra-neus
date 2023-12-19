import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, accumulate_along_rays


@models.register('nerf')
class NeRFModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.second_integral:
            self.se_geometry = models.se_make(self.config.se_geometry.se_name, self.config.se_geometry)
            print('self.se_geometry', self.se_geometry)
            # self.se_geometry = models.make(self.config.geometry.name, self.config.geometry)

        if self.config.learned_background:
            self.occupancy_grid_res = 256
            self.near_plane, self.far_plane = 0.2, 1e4
            self.cone_angle = 10**(math.log10(self.far_plane) / self.config.num_samples_per_ray) - 1. # approximate
            self.render_step_size = 0.01 # render_step_size = max(distance_to_camera * self.cone_angle, self.render_step_size)
            self.contraction_type = ContractionType.UN_BOUNDED_SPHERE
        else:
            self.occupancy_grid_res = 128
            self.near_plane, self.far_plane = None, None
            self.cone_angle = 0.0
            self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
            self.contraction_type = ContractionType.AABB

        self.geometry.contraction_type = self.contraction_type

        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=self.occupancy_grid_res,
                contraction_type=self.contraction_type
            )
        self.randomized = self.config.randomized
        self.background_color = None
    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)

        def occ_eval_fn(x):
            density, _ = self.geometry(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size) based on taylor series
            return density[...,None] * self.render_step_size
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn)

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def forward_(self, rays, camera_indices):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry(positions)
            return density[...,None]

        def se_sigma_fn(se_t_starts, se_t_ends, se_ray_indices):
            se_ray_indices = se_ray_indices.long()
            t_origins = rays_o[se_ray_indices]
            t_dirs = rays_d[se_ray_indices]
            positions = t_origins + t_dirs * (se_t_starts + se_t_ends) / 2.
            se_density, _ = self.se_geometry(positions)
            return se_density[..., None]
        
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, feature = self.geometry(positions) 
            rgb = self.texture(feature, t_dirs)
            return rgb, density[...,None]

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None if self.config.learned_background else self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=self.near_plane, far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=self.cone_angle,
                alpha_thre=0.0
            )

            # print('ray_indices', ray_indices)
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints  
        intervals = t_ends - t_starts

        density, feature = self.geometry(positions)
        # print('density', density.shape, 'feature', feature.shape)   # density torch.Size([210352]) feature torch.Size([210352, 16])
        rgb = self.texture(feature, t_dirs, camera_indices.to(self.rank), ray_indices)

        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       

        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })

        if self.config.second_integral:
            self.se_near_plane = depth.max() - 0.05   #  tensor(0.4519)
            self.se_far_plane = self.config.se_far_plane
            with torch.no_grad():
                se_ray_indices, se_t_starts, se_t_ends = ray_marching(
                    rays_o, rays_d,
                    scene_aabb=None if self.config.learned_background else self.scene_aabb,
                    grid=self.occupancy_grid if self.config.grid_prune else None,
                    sigma_fn=se_sigma_fn,
                    near_plane=self.se_near_plane, far_plane=self.se_far_plane,
                    render_step_size=self.render_step_size,
                    stratified=self.randomized,
                    cone_angle=self.cone_angle,
                    alpha_thre=0.0
                )

            se_ray_indices = se_ray_indices.long()
            se_t_origins = rays_o[se_ray_indices]
            se_t_dirs = re_rays_d[se_ray_indices]
            se_midpoints = (se_t_starts + se_t_ends) / 2.
            se_positions = se_t_origins + se_t_dirs * se_midpoints

            se_density, _ = self.se_geometry(se_positions)
            se_weights = render_weight_from_density(se_t_starts, se_t_ends, se_density[..., None], ray_indices=se_ray_indices,
                                                 n_rays=n_rays)
            opa_occ = accumulate_along_rays(se_weights, se_ray_indices, values=None, n_rays=n_rays)   # n_rays????
            out['opa_occ'] = opa_occ
        
        return out

    def forward(self, rays, camera_indices):
        # print('rays', rays)
        if self.training:
            out = self.forward_(rays, camera_indices)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays, camera_indices)
        return {
            **out,
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank))
            viewdirs = torch.zeros(feature.shape[0], 3).to(feature)
            viewdirs[...,2] = -1. # set the viewing directions to be -z (looking down)
            rgb = self.texture(feature, viewdirs).clamp(0,1)
            mesh['v_rgb'] = rgb.cpu()
        return mesh
