import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
import numpy as np


@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch['rays'], batch['camera_indices'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index'].to(self.dataset.all_images.device)
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x].to(self.rank)
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x].to(self.rank)
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
            if self.dataset.apply_depth:
                depth = self.dataset.all_depths[index, y, x].view(-1).to(self.rank)
                depth_mask = self.dataset.all_depths_mask[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions.to(self.rank)
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0].to(self.rank)
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            if self.dataset.apply_depth:
                depth = self.dataset.all_depths[index].view(-1).to(self.rank)
                depth_mask = self.dataset.all_depths_mask[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'camera_indices': index
        })      
        if self.dataset.apply_depth:
            batch.update({
                'depth': depth,
                'depth_mask': depth_mask,
            })      
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        loss_rgb_cos = torch.mean(1.0 - F.cosine_similarity(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]], dim=-1))
        self.log('train/loss_rgb_cos', loss_rgb_cos)
        loss += loss_rgb_cos * self.C(self.config.system.loss.lambda_rgb_cos)

        with torch.no_grad():
            rgb_mse = torch.mean((out['comp_rgb_full'] - batch['rgb']) ** 2, dim=1)
            exposure_mse = torch.mean((torch.zeros_like(batch['rgb']) - batch['rgb']) ** 2, dim=1) + 0.0001
        if self.C(self.config.system.loss.lambda_adaptive) > 0.0 and self.C(self.config.system.loss.lambda_adaptive) < 1.0:
            diff_rays = self.C(self.config.system.loss.lambda_adaptive) / (rgb_mse + self.C(self.config.system.loss.lambda_adaptive))
            diff_pts = diff_rays[out['ray_indices']]
        else:
            diff_pts = 1.0
        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2 * diff_pts).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        if self.dataset.apply_depth:
            # patchmatch_mask = torch.logical_and(patchmatch_valid_mask, out['rays_valid_full'].squeeze(-1))
            depth_mask = batch['depth_mask'][out['rays_valid_full'][...,0]]
            if depth_mask.shape[0] == (batch['fg_mask'] > 0.5).shape[0] and depth_mask.shape[0] == rgb_mse.shape[0]:
                depth_mask *= (batch['fg_mask'] > 0.5)
                depth_mask *= (rgb_mse / exposure_mse)
            loss_depth_l1 = F.l1_loss(out['sdf0_ray_distance'][out['rays_valid_full']] * depth_mask, batch['depth'][out['rays_valid_full'][...,0]] * depth_mask)
            self.log('train/loss_depth_l1', loss_depth_l1)
            loss += loss_depth_l1 * self.C(self.config.system.loss.lambda_depth_l1)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.apply_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_surface_bias) > 0:
            loss_bias = F.l1_loss(out['sdf_rendered'], torch.zeros_like(out['sdf_rendered']), reduction='mean')
            # loss_bias = F.l1_loss(out['depth'], torch.zeros_like(out['depth']), reduction='mean')
            loss += loss_bias * self.C(self.config.system.loss.lambda_surface_bias) 
        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            if self.global_step < self.config.system.loss.curvature_warmup_steps:
                curv_warmup_factor = (self.global_step+1)  / self.config.system.loss.curvature_warmup_steps
                loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature) * curv_warmup_factor
            else:
                # decay the weight of curv loss
                growth_ratio = self.config.model.geometry.xyz_encoding_config.per_level_scale
                growth_steps = self.config.model.geometry.xyz_encoding_config.update_steps
                curvature_weight = np.exp(-max(self.global_step - self.config.system.loss.curvature_warmup_steps,0) * np.log(growth_ratio) / growth_steps) * self.C(self.config.system.loss.lambda_curvature)
                loss += loss_curvature * curvature_weight

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0 and out['ray_indices'].nelement() > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0 and out['ray_indices_bg'][0].nelement() > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        # regulazing rendering weights (transparency, alpha), encourage them to be either 0 or 1, with 1e-3
        if self.C(self.config.system.loss.lambda_entropy) > 0:
            weights = out['weights'].clamp(1e-5, 1 - 1e-5)
            loss_entropy = binary_cross_entropy(weights, weights).mean()
            alpha = out['alpha'].clamp(1e-5, 1 - 1e-5)
            loss_entropy += binary_cross_entropy(alpha, alpha).mean()
            loss += self.C(self.config.system.loss.lambda_entropy) * loss_entropy
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.export()

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            filename=f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        
        try:
            self.dataset.transform_s
            # export with the original pose
            vertices = np.asarray(mesh['v_pos'])
            vertices *= self.dataset.transform_s
            vertices -= torch.squeeze(self.dataset.transform_t).repeat(vertices.shape[0], 1).numpy()
            vertices = np.transpose(torch.inverse(self.dataset.transform_R).numpy() @ np.transpose(vertices))
            mesh['v_pos'] = torch.tensor(vertices)
            if self.config.dataset.mesh_outpath != False:
                self.save_mesh(
                    filename=None,
                    filename_full=f"{self.config.dataset.mesh_outpath}",
                    **mesh
                )
            else:
                self.save_mesh(
                    filename=f"mesh_neus.obj",
                    **mesh
                )        
            if self.config.dataset.name == "hmvs":
                from scripts.gen_tasks import write_json
                write_json(status=0)
        except:
            print("Model is already saved in original scale.")
