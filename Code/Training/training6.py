"""
Conditional Finetuning/Training Script for RIM + Conditional Physics Forward Operator - AstroRIM_V1
Compatible with 96x96 FITS images.
Maintains per-lens conditioning with forward fidelity loss.
"""
import os
import glob
import time
import argparse
from datetime import datetime
import math
from typing import Optional
import numpy as np
from astropy.io import fits

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Utilities for weight initialization

def init_weights(module):
    """Xavier init for Conv and Linear, zeros for biases."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        if getattr(module, 'weight', None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, 'bias', None) is not None:
            nn.init.zeros_(module.bias)

# Dataset loading and management
class ConditionalLensingFitsDataset(Dataset):
    def __init__(self, files, augment=False):
        self.files = sorted(files)
        assert len(self.files) > 0, "No FITS files provided"
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _read_pair(self, fn):
        hdul = fits.open(fn, memmap=False)
        gt = None
        lensed = None

        for i, h in enumerate(hdul):
            extname_hdr = h.header.get('EXTNAME', '') if hasattr(h, 'header') else ''
            extname_attr = getattr(h, 'name', '') or ''
            name = (extname_hdr or extname_attr).upper()
            data = h.data
            if data is None:
                continue
            try:
                d = data.astype(np.float32)
            except Exception:
                continue
            if name == 'GT':
                gt = d
            elif name in ('LENSED', 'OBS', 'OBSERVED'):
                lensed = d

        if gt is None and hdul[0].data is not None:
            try:
                gt = hdul[0].data.astype(np.float32)
            except Exception:
                pass

        if lensed is None and len(hdul) > 1 and hdul[1].data is not None:
            try:
                lensed = hdul[1].data.astype(np.float32)
            except Exception:
                pass

        if gt is None or lensed is None:
            info_lines = []
            for idx, h in enumerate(hdul):
                name = getattr(h, 'name', '') or h.header.get('EXTNAME', '')
                shape = None if h.data is None else getattr(h.data, 'shape', None)
                keys = list(h.header.keys())[:12]
                info_lines.append(f"ext#{idx}: name='{name}' shape={shape} keys={keys}")
            hdul.close()
            raise RuntimeError(f"{fn} missing GT or LENSED HDU.\nDetected:\n" + "\n".join(info_lines))

        hdul.close()

        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        eps = 1e-8
        m = max(gt.max() if gt.size else 0.0, lensed.max() if lensed.size else 0.0, eps)
        gt = (gt / m).astype(np.float32)
        lensed = (lensed / m).astype(np.float32)

        return lensed, gt

    def __getitem__(self, idx):
        fn = self.files[idx]
        obs, gt = self._read_pair(fn)

        if self.augment:
            if np.random.rand() < 0.5:
                obs = np.fliplr(obs)
                gt = np.fliplr(gt)
            if np.random.rand() < 0.5:
                obs = np.flipud(obs)
                gt = np.flipud(gt)
            k = np.random.choice([0, 1, 2, 3])
            if k != 0:
                obs = np.rot90(obs, k)
                gt = np.rot90(gt, k)

        obs_t = torch.from_numpy(obs.copy()).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return obs_t, gt_t

# Lens Parameter Encoder

class LensParameterEncoder(nn.Module):
    """Encodes observed image to lens parameters (improved depth & GroupNorm)"""
    def __init__(self, input_channels=1, latent_dim=128, output_dim=9):
        super().__init__()
        ch1, ch2, ch3, ch4 = 32, 64, 96, 128
        self.conv1 = nn.Conv2d(input_channels, ch1, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, ch1)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, ch2)
        self.conv3 = nn.Conv2d(ch2, ch3, 3, padding=1)
        self.gn3 = nn.GroupNorm(8, ch3)
        self.conv4 = nn.Conv2d(ch3, ch4, 3, padding=1)
        self.gn4 = nn.GroupNorm(8, ch4)

        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(ch4 * 8 * 8, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim // 2)

        # Output heads
        self.b_head = nn.Linear(latent_dim // 2, 1)
        self.q_head = nn.Linear(latent_dim // 2, 1)
        self.phi_head = nn.Linear(latent_dim // 2, 1)
        self.x0_head = nn.Linear(latent_dim // 2, 1)
        self.y0_head = nn.Linear(latent_dim // 2, 1)
        self.gamma_head = nn.Linear(latent_dim // 2, 1)
        self.gamma_phi_head = nn.Linear(latent_dim // 2, 1)
        self.kappa_s_head = nn.Linear(latent_dim // 2, 1)
        self.rs_head = nn.Linear(latent_dim // 2, 1)

        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Parameter ranges
        b = 0.01 + 0.29 * torch.sigmoid(self.b_head(x))
        q = 0.2 + 0.8 * torch.sigmoid(self.q_head(x))
        phi = torch.pi * torch.tanh(self.phi_head(x))
        x0 = 0.2 * torch.tanh(self.x0_head(x))
        y0 = 0.2 * torch.tanh(self.y0_head(x))
        gamma = 0.2 * torch.sigmoid(self.gamma_head(x))
        gamma_phi = torch.pi * torch.tanh(self.gamma_phi_head(x))
        kappa_s = 0.005 + 0.495 * torch.sigmoid(self.kappa_s_head(x))
        rs = 0.05 + 0.45 * torch.sigmoid(self.rs_head(x))

        return {
            'b': b.squeeze(-1),
            'q': q.squeeze(-1),
            'phi': phi.squeeze(-1),
            'x0': x0.squeeze(-1),
            'y0': y0.squeeze(-1),
            'gamma': gamma.squeeze(-1),
            'gamma_phi': gamma_phi.squeeze(-1),
            'kappa_s': kappa_s.squeeze(-1),
            'rs': rs.squeeze(-1)
        }

# Conditional NFW Deflection

class ConditionalNFWDeflection(nn.Module):
    def __init__(self):
        super().__init__()

    def nfw_deflection_angle(self, x, y, kappa_s, rs, x0, y0):
        eps = 1e-8
        dx = x - x0.unsqueeze(-1).unsqueeze(-1)
        dy = y - y0.unsqueeze(-1).unsqueeze(-1)
        r = torch.sqrt(dx**2 + dy**2 + eps)

        x_norm = r / (rs.unsqueeze(-1).unsqueeze(-1) + eps)
        x_norm = torch.clamp(x_norm, min=1e-6, max=1e6)

        alpha_r = torch.zeros_like(x_norm, device=x.device)

        def safe_atanh(t):
            t = torch.clamp(t, min=-1.0 + 1e-6, max=1.0 - 1e-6)
            return 0.5 * (torch.log1p(t) - torch.log1p(-t))

        mask_lt1 = x_norm < 1.0
        mask_gt1 = x_norm > 1.0
        mask_eq1 = (~mask_lt1) & (~mask_gt1)

        if mask_lt1.any():
            xs = x_norm[mask_lt1]
            sqrt_term = torch.sqrt(torch.clamp(1.0 - xs**2, min=1e-12))
            u = torch.sqrt(torch.clamp((1.0 - xs) / (1.0 + xs), min=0.0, max=1.0 - 1e-6))
            atanh_u = safe_atanh(u)
            val = (4.0 / (xs + 1e-12)) * (1.0 - (2.0 * atanh_u / (sqrt_term + 1e-12)))
            alpha_r[mask_lt1] = val

        if mask_gt1.any():
            xl = x_norm[mask_gt1]
            sqrt_term = torch.sqrt(torch.clamp(xl**2 - 1.0, min=1e-12))
            v = torch.sqrt(torch.clamp((xl - 1.0) / (1.0 + xl), min=0.0))
            atan_v = torch.atan(v)
            val = (4.0 / (xl + 1e-12)) * (1.0 - (2.0 * atan_v / (sqrt_term + 1e-12)))
            alpha_r[mask_gt1] = val

        if mask_eq1.any():
            alpha_r[mask_eq1] = 2.0

        alpha_r = alpha_r * kappa_s.unsqueeze(-1).unsqueeze(-1) * rs.unsqueeze(-1).unsqueeze(-1)

        cos_theta = dx / (r + eps)
        sin_theta = dy / (r + eps)
        alpha_x = alpha_r * cos_theta
        alpha_y = alpha_r * sin_theta
        return alpha_x, alpha_y

    def nfw_convergence(self, x, y, kappa_s, rs, x0, y0):
        eps = 1e-8
        dx = x - x0.unsqueeze(-1).unsqueeze(-1)
        dy = y - y0.unsqueeze(-1).unsqueeze(-1)
        r = torch.sqrt(dx**2 + dy**2 + eps)
        x_norm = r / (rs.unsqueeze(-1).unsqueeze(-1) + eps)
        x_norm = torch.clamp(x_norm, min=1e-6, max=1e6)

        kappa = torch.zeros_like(x_norm, device=x.device)

        def safe_atanh(t):
            t = torch.clamp(t, min=-1.0 + 1e-6, max=1.0 - 1e-6)
            return 0.5 * (torch.log1p(t) - torch.log1p(-t))

        mask_lt1 = x_norm < 1.0
        mask_gt1 = x_norm > 1.0
        mask_eq1 = (~mask_lt1) & (~mask_gt1)

        if mask_lt1.any():
            xm = x_norm[mask_lt1]
            denom = xm**2 - 1.0
            root = torch.sqrt(torch.clamp(1.0 - xm**2, min=1e-12))
            u = torch.sqrt(torch.clamp((1.0 - xm) / (1.0 + xm), min=0.0, max=1.0 - 1e-6))
            atanh_u = safe_atanh(u)
            val = (2.0 * kappa_s.unsqueeze(-1).unsqueeze(-1) / (denom + 1e-12)) * \
                  (1.0 - (2.0 / (root + 1e-12)) * atanh_u)
            kappa[mask_lt1] = val

        if mask_gt1.any():
            xm = x_norm[mask_gt1]
            denom = xm**2 - 1.0
            root = torch.sqrt(torch.clamp(xm**2 - 1.0, min=1e-12))
            v = torch.sqrt(torch.clamp((xm - 1.0) / (1.0 + xm), min=0.0))
            atan_v = torch.atan(v)
            val = (2.0 * kappa_s.unsqueeze(-1).unsqueeze(-1) / (denom + 1e-12)) * \
                  (1.0 - (2.0 / (root + 1e-12)) * atan_v)
            kappa[mask_gt1] = val

        if mask_eq1.any():
            kappa[mask_eq1] = 2.0 * kappa_s.unsqueeze(-1).unsqueeze(-1) / 3.0

        return kappa

# Conditional Subhalo Component

class ConditionalSubhaloComponent(nn.Module):
    def __init__(self, n_subhalos=3):
        super().__init__()
        self.n_subhalos = n_subhalos

    def compute_deflection(self, x, y, subhalo_params):
        B = x.shape[0]
        H, W = x.shape[1], x.shape[2]
        alpha_x_total = torch.zeros(B, H, W, device=x.device)
        alpha_y_total = torch.zeros(B, H, W, device=x.device)
        params = subhalo_params.view(B, self.n_subhalos, 4)
        for i in range(self.n_subhalos):
            x_sub = params[:, i, 0].unsqueeze(-1).unsqueeze(-1)
            y_sub = params[:, i, 1].unsqueeze(-1).unsqueeze(-1)
            theta_e = params[:, i, 2].unsqueeze(-1).unsqueeze(-1)
            rc = params[:, i, 3].unsqueeze(-1).unsqueeze(-1)
            dx = x - x_sub
            dy = y - y_sub
            r = torch.sqrt(dx**2 + dy**2 + rc**2 + 1e-12)
            alpha_r = theta_e * (1.0 - rc / (r + rc + 1e-12))
            alpha_x_total += alpha_r * dx / (r + 1e-12)
            alpha_y_total += alpha_r * dy / (r + 1e-12)
        return alpha_x_total, alpha_y_total

# Conditional Physical Forward 

class ConditionalPhysicalForward(nn.Module):
    def __init__(self,
                 kernel_size: int = 21,
                 enforce_nonneg: bool = True,
                 init_fwhm: float = 3.0,
                 init_beta: float = 4.5,
                 pixel_scale: float = 0.05,
                 learn_residual_psf: bool = True,
                 residual_scale: float = 1e-2,
                 use_fft_conv: bool = True,
                 use_nfw: bool = True,
                 use_subhalos: bool = False,
                 n_subhalos: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.enforce_nonneg = enforce_nonneg
        self.pixel_scale = pixel_scale
        self.use_fft_conv = use_fft_conv
        self.use_nfw = use_nfw
        self.use_subhalos = use_subhalos
        self.n_subhalos = n_subhalos

        self.encoder = LensParameterEncoder(latent_dim=128)

        if self.use_subhalos:
            self.subhalo_encoder = nn.Sequential(
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, n_subhalos * 4)
            )
            self.subhalo_encoder.apply(init_weights)

        if self.use_nfw:
            self.nfw = ConditionalNFWDeflection()
        if self.use_subhalos:
            self.subhalos = ConditionalSubhaloComponent(n_subhalos=n_subhalos)

        if learn_residual_psf:
            resid = torch.randn(1, 1, kernel_size, kernel_size) * residual_scale
            self.raw_res_psf = nn.Parameter(resid)
        else:
            self.raw_res_psf = None

        self.log_fwhm = nn.Parameter(torch.log(torch.tensor(init_fwhm + 1e-12)))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(init_beta + 1e-12)))
        self.log_background = nn.Parameter(torch.log(torch.tensor(1e-2)))
        self.log_gain = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_sigma_read = nn.Parameter(torch.log(torch.tensor(1e-3)))

        self.apply(init_weights)

        if self.raw_res_psf is not None:
            with torch.no_grad():
                self.raw_res_psf.mul_(1e-2)

    @property
    def fwhm(self):
        return torch.exp(self.log_fwhm)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    @property
    def background(self):
        return torch.exp(self.log_background)

    @property
    def gain(self):
        return torch.exp(self.log_gain)

    @property
    def sigma_read(self):
        return torch.exp(self.log_sigma_read)

    def get_parametric_psf(self, size=None, device=None):
        if device is None:
            device = next(self.parameters()).device
        size = size or self.kernel_size
        fwhm = float(self.fwhm.detach().cpu().item())
        beta = float(self.beta.detach().cpu().item())
        alpha = fwhm / (2.0 * math.sqrt(2 ** (1.0 / beta) - 1.0) + 1e-12)
        k = size
        ax = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        rr2 = xx ** 2 + yy ** 2
        moff = (1.0 + (rr2 / (alpha ** 2))) ** (-beta)
        moff = moff / (moff.sum() + 1e-12)
        return moff.unsqueeze(0).unsqueeze(0)

    def get_psf(self):
        device = next(self.parameters()).device
        param_psf = self.get_parametric_psf(size=self.kernel_size, device=device)
        if self.raw_res_psf is None:
            k = param_psf
        else:
            resid = self.raw_res_psf.to(device)
            k = param_psf + resid
        if self.enforce_nonneg:
            k = torch.relu(k)
        s = k.sum(dim=(2, 3), keepdim=True)
        s = s.clamp_min(1e-12)
        k = k / s
        if self.raw_res_psf is not None:
            k = torch.clamp(k, min=0.0, max=1.0)
            k = k / (k.sum(dim=(2,3), keepdim=True) + 1e-12)
        return k

    def _build_mesh(self, B, H, W, device):
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((xv, yv), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def _compute_deflection(self, H, W, device, lens_params, subhalo_params=None):
        B = lens_params['b'].shape[0]
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='ij')
        xv = xv.unsqueeze(0).repeat(B, 1, 1)
        yv = yv.unsqueeze(0).repeat(B, 1, 1)

        b = lens_params['b'].view(B, 1, 1)
        q = lens_params['q'].view(B, 1, 1)
        phi = lens_params['phi'].view(B, 1, 1)
        x0 = lens_params['x0'].view(B, 1, 1)
        y0 = lens_params['y0'].view(B, 1, 1)
        gamma = lens_params['gamma'].view(B, 1, 1)
        gamma_phi = lens_params['gamma_phi'].view(B, 1, 1)

        rc = 0.01 * torch.ones_like(b)

        dx = xv - x0
        dy = yv - y0

        c = torch.cos(-phi)
        s = torch.sin(-phi)
        x_rot = c * dx - s * dy
        y_rot = s * dx + c * dy

        x_ell = x_rot * q
        r = torch.sqrt(x_ell ** 2 + y_rot ** 2 + rc ** 2)

        ax_ell = b * x_ell / (r + 1e-12)
        ay_ell = b * y_rot / (r + 1e-12)

        c2 = torch.cos(phi)
        s2 = torch.sin(phi)
        ax = c2 * ax_ell - s2 * ay_ell
        ay = s2 * ax_ell + c2 * ay_ell

        cos2 = torch.cos(2.0 * gamma_phi)
        sin2 = torch.sin(2.0 * gamma_phi)
        ax_s = gamma * (xv * cos2 + yv * sin2)
        ay_s = gamma * (xv * sin2 - yv * cos2)

        ax = ax + ax_s
        ay = ay + ay_s

        if self.use_nfw:
            kappa_s = lens_params['kappa_s'].view(B)
            rs = lens_params['rs'].view(B)
            ax_nfw, ay_nfw = self.nfw.nfw_deflection_angle(xv, yv, kappa_s, rs, x0.squeeze(), y0.squeeze())
            ax = ax + ax_nfw
            ay = ay + ay_nfw

        if self.use_subhalos and subhalo_params is not None:
            ax_sub, ay_sub = self.subhalos.compute_deflection(xv, yv, subhalo_params)
            ax = ax + ax_sub
            ay = ay + ay_sub

        return ax.unsqueeze(1), ay.unsqueeze(1)

    def encode_lens_parameters(self, obs):
        x = obs
        x_features = F.relu(self.encoder.gn1(self.encoder.conv1(x)))
        x_features = self.encoder.pool(x_features)
        x_features = F.relu(self.encoder.gn2(self.encoder.conv2(x_features)))
        x_features = self.encoder.pool(x_features)
        x_features = F.relu(self.encoder.gn3(self.encoder.conv3(x_features)))
        x_features = self.encoder.pool(x_features)
        x_features = F.relu(self.encoder.gn4(self.encoder.conv4(x_features)))
        x_features = self.encoder.pool(x_features)
        x_features = self.encoder.adaptive_pool(x_features)
        features_flat = x_features.view(x.size(0), -1)
        lens_params = self.encoder(obs)
        subhalo_params = None
        if self.use_subhalos:
            subhalo_params = self.subhalo_encoder(features_flat)
            B = subhalo_params.shape[0]
            subhalo_params = subhalo_params.view(B, self.n_subhalos, 4)
            subhalo_params[:, :, 0] = 0.5 * torch.tanh(subhalo_params[:, :, 0])
            subhalo_params[:, :, 1] = 0.5 * torch.tanh(subhalo_params[:, :, 1])
            subhalo_params[:, :, 2] = 0.05 * torch.sigmoid(subhalo_params[:, :, 2])
            subhalo_params[:, :, 3] = 0.01 + 0.09 * torch.sigmoid(subhalo_params[:, :, 3])
        return lens_params, subhalo_params

    def forward(self, src, obs, add_noise: bool = False, return_params: bool = False):
        B, C, H, W = src.shape
        device = src.device
        lens_params, subhalo_params = self.encode_lens_parameters(obs)
        ax, ay = self._compute_deflection(H, W, device, lens_params, subhalo_params)

        grid = self._build_mesh(B, H, W, device)
        grid_x = grid[..., 0] - ax.squeeze(1)
        grid_y = grid[..., 1] - ay.squeeze(1)
        samp_grid = torch.stack((grid_x, grid_y), dim=-1)
        eps = 1e-6
        samp_grid = torch.clamp(samp_grid, min=-1.0 + eps, max=1.0 - eps)
        y_warp = F.grid_sample(src, samp_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        psf = self.get_psf().to(device)
        if self.use_fft_conv and max(H, W) > 64:
            kH, kW = psf.shape[-2], psf.shape[-1]
            pad_h = kH // 2
            pad_w = kW // 2
            y_padded = F.pad(y_warp, (pad_w, pad_w, pad_h, pad_h), mode='constant')
            y_conv = F.conv2d(y_padded, psf, padding=0)
        else:
            y_conv = F.conv2d(y_warp, psf, padding=self.kernel_size // 2)

        y_conv = y_conv + self.background

        if add_noise:
            gain = self.gain
            counts = torch.clamp(y_conv * gain, min=0.0)
            counts_noisy = torch.poisson(counts)
            sigma_read = self.sigma_read
            gauss = torch.randn_like(counts_noisy) * sigma_read
            counts_noisy = counts_noisy + gauss
            y_noisy = counts_noisy / (gain + 1e-12)
            if return_params:
                return y_noisy, lens_params, subhalo_params
            return y_noisy
        else:
            if return_params:
                return y_conv, lens_params, subhalo_params
            return y_conv

    def adjoint(self, residual, obs):
        """Adjoint operator - requires obs for conditioning"""
        lens_params, subhalo_params = self.encode_lens_parameters(obs)

        psf = self.get_psf().to(next(self.parameters()).device)
        psf_flipped = torch.flip(psf, dims=(2, 3))

        if self.use_fft_conv and max(residual.shape[-2:]) > 64:
            kH, kW = psf_flipped.shape[-2], psf_flipped.shape[-1]
            pad_h = kH // 2
            pad_w = kW // 2
            r_padded = F.pad(residual, (pad_w, pad_w, pad_h, pad_h), mode='constant')
            r_conv = F.conv2d(r_padded, psf_flipped, padding=0)
        else:
            r_conv = F.conv2d(residual, psf_flipped, padding=self.kernel_size // 2)

        B, C, H, W = residual.shape
        device = residual.device
        grid = self._build_mesh(B, H, W, device)
        ax, ay = self._compute_deflection(H, W, device, lens_params, subhalo_params)

        grid_x = grid[..., 0] + ax.squeeze(1)
        grid_y = grid[..., 1] + ay.squeeze(1)
        adj_grid = torch.stack((grid_x, grid_y), dim=-1)
        eps = 1e-6
        adj_grid = torch.clamp(adj_grid, min=-1.0 + eps, max=1.0 - eps)
        src_grad = F.grid_sample(r_conv, adj_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return src_grad

    def compute_regularization_loss(self, lens_params, subhalo_params=None):
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        b = lens_params['b']
        q = lens_params['q']
        gamma = lens_params['gamma']
        reg_loss += torch.mean((b - 0.08) ** 2) * 0.1
        reg_loss += torch.mean((q - 0.7) ** 2) * 0.1
        reg_loss += torch.mean(gamma ** 2) * 0.1
        if self.use_nfw:
            kappa_s = lens_params['kappa_s']
            rs = lens_params['rs']
            reg_loss += torch.mean((kappa_s - 0.05) ** 2) * 0.05
            reg_loss += torch.mean((rs - 0.2) ** 2) * 0.05
        if self.use_subhalos and subhalo_params is not None:
            B = subhalo_params.shape[0]
            sub_params = subhalo_params.view(B, self.n_subhalos, 4)
            theta_e = sub_params[:, :, 2]
            reg_loss += torch.mean(theta_e ** 2) * 0.1
            x_sub = sub_params[:, :, 0]
            y_sub = sub_params[:, :, 1]
            reg_loss += torch.mean(torch.relu(torch.abs(x_sub) - 0.9) ** 2) * 0.01
            reg_loss += torch.mean(torch.relu(torch.abs(y_sub) - 0.9) ** 2) * 0.01
            if self.n_subhalos > 1:
                for i in range(self.n_subhalos):
                    for j in range(i + 1, self.n_subhalos):
                        dx = x_sub[:, i] - x_sub[:, j]
                        dy = y_sub[:, i] - y_sub[:, j]
                        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
                        reg_loss += torch.mean(torch.relu(0.1 - dist) ** 2) * 0.1
        return reg_loss

# RIM (Modified for Conditional inputs from the Forward Operator)

class ConvGate(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class ConditionalRIMCell(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        in_ch = 1 + 1 + hidden_dim
        self.gates = ConvGate(in_ch, hidden_dim)
        self.candidate = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.step = nn.Parameter(torch.tensor(0.1))
        self.to_image = nn.Conv2d(self.hidden_dim, 1, 3, padding=1)

    def forward(self, x, grad, h):
        combined = torch.cat([x, grad, h], dim=1)
        gates = torch.sigmoid(self.gates(combined))
        update_gate = gates
        cand = torch.tanh(self.candidate(combined))
        h_new = (1 - update_gate) * h + update_gate * cand
        delta = self.step * self.to_image(h_new)
        x_new = x - delta
        return x_new, h_new

class ConditionalRIM(nn.Module):
    def __init__(self, n_iter=8, hidden_dim=128):
        super().__init__()
        self.n_iter = n_iter
        self.hidden_dim = hidden_dim
        self.cell = ConditionalRIMCell(hidden_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    def forward(self, y, obs, forward_operator):
        """Forward pass with observation conditioning"""
        B, C, H, W = y.shape
        device = y.device
        h = torch.zeros(B, self.hidden_dim, H, W, device=device)

        x = forward_operator.adjoint(y, obs)

        for _ in range(self.n_iter):
            y_sim = forward_operator.forward(x, obs)
            residual = y_sim - y
            grad = forward_operator.adjoint(residual, obs)
            x, h = self.cell(x, grad, h)

        x = x + self.refine(h)
        return x

# Utility Functions
def compute_grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            try:
                param_norm = p.grad.data.norm(2)
                total += param_norm.item() ** 2
            except Exception:
                pass
    return float(total ** 0.5)

def save_reconstructions(epoch, model, forward_operator, val_loader, device, num_examples=6, out_dir='recons'):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    forward_operator.eval()

    try:
        batch = next(iter(val_loader))
    except StopIteration:
        print("Validation loader empty; skipping reconstructions.")
        return

    obs, gt = batch
    obs = obs.to(device)
    gt = gt.to(device)

    with torch.no_grad():
        recon = model(obs, obs, forward_operator)

    B = obs.shape[0]
    n = min(B, num_examples)

    obs_np = obs.cpu().numpy()
    recon_np = recon.cpu().numpy()
    gt_np = gt.cpu().numpy()

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 3*n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i in range(n):
        o = np.clip(obs_np[i, 0], 0.0, 1.0)
        r = np.clip(recon_np[i, 0], 0.0, 1.0)
        g = np.clip(gt_np[i, 0], 0.0, 1.0)

        axes[i, 0].imshow(o, origin='lower', cmap='viridis')
        axes[i, 0].set_title('Observed')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(r, origin='lower', cmap='viridis')
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(g, origin='lower', cmap='viridis')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')

    plt.suptitle(f'Conditional Reconstructions - Epoch {epoch}', fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'cond_recons_epoch_{epoch}.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved reconstructions to {out_path}")

# Conditional Finetuning and Training Loop
def conditional_finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"CONDITIONAL FINETUNING WITH PER-LENS PARAMETERS")
    print(f"{'='*80}\n")
    print(f"Device: {device}")
    print(f"Forward fidelity weight: λ = {args.lambda_forward_fidelity:.2e}")
    print(f"{'='*80}\n")

    # Dataset
    all_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.fits")))
    if len(all_files) == 0:
        raise RuntimeError(f"No .fits files found in {args.dataset_dir}")

    train_files, val_files = train_test_split(all_files, test_size=args.val_split, random_state=42)

    train_dataset = ConditionalLensingFitsDataset(train_files, augment=args.augment)
    val_dataset = ConditionalLensingFitsDataset(val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type=='cuda'), num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=(device.type=='cuda'), num_workers=args.num_workers)

    # Instantiate improved conditional forward operator (from document 3)
    forward_operator = ConditionalPhysicalForward(
        kernel_size=args.kernel_size,
        enforce_nonneg=True,
        init_fwhm=3.0,
        init_beta=4.5,
        pixel_scale=0.05,
        learn_residual_psf=True,
        residual_scale=1e-2,
        use_fft_conv=True,
        use_nfw=args.use_nfw,
        use_subhalos=args.use_subhalos,
        n_subhalos=args.n_subhalos
    ).to(device)

    # Instantiate conditional RIM
    model = ConditionalRIM(n_iter=args.n_iter, hidden_dim=args.hidden_dim).to(device)

    start_epoch = 0
    best_val_loss = float('inf')

    # Load checkpoints if provided
    if args.rim_checkpoint and os.path.exists(args.rim_checkpoint):
        print(f"Loading RIM weights from {args.rim_checkpoint}")
        sd = torch.load(args.rim_checkpoint, map_location='cpu')
        try:
            if 'model_state' in sd:
                model.load_state_dict(sd['model_state'], strict=False)
            else:
                model.load_state_dict(sd, strict=False)
            print("✓ Loaded RIM checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load RIM checkpoint: {e}")

    if args.forward_checkpoint and os.path.exists(args.forward_checkpoint):
        print(f"Loading Forward operator weights from {args.forward_checkpoint}")
        sd = torch.load(args.forward_checkpoint, map_location='cpu')
        try:
            if 'forward_state' in sd:
                forward_operator.load_state_dict(sd['forward_state'], strict=False)
            elif 'state_dict' in sd:
                forward_operator.load_state_dict(sd['state_dict'], strict=False)
            else:
                forward_operator.load_state_dict(sd, strict=False)
            print("✓ Loaded forward operator checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load forward checkpoint: {e}")

    forward_operator.to(device)
    model.to(device)

    # Freeze logic based on finetune mode
    if args.finetune_mode == 'rim':
        for p in forward_operator.parameters():
            p.requires_grad = False
        print("→ Freezing forward operator (finetuning RIM only)")
    elif args.finetune_mode == 'forward':
        for p in model.parameters():
            p.requires_grad = False
        print("→ Freezing RIM (finetuning forward operator only)")
    else:
        print("→ Finetuning both RIM and forward operator")

    # Optional PSF freeze
    if args.freeze_psf:
        print("→ Freezing PSF parameters in forward operator")
        for name, p in forward_operator.named_parameters():
            if any(k in name for k in ['raw_res_psf', 'log_fwhm', 'log_beta']):
                p.requires_grad = False

    # Setup optimizer with separate param groups
    params = []
    if args.finetune_mode in ['both', 'rim']:
        rim_params = [p for p in model.parameters() if p.requires_grad]
        if rim_params:
            params.append({"params": rim_params, "lr": args.lr_rim})

    if args.finetune_mode in ['both', 'forward']:
        forward_params = [p for p in forward_operator.parameters() if p.requires_grad]
        if forward_params:
            params.append({"params": forward_params, "lr": args.lr_forward})

    if len(params) == 0:
        raise RuntimeError("No parameters to optimize (check finetune_mode / freeze flags)")

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.use_amp))

    # Loss history tracking
    loss_history = {
        "train": [], "val": [], "val_ssim": [], "grad_norm": [],
        "forward_fidelity": [], "param_reg": []
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume from checkpoint if provided
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        ck = torch.load(args.resume_checkpoint, map_location='cpu')
        start_epoch = ck.get('epoch', 0)
        if 'optimizer' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer'])
                print("✓ Loaded optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load optimizer: {e}")
        if 'best_val_loss' in ck:
            best_val_loss = ck['best_val_loss']
        print(f"→ Resuming from epoch {start_epoch}")

    print("\n" + "="*80)
    print("STARTING CONDITIONAL FINETUNING")
    print("="*80 + "\n")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        epoch_idx = epoch + 1
        start_epoch_time = time.time()

        model.train()
        forward_operator.train()

        running_loss = 0.0
        running_forward_fidelity = 0.0
        running_param_reg = 0.0
        grad_norm_epoch = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch_idx}/{args.num_epochs}")
        for obs, gt in loop:
            obs = obs.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda' and args.use_amp)):
                recon = model(obs, obs, forward_operator)
                mse_recon = F.mse_loss(recon, gt, reduction='mean')

                lensed_pred, lens_params, subhalo_params = forward_operator(gt, obs, return_params=True)
                forward_fidelity_loss = F.mse_loss(lensed_pred, obs, reduction='mean')

                psf = forward_operator.get_psf()
                lap = psf - F.avg_pool2d(psf, 3, 1, padding=1)
                psf_pen = (lap ** 2).mean()

                param_reg = forward_operator.compute_regularization_loss(lens_params, subhalo_params)

                loss = (mse_recon +
                        args.lambda_forward_fidelity * forward_fidelity_loss +
                        args.lambda_psf * psf_pen +
                        args.lambda_param_reg * param_reg)

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at epoch {epoch_idx}, skipping batch. loss={loss}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if args.use_amp and device.type == 'cuda':
                scaler.unscale_(optimizer)

            trainable_params = [p for p in list(model.parameters()) + list(forward_operator.parameters()) if p.requires_grad]
            grad_norm = compute_grad_norm(trainable_params)
            try:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
            except Exception:
                pass

            try:
                scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                print(f"[ERROR] Optimizer step failed: {e}. Skipping update and zeroing grads.")
                optimizer.zero_grad(set_to_none=True)
                continue

            for p in trainable_params:
                try:
                    if p.grad is not None and (not torch.isfinite(p.grad).all()):
                        p.grad = None
                except Exception:
                    pass
                try:
                    if not torch.isfinite(p).all():
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e6, neginf=-1e6)
                except Exception:
                    pass

            if getattr(forward_operator, 'raw_res_psf', None) is not None:
                with torch.no_grad():
                    try:
                        forward_operator.raw_res_psf.data = torch.nan_to_num(forward_operator.raw_res_psf.data, nan=0.0, posinf=1e3, neginf=-1e3)
                        forward_operator.raw_res_psf.data.clamp_(-1.0, 1.0)
                    except Exception:
                        pass

            running_loss += loss.item() * obs.size(0)
            running_forward_fidelity += forward_fidelity_loss.item() * obs.size(0)
            running_param_reg += param_reg.item() * obs.size(0)
            grad_norm_epoch += grad_norm

            loop.set_postfix({
                "loss": f"{loss.item():.6e}",
                "fwd_fid": f"{forward_fidelity_loss.item():.6e}",
                "param_reg": f"{param_reg.item():.6e}",
                "g_norm": f"{grad_norm:.3f}"
            })

        avg_train_loss = running_loss / max(1, len(train_dataset))
        avg_forward_fidelity = running_forward_fidelity / max(1, len(train_dataset))
        avg_param_reg = running_param_reg / max(1, len(train_dataset))
        avg_grad_norm = grad_norm_epoch / max(1, len(train_loader))

        loss_history['train'].append(avg_train_loss)
        loss_history['forward_fidelity'].append(avg_forward_fidelity)
        loss_history['param_reg'].append(avg_param_reg)
        loss_history['grad_norm'].append(avg_grad_norm)

        # Validation
        model.eval()
        forward_operator.eval()
        val_loss = 0.0
        val_ssim = 0.0
        val_forward_fidelity = 0.0
        n_val = 0

        with torch.no_grad():
            for obs, gt in val_loader:
                obs = obs.to(device)
                gt = gt.to(device)

                recon = model(obs, obs, forward_operator)
                l = F.mse_loss(recon, gt).item()
                if not np.isfinite(l):
                    print("[WARN] Non-finite validation mse, replacing with large value.")
                    l = 1e6
                val_loss += l * obs.size(0)

                lensed_pred = forward_operator(gt, obs)
                fwd_fid = F.mse_loss(lensed_pred, obs).item()
                if not np.isfinite(fwd_fid):
                    fwd_fid = 1e6
                val_forward_fidelity += fwd_fid * obs.size(0)

                recon_np = recon.cpu().numpy()
                gt_np = gt.cpu().numpy()
                for i in range(recon_np.shape[0]):
                    r = np.clip(recon_np[i, 0], 0.0, 1.0)
                    g = np.clip(gt_np[i, 0], 0.0, 1.0)
                    try:
                        val_ssim += ssim(g, r, data_range=1.0)
                    except Exception:
                        val_ssim += 0.0
                n_val += recon_np.shape[0]

        avg_val_loss = val_loss / max(1, n_val)
        avg_val_ssim = val_ssim / max(1, n_val)
        avg_val_forward_fidelity = val_forward_fidelity / max(1, n_val)

        loss_history['val'].append(avg_val_loss)
        loss_history['val_ssim'].append(avg_val_ssim)

        epoch_time = time.time() - start_epoch_time

        print(f"\n[Epoch {epoch_idx}]")
        print(f"  Train Loss: {avg_train_loss:.6e} | Fwd Fidelity: {avg_forward_fidelity:.6e} | Param Reg: {avg_param_reg:.6e}")
        print(f"  Val Loss: {avg_val_loss:.6e} | Val Fwd Fid: {avg_val_forward_fidelity:.6e}")
        print(f"  Val SSIM: {avg_val_ssim:.4f} | Grad Norm: {avg_grad_norm:.3f} | Time: {epoch_time:.1f}s")

        try:
            scheduler.step(avg_val_loss)
        except Exception:
            pass

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"✓ New best val loss: {best_val_loss:.6e}")

            rim_best_path = os.path.join(args.output_dir, args.save_rim_best)
            forward_best_path = os.path.join(args.output_dir, args.save_forward_best)

            torch.save(model.state_dict(), rim_best_path)
            torch.save(forward_operator.state_dict(), forward_best_path)

            print(f"  → Saved RIM: {rim_best_path}")
            print(f"  → Saved Forward: {forward_best_path}")

        if epoch_idx % args.checkpoint_every == 0 or epoch_idx == args.num_epochs:
            ck = {
                'epoch': epoch_idx,
                'model_state': model.state_dict(),
                'forward_state': forward_operator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            ck_path = os.path.join(args.checkpoint_dir, f"cond_finetune_checkpoint_epoch_{epoch_idx}.pt")
            torch.save(ck, ck_path)
            print(f"  → Checkpoint: {ck_path}")

            with torch.no_grad():
                k = forward_operator.get_psf().cpu().numpy()[0, 0]
                plt.figure(figsize=(4, 4))
                plt.imshow(k, cmap='hot')
                plt.colorbar()
                plt.title(f'PSF Kernel - Epoch {epoch_idx}')
                plt.axis('off')
                plt.tight_layout()
                psf_path = os.path.join(args.checkpoint_dir, f'cond_psf_epoch_{epoch_idx}.png')
                plt.savefig(psf_path, dpi=300)
                plt.close()
                print(f"  → PSF: {psf_path}")

        if epoch_idx % args.recon_every == 0 or epoch_idx == args.num_epochs:
            try:
                out_recon_dir = os.path.join(args.recon_out_dir, f'epoch_{epoch_idx}')
                save_reconstructions(epoch_idx, model, forward_operator, val_loader, device,
                                     num_examples=args.recon_num_examples, out_dir=out_recon_dir)
            except Exception as e:
                print(f"Failed to save reconstructions: {e}")

    # Final loss curves
    print("\nGenerating loss curves...")

    mse_curve_path = os.path.join(args.output_dir, 'cond_mse_curve_finetune.png')
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['train'], label='Train MSE', linewidth=2)
    plt.plot(loss_history['val'], label='Val MSE', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Conditional Reconstruction Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(mse_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ MSE curve: {mse_curve_path}")

    ssim_curve_path = os.path.join(args.output_dir, 'cond_ssim_curve_finetune.png')
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['val_ssim'], label='Val SSIM', linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('Structural Similarity', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(ssim_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ SSIM curve: {ssim_curve_path}")

    fidelity_curve_path = os.path.join(args.output_dir, 'cond_forward_fidelity_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['forward_fidelity'], label='Train Forward Fidelity',
             linewidth=2, color='purple')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Forward Fidelity MSE', fontsize=12)
    plt.title('Per-Lens Forward Operator Physics Accuracy\n(Lower = Better Match to Simulator)',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(fidelity_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Forward fidelity curve: {fidelity_curve_path}")

    param_reg_path = os.path.join(args.output_dir, 'cond_param_reg_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['param_reg'], label='Parameter Regularization',
             linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Regularization Loss', fontsize=12)
    plt.title('Per-Lens Parameter Regularization', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(param_reg_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Parameter reg curve: {param_reg_path}")

    print("\n" + "="*80)
    print("CONDITIONAL FINETUNING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.6e}")
    if loss_history['forward_fidelity']:
        print(f"Final forward fidelity: {loss_history['forward_fidelity'][-1]:.6e}")
    if loss_history['param_reg']:
        print(f"Final param reg: {loss_history['param_reg'][-1]:.6e}")
    print("="*80 + "\n")

# CLI 
def parse_args():
    p = argparse.ArgumentParser(description="Conditional finetune RIM + Physics Forward operator")

    p.add_argument('--dataset-dir', type=str,
                   default="YOUR DATASET DIRECTORY HERE")
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--augment', action='store_true')

    p.add_argument('--num-epochs', type=int, default=50)
    p.add_argument('--n-iter', type=int, default=10)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--kernel-size', type=int, default=21)

    p.add_argument('--finetune-mode', choices=['both', 'rim', 'forward'], default='both')
    p.add_argument('--freeze-psf', action='store_true')

    p.add_argument('--rim-checkpoint', type=str, default="/cond_rim_finetune_best.pt")
    p.add_argument('--forward-checkpoint', type=str, default="/cond_forward_finetune_best.pt")
    p.add_argument('--resume-checkpoint', type=str, default='YOUR RESUME CHECKPOINT PATH HERE')

    p.add_argument('--lr-rim', type=float, default=1e-5)
    p.add_argument('--lr-forward', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=1e-6)

    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-patience', type=int, default=6)

    p.add_argument('--lambda-psf', type=float, default=1e-3)
    p.add_argument('--lambda-param-reg', type=float, default=1e-3,
                   help="Weight for per-lens parameter regularization")
    p.add_argument('--lambda-forward-fidelity', type=float, default=0.5,
                   help="Weight for forward fidelity loss")

    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--use-amp', action='store_true')

    p.add_argument('--checkpoint-every', type=int, default=5)
    p.add_argument('--output-dir', type=str, default="YOUR OUTPUT DIRECTORY HERE")
    p.add_argument('--checkpoint-dir', type=str, default="YOUR CHECKPOINT DIRECTORY HERE")
    p.add_argument('--save-rim-best', type=str, default='cond_rim_finetune_best.pt')
    p.add_argument('--save-forward-best', type=str, default='cond_forward_finetune_best.pt')

    p.add_argument('--recon-every', type=int, default=1)
    p.add_argument('--recon-num-examples', type=int, default=8)
    p.add_argument('--recon-out-dir', type=str, default="YOUR RECONSTRUCTION OUTPUT DIRECTORY HERE")

    p.add_argument('--use-nfw', action='store_true', default=True,
                   help="Enable NFW halo in forward operator")
    p.add_argument('--use-subhalos', action='store_true', default=False,
                   help="Enable subhalos in forward operator")
    p.add_argument('--n-subhalos', type=int, default=2,
                   help="Number of subhalos")

    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)
    conditional_finetune(args)