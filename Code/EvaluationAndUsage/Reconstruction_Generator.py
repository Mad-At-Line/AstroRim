#!/usr/bin/env python3
"""
Evaluation script for conditional RIM + physics forward operator models.
Compatible with the conditional training architecture from document 2.
"""

import os
import glob
import time
import argparse
import math
from typing import Optional
import numpy as np
from astropy.io import fits
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# CONFIG (paste your paths here; set to None to use CLI)

INPUT_FILE = None
INPUT_DIR = r"YOUR INPUT DIRECTORY PATH"
OUT_DIR = r"YOUR OUTPUT DIRECTORY PATH"
MODEL_PATH = r"YOUR RIM MODEL PATH"
FORWARD_PATH = r"YOUR FORWARD MODEL PATH"
DEVICE = None  # 'cpu' or 'cuda' or None auto-detect
KERNEL_SIZE = None
N_ITER = None
RESCALE_OUTPUT = True
SAVE_PNG = True
DEBUG_FITS_STRUCTURE = False
USE_NFW = True  # Should match your training config
USE_SUBHALOS = False  # Should match your training config
N_SUBHALOS = 2  # Should match your training config

# Helper: Xavier initialization

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

# Lens Parameter Encoder

class LensParameterEncoder(nn.Module):
    """Encodes observed image to lens parameters"""
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

# Conditional RIM
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

# FITS I/O utilities 

def read_fits_pair(fn):
    """Robustly read FITS files from different simulator versions."""
    hdul = fits.open(fn, memmap=False)
    gt = None
    lensed = None
    hdrs = {}
    
    if DEBUG_FITS_STRUCTURE:
        print(f"  Debug: {os.path.basename(fn)} has {len(hdul)} HDUs:")
        for idx, h in enumerate(hdul):
            name = getattr(h, 'name', '') or h.header.get('EXTNAME', '')
            has_data = h.data is not None
            shape = h.data.shape if has_data else None
            hdu_type = type(h).__name__
            print(f"    HDU[{idx}]: type={hdu_type}, name='{name}', has_data={has_data}, shape={shape}")
    
    # First pass: Look for explicitly named HDUs
    for idx, h in enumerate(hdul):
        name = getattr(h, 'name', '') or h.header.get('EXTNAME', '')
        name = str(name).upper().strip()
        data = h.data
        
        if name == 'GT' and data is not None:
            gt = data.astype(np.float32)
            hdrs['GT'] = h.header
        elif name in ('LENSED', 'OBS', 'OBSERVED') and data is not None:
            lensed = data.astype(np.float32)
            hdrs['LENSED'] = h.header
    
    # Second pass: Handle v3/v4 format
    if gt is None and len(hdul) >= 2:
        primary_data = hdul[0].data
        
        if primary_data is not None:
            if primary_data.ndim > 2:
                primary_data = np.squeeze(primary_data)
            
            if primary_data.ndim == 2:
                gt = primary_data.astype(np.float32)
                hdrs['GT'] = hdul[0].header
                print(f"  [v3/v4 format] GT from Primary[0]")
                
                if lensed is None:
                    for idx in range(1, len(hdul)):
                        h = hdul[idx]
                        if h.data is not None:
                            second_data = h.data
                            if second_data.ndim > 2:
                                second_data = np.squeeze(second_data)
                            
                            if second_data.ndim == 2:
                                lensed = second_data.astype(np.float32)
                                hdrs['LENSED'] = h.header
                                print(f"  [v3/v4 format] LENSED from HDU[{idx}]")
                                break
    
    # Third pass: Fallback
    if lensed is None:
        if hdul[0].data is not None:
            primary_data = hdul[0].data
            if primary_data.ndim > 2:
                primary_data = np.squeeze(primary_data)
            if primary_data.ndim == 2:
                lensed = primary_data.astype(np.float32)
                hdrs['LENSED'] = hdul[0].header
                print(f"  [Fallback] Using Primary[0] as LENSED (no GT available)")
    
    hdul.close()
    
    if lensed is None:
        raise RuntimeError(f"No valid LENSED data found in {fn}")
    
    lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if gt is not None:
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if gt.shape != lensed.shape:
            print(f"  Warning: GT shape {gt.shape} != LENSED shape {lensed.shape}, discarding GT")
            gt = None
    
    if gt is None:
        print(f"  Note: No GT found - metrics will not be computed")
    
    return lensed, gt, hdrs

def save_recon_fits(outpath, lensed, recon, gt=None, hdrs=None):
    prih = fits.PrimaryHDU()
    hdul = fits.HDUList([prih])
    hdul.append(fits.ImageHDU(data=lensed.astype(np.float32), name='LENSED', header=(hdrs.get('LENSED') if hdrs else None)))
    if gt is not None:
        hdul.append(fits.ImageHDU(data=gt.astype(np.float32), name='GT', header=(hdrs.get('GT') if hdrs else None)))
    hdul.append(fits.ImageHDU(data=recon.astype(np.float32), name='RECON'))
    hdul.writeto(outpath, overwrite=True)

def save_preview_png(png_path, lensed, recon, gt=None, vmax=None):
    ncols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(lensed, origin='lower')
    axes[0].set_title('LENSED')
    axes[0].axis('off')
    axes[1].imshow(recon, origin='lower', vmax=vmax)
    axes[1].set_title('RECON')
    axes[1].axis('off')
    if gt is not None:
        axes[2].imshow(gt, origin='lower', vmax=vmax)
        axes[2].set_title('GT')
        axes[2].axis('off')
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

def create_top8_visualization(results_data, out_dir):
    """Create top-8 visualization"""
    valid_results = [(fn, data) for fn, data in results_data.items() 
                     if data.get('ssim') is not None and data.get('gt_norm') is not None]
    
    if len(valid_results) == 0:
        print("  Warning: No valid results with GT to create top-8 visualization")
        return
    
    valid_results.sort(key=lambda x: x[1]['ssim'], reverse=True)
    top8 = valid_results[:min(8, len(valid_results))]
    n_show = len(top8)
    
    if n_show == 0:
        return
    
    avg_ssim = np.mean([data['ssim'] for _, data in top8])
    avg_mse = np.mean([data['mse_norm'] for _, data in top8])
    
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4*n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Top {n_show} Reconstructions by SSIM\nAverage SSIM: {avg_ssim:.4f} | Average MSE: {avg_mse:.6f}', 
                fontsize=16, y=0.995)
    
    for idx, (fn, data) in enumerate(top8):
        basename = os.path.basename(fn)
        lensed_norm = data['lensed_norm']
        gt_norm = data['gt_norm']
        recon_norm = data['recon_norm']
        ssim_val = data['ssim']
        mse_val = data['mse_norm']
        
        vmax = np.percentile(np.concatenate([lensed_norm.flatten(), 
                                             gt_norm.flatten(), 
                                             recon_norm.flatten()]), 99.5)
        
        axes[idx, 0].imshow(lensed_norm, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        axes[idx, 0].set_title(f'LENSED', fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(gt_norm, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        axes[idx, 1].set_title(f'GT', fontsize=10)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(recon_norm, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        axes[idx, 2].set_title(f'RECON', fontsize=10)
        axes[idx, 2].axis('off')
        
        text_y = 0.98 - (idx / n_show)
        fig.text(0.5, text_y, 
                f'#{idx+1}: {basename}',
                ha='center', va='top', fontsize=9, weight='bold',
                transform=fig.transFigure)
        fig.text(0.5, text_y - 0.01, 
                f'SSIM: {ssim_val:.4f} | MSE: {mse_val:.6f}',
                ha='center', va='top', fontsize=8,
                transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_path = os.path.join(out_dir, 'top8_reconstructions.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved top-{n_show} visualization to: {out_path}")

def identify_sim_script(filename):
    """Identify which simulation script generated this file"""
    basename = os.path.basename(filename)
    
    if basename.startswith('rim_simv2_'):
        return 'simgenv2'
    elif basename.startswith('rim_simv3_'):
        return 'simgenv3'
    elif basename.startswith('rim_simv4_'):
        return 'simgenv4'
    elif basename.startswith('rim_simv5-6_'):
        return 'simgenv5/v6'
    elif basename.startswith('rim_simv1_'):
        return 'simgenv1'
    else:
        return 'unknown'

# Checkpoint loaders

def load_forward_checkpoint(forward_operator, ck_path, map_device='cpu'):
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f"Forward checkpoint not found: {ck_path}")
    state = torch.load(ck_path, map_location=map_device)
    if isinstance(state, dict):
        if 'forward_state' in state and isinstance(state['forward_state'], dict):
            forward_operator.load_state_dict(state['forward_state'], strict=False)
            return
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            forward_operator.load_state_dict(state['state_dict'], strict=False)
            return
        forward_operator.load_state_dict(state, strict=False)
    else:
        forward_operator.load_state_dict(state, strict=False)

def load_rim_checkpoint(model, ck_path, map_device='cpu'):
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f"RIM checkpoint not found: {ck_path}")
    state = torch.load(ck_path, map_location=map_device)
    if isinstance(state, dict):
        if 'model_state' in state and isinstance(state['model_state'], dict):
            model.load_state_dict(state['model_state'], strict=False)
            return
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            model.load_state_dict(state['state_dict'], strict=False)
            return
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)

# Inference per file for conditional models

def infer_file(fn, model, forward_operator, device, out_dir, rescale=True, save_png=True):
    lensed, gt, hdrs = read_fits_pair(fn)
    eps = 1e-8
    
    # Normalize (Very important you keep the same as training, I made that mistake, default is assume max normalization)
    if gt is not None:
        normalization_factor = max(lensed.max(), gt.max(), eps)
    else:
        normalization_factor = max(lensed.max(), eps)
    
    lensed_norm = lensed / normalization_factor
    gt_norm = gt / normalization_factor if gt is not None else None
    
    if lensed_norm.ndim != 2:
        raise RuntimeError("Expected 2D images")
    
    obs_t = torch.from_numpy(lensed_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    forward_operator.eval()
    with torch.no_grad():
        # Pass obs_t twice (y and obs for conditional RIM)
        recon_t = model(obs_t, obs_t, forward_operator)
    recon_norm = recon_t.cpu().numpy()[0, 0]
    recon_norm = np.clip(recon_norm, 0.0, 1.0)

    # Compute metrics
    metrics = {}
    if gt_norm is not None:
        try:
            metrics['mse_norm'] = float(np.mean((gt_norm - recon_norm) ** 2))
            metrics['ssim'] = float(ssim(gt_norm, recon_norm, data_range=1.0))
            
            gt_orig = gt_norm * normalization_factor
            recon_orig = recon_norm * normalization_factor
            metrics['mse_orig'] = float(np.mean((gt_orig - recon_orig) ** 2))
        except Exception as e:
            print(f"Warning: Error computing metrics for {fn}: {e}")
            metrics['mse_norm'] = None
            metrics['mse_orig'] = None
            metrics['ssim'] = None

    # Prepare output
    if rescale:
        lensed_out = lensed.astype(np.float32)
        recon_out = (recon_norm * normalization_factor).astype(np.float32)
        gt_out = gt.astype(np.float32) if gt is not None else None
    else:
        lensed_out = lensed_norm.astype(np.float32)
        recon_out = recon_norm.astype(np.float32)
        gt_out = gt_norm.astype(np.float32) if gt_norm is not None else None

    base = os.path.splitext(os.path.basename(fn))[0]
    out_fits = os.path.join(out_dir, f"{base}_recon.fits")
    save_recon_fits(out_fits, lensed_out, recon_out, gt_out, hdrs=hdrs)
    
    if save_png:
        png_path = os.path.join(out_dir, f"{base}_recon_preview.png")
        vmax = np.percentile(recon_out, 99.0)
        save_preview_png(png_path, lensed_out, recon_out, gt_out, vmax=vmax)

    result_data = metrics.copy()
    result_data['lensed_norm'] = lensed_norm
    result_data['gt_norm'] = gt_norm
    result_data['recon_norm'] = recon_norm
    
    return out_fits, result_data

# CLI / main

def parse_args():
    p = argparse.ArgumentParser(description="Inference for conditional RIM + Physical forward operator")
    p.add_argument('--input-file', type=str, default=None)
    p.add_argument('--input-dir', type=str, default='.')
    p.add_argument('--out-dir', type=str, default='./results')
    p.add_argument('--model-path', type=str, required=False)
    p.add_argument('--forward-path', type=str, required=False)
    p.add_argument('--kernel-size', type=int, default=21)
    p.add_argument('--n-iter', type=int, default=10)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--device', type=str, default=None, help='cpu or cuda')
    p.add_argument('--no-rescale', dest='rescale', action='store_false')
    p.add_argument('--no-png', dest='save_png', action='store_true')
    p.add_argument('--use-nfw', action='store_true', default=True)
    p.add_argument('--use-subhalos', action='store_true', default=False)
    p.add_argument('--n-subhalos', type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()

    # Apply USER CONFIG overrides
    if INPUT_FILE is not None:
        args.input_file = INPUT_FILE
    if INPUT_DIR is not None:
        args.input_dir = INPUT_DIR
    if OUT_DIR is not None:
        args.out_dir = OUT_DIR
    if MODEL_PATH is not None:
        args.model_path = MODEL_PATH
    if FORWARD_PATH is not None:
        args.forward_path = FORWARD_PATH
    if DEVICE is not None:
        args.device = DEVICE
    if KERNEL_SIZE is not None:
        args.kernel_size = KERNEL_SIZE
    if N_ITER is not None:
        args.n_iter = N_ITER
    
    # Override with config
    args.use_nfw = USE_NFW
    args.use_subhalos = USE_SUBHALOS
    args.n_subhalos = N_SUBHALOS

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"NFW enabled: {args.use_nfw}, Subhalos enabled: {args.use_subhalos}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Instantiate conditional models
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
    
    model = ConditionalRIM(n_iter=args.n_iter, hidden_dim=args.hidden_dim).to(device)

    # Load weights
    if args.forward_path and os.path.isfile(args.forward_path):
        try:
            load_forward_checkpoint(forward_operator, args.forward_path, map_device=device)
            print(f"Loaded forward operator weights from {args.forward_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load forward operator: {e}")
    else:
        raise FileNotFoundError(f"Forward operator file not found: {args.forward_path}")

    if args.model_path and os.path.isfile(args.model_path):
        try:
            load_rim_checkpoint(model, args.model_path, map_device=device)
            print(f"Loaded RIM model weights from {args.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load RIM model: {e}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Prepare file list
    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.fits")))
    if len(files) == 0:
        print("No files found. Check input-file/input-dir.")
        return

    print(f"\nProcessing {len(files)} files...")
    t0 = time.time()
    results = {}
    for fn in files:
        print(f"Processing {fn} ...")
        try:
            out_fits, result_data = infer_file(fn, model, forward_operator, device, args.out_dir,
                                               rescale=args.rescale, save_png=args.save_png)
            print(f"  -> wrote {out_fits}")
            if result_data.get('mse_norm') is not None:
                print(f"     MSE (norm): {result_data['mse_norm']:.6f}, SSIM: {result_data.get('ssim', 'N/A'):.4f}")
            results[fn] = result_data
        except Exception as e:
            print(f"  Error processing {fn}: {e}")
            results[fn] = {'error': str(e)}

    dt = time.time() - t0
    print(f"\n✓ Done. Processed {len(files)} files in {dt:.1f}s")

    # Create visualizations
    try:
        create_top8_visualization(results, args.out_dir)
    except Exception as e:
        print(f"Warning: Could not create top-8 visualization: {e}")

    # Write summary CSV
    try:
        import csv
        csvp = os.path.join(args.out_dir, 'recon_summary.csv')
        with open(csvp, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'mse_norm', 'mse_orig', 'ssim', 'error'])
            for f, m in results.items():
                if m is None or 'error' in m:
                    w.writerow([f, '', '', '', m.get('error', 'Unknown error')])
                else:
                    w.writerow([f, m.get('mse_norm', ''), m.get('mse_orig', ''), 
                              m.get('ssim', ''), ''])
        print(f"Summary CSV written to {csvp}")
    except Exception as e:
        print(f"Warning: Could not write CSV: {e}")

if __name__ == '__main__':
    main()
