"""
Mass Profile Analysis and Visualization for Conditional Lensing Models.
Usage examples:
  python mass_profile_analysis_improved.py --forward-model model.pt --input single.fits --output-dir out \
      --pixscale-method psf_fwhm --assumed-seeing 0.8 --interactive --include-3d --save-individual-panels
  python mass_profile_analysis_improved.py --input /path/to/fits_dir --n-procs 4 --hdf5-output --summary-pdf
"""
import os
import argparse
import json
from astropy import cosmology as astro_cosmo
import warnings
import math
import logging
import multiprocessing as mp
import tempfile
from functools import partial
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy import units as u
from astropy.constants import c as C_light, G as G_const, M_sun
from astropy.cosmology import Planck18, FlatLambdaCDM
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy import cosmology as astro_cosmo

# Optional niceties
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

try:
    import h5py
    HDF5_AVAILABLE = True
except Exception:
    HDF5_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

try:
    from joblib import Memory
    CACHE_AVAILABLE = True
except Exception:
    CACHE_AVAILABLE = False

# Small image processing helpers
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


# Logging setup
logger = logging.getLogger("mass_profile_analyzer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%H:%M:%S")
ch.setFormatter(fmt)
logger.addHandler(ch)

# Simple disk cache (optional)

CACHE_DIR = os.path.join(tempfile.gettempdir(), "mpa_cache")
if CACHE_AVAILABLE:
    memory = Memory(CACHE_DIR, verbose=0)
else:
    memory = None

# FORWARD MODEL CLASS DEFINITION 

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

    def compute_convergence_map(self, obs, H=None, W=None):
        """Compute convergence map from encoded lens parameters."""
        device = obs.device
        B = obs.shape[0]
        
        if H is None or W is None:
            H, W = obs.shape[-2], obs.shape[-1]
        
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='ij')
        xv = xv.unsqueeze(0).repeat(B, 1, 1)
        yv = yv.unsqueeze(0).repeat(B, 1, 1)
        
        lens_params, _ = self.encode_lens_parameters(obs)
        
        # SIS convergence
        b = lens_params['b'].view(B, 1, 1)
        q = lens_params['q'].view(B, 1, 1)
        phi = lens_params['phi'].view(B, 1, 1)
        x0 = lens_params['x0'].view(B, 1, 1)
        y0 = lens_params['y0'].view(B, 1, 1)
        
        rc = 0.01 * torch.ones_like(b)
        dx = xv - x0
        dy = yv - y0
        
        c = torch.cos(-phi)
        s = torch.sin(-phi)
        x_rot = c * dx - s * dy
        y_rot = s * dx + c * dy
        
        x_ell = x_rot * q
        r = torch.sqrt(x_ell**2 + y_rot**2 + rc**2)
        kappa_sis = b / (2.0 * r + 1e-12)
        
        # External shear convergence (approximate)
        gamma = lens_params['gamma'].view(B, 1, 1)
        gamma_phi = lens_params['gamma_phi'].view(B, 1, 1)
        
        # NFW convergence if enabled
        if self.use_nfw:
            kappa_s = lens_params['kappa_s'].view(B)
            rs = lens_params['rs'].view(B)
            kappa_nfw = self.nfw.nfw_convergence(xv, yv, kappa_s, rs, x0.squeeze(), y0.squeeze())
            kappa_total = kappa_sis + kappa_nfw
        else:
            kappa_total = kappa_sis
        
        return kappa_total.unsqueeze(1), lens_params

# END OF FORWARD MODEL CLASS DEFINITION

# Small utility functions

def safe_nanmean(a):
    """Return mean ignoring NaNs (handles empty)"""
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    m = np.nanmean(a)
    if np.isnan(m):
        return 0.0
    return float(m)

def ensure_finite_arr(arr, fill=0.0):
    """Replace NaN/inf in array with 'fill' and return float ndarray"""
    a = np.array(arr, dtype=float)
    a = np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)
    return a

# Gaussian fit helpers

def gaussian_2d(xy, amp, x0, y0, sx, sy, theta, bg):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sx*sx) + (np.sin(theta)**2)/(2*sy*sy)
    b = -(np.sin(2*theta))/(4*sx*sx) + (np.sin(2*theta))/(4*sy*sy)
    c = (np.sin(theta)**2)/(2*sx*sx) + (np.cos(theta)**2)/(2*sy*sy)
    return (amp*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) + bg).ravel()

def measure_fwhm_from_stamp(stamp):
    """
    Fit a symmetric 2D Gaussian to 'stamp' and return approximate FWHM (pixels).
    Returns None if the fit fails or result is unreasonable.
    """
    stamp = np.asarray(stamp, dtype=float)
    if stamp.size == 0:
        return None
    ny, nx = stamp.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    amp = float(np.percentile(stamp, 99) - np.percentile(stamp, 10))
    if amp <= 0:
        return None
    x0 = nx / 2.0
    y0 = ny / 2.0
    sx = 1.5
    sy = 1.5
    theta = 0.0
    bg = float(np.percentile(stamp, 10))
    p0 = (amp, x0, y0, sx, sy, theta, bg)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            popt, _ = curve_fit(gaussian_2d, (X, Y), stamp.ravel(), p0=p0, maxfev=4000)
        sx_fit, sy_fit = abs(popt[3]), abs(popt[4])
        sigma_eff = np.sqrt(sx_fit * sy_fit)
        fwhm = 2.355 * sigma_eff
        if 0.3 < fwhm < 200.0:
            return float(fwhm)
    except Exception:
        return None
    return None

def estimate_pixel_scale_from_psf(obs_img: np.ndarray, assumed_seeing_arcsec: float = 0.8,
                                  stamp_size: int = 25, min_peaks: int = 3, verbose: bool = False):
    """
    Estimate pixel scale (arcsec/pixel) from observed image by measuring
    stellar FWHM in pixels and assuming a seeing FWHM 'assumed_seeing_arcsec'.

    Returns: float arcsec/pixel or None
    """
    obs = np.asarray(obs_img, dtype=float)
    H, W = obs.shape
    thresh = np.percentile(obs, 90)
    coords = peak_local_max(obs, min_distance=5, threshold_abs=thresh)
    if coords.size == 0:
        if verbose:
            logger.warning("No peaks found for PSF estimation.")
        return None
    fwhms = []
    half = stamp_size // 2
    for (r, c) in coords:
        if r - half < 0 or r + half >= H or c - half < 0 or c + half >= W:
            continue
        stamp = obs[r-half:r+half+1, c-half:c+half+1]
        fwhm_pix = measure_fwhm_from_stamp(stamp)
        if fwhm_pix is not None and fwhm_pix > 0:
            fwhms.append(fwhm_pix)
        if len(fwhms) >= 50:
            break
    if len(fwhms) < min_peaks:
        if verbose:
            logger.warning("Insufficient good star stamps for PSF FWHM estimation.")
        return None
    median_fwhm_pix = float(np.median(fwhms))
    pixel_scale = assumed_seeing_arcsec / median_fwhm_pix
    if verbose:
        logger.info(f"Measured median FWHM = {median_fwhm_pix:.2f} px -> pixel scale = {pixel_scale:.5f} arcsec/pix")
    return pixel_scale

# ----------------------------
# MassProfileAnalyzer (CHANGED/ADDED cosmology & Σ_crit & conversions)
# ----------------------------
class MassProfileAnalyzer:
    """
    Compute lensing-derived masses and profiles with cosmology support.

    Parameters
    ----------
    pixel_scale : float
        Arcsec per pixel used to convert image-normalized coordinates.
    cosmology : astropy.cosmology instance
        Cosmology object to use for distance calculations (angular diameter distances).
    z_lens : float
        Lens redshift (needed for Σ_crit and physical conversions).
    z_source : float
        Source redshift.

    Notes
    -----
    - Many calculations assume the kappa map is on the lens plane and pixel_scale relates normalized image coords to arcsec.
    """
    def __init__(self, pixel_scale: float = 0.05, cosmology=None, z_lens: float = 0.5, z_source: float = 2.0):
        self.pixel_scale = float(pixel_scale)
        self.cosmo = cosmology or Planck18
        self.z_lens = z_lens
        self.z_source = z_source

    def angular_diameter_distances(self):
        """Return D_l, D_s, D_ls as astropy Quantities (meters)."""
        D_l = self.cosmo.angular_diameter_distance(self.z_lens).to(u.m)
        D_s = self.cosmo.angular_diameter_distance(self.z_source).to(u.m)
        # compute D_ls properly (angular diameter distance between lens and source)
        try:
            D_ls = self.cosmo.angular_diameter_distance_z1z2(self.z_lens, self.z_source).to(u.m)
        except Exception:
            # fallback approximate: (1+z_s)/(1+z_ls) difference but better to use astropy's method
            D_ls = (D_s - D_l)
        return D_l, D_s, D_ls

    def sigma_crit(self):
        """
        Compute critical surface density Σ_crit in kg/m^2 and also in M_sun / kpc^2 for easy to read numbers.
        Σ_crit = (c^2 / (4πG)) * (D_s / (D_l * D_ls))
        """
        D_l, D_s, D_ls = self.angular_diameter_distances()
        const = (C_light**2 / (4.0 * math.pi * G_const)).to(u.kg * u.m**-1)  # kg / m
        sigma_crit = (const * (D_s / (D_l * D_ls))).to(u.kg / u.m**2)
        # conversions
        sigma_crit_Msun_per_kpc2 = (sigma_crit / (M_sun / (u.kpc**2))).value
        return sigma_crit, sigma_crit_Msun_per_kpc2

    def einstein_radius_arcsec(self, lens_params: Dict[str, torch.Tensor], img_shape: Tuple[int, int]):
        """
        Return Einstein radius in arcseconds from normalized b parameter.
        Normalization: b_norm * arcsec_per_norm where arcsec_per_norm = (W/2) * pixel_scale
        """
        b_norm = float(lens_params['b'].cpu().item())
        H, W = img_shape
        arcsec_per_norm = (W / 2.0) * self.pixel_scale
        return float(b_norm * arcsec_per_norm)

    def sigma_v_from_thetaE(self, thetaE_arcsec, img_shape):
        """
        Estimate velocity dispersion (σ_v) from Einstein radius assuming SIS:
        θ_E = 4π (σ_v^2 / c^2) * (D_ls / D_s)  => σ_v = c * sqrt(θ_E / (4π) * (D_s / D_ls))
        θ_E in arcsec -> convert to radians.
        Returns σ_v in km/s (float).
        """
        theta_rad = (thetaE_arcsec * u.arcsec).to(u.rad).value
        D_l, D_s, D_ls = self.angular_diameter_distances()
        # convert to dimensionless ratio
        ratio = (D_s / D_ls).value
        sigma_si = (C_light.value * math.sqrt(max(theta_rad / (4.0 * math.pi) * ratio, 0.0)))
        sigma_kms = sigma_si / 1000.0
        return float(sigma_kms)

    def mass_within_radius(self, kappa_map: np.ndarray, radius_arcsec: float, center_xy_norm: Tuple[float, float]):
        """
        Compute mass within radius (physical) using Σ_crit and kappa_map:
         M = Σ_crit * ∑_pixels κ_pixel * area_pixel_phys

        Returns dict with masses in M_sun and statistical estimates.
        """
        kappa = ensure_finite_arr(kappa_map, fill=0.0)
        H, W = kappa.shape
        # center in normalized coords [-1,1] -> convert to pixel coords
        x0_norm, y0_norm = center_xy_norm
        x0_pix = (x0_norm + 1.0) * W / 2.0
        y0_pix = (y0_norm + 1.0) * H / 2.0
        # pixel scale arcsec/pix
        pix_arcsec = self.pixel_scale
        pix_rad = (pix_arcsec * u.arcsec).to(u.rad).value
        # physical area per pixel (m^2) at lens plane
        D_l, _, _ = self.angular_diameter_distances()
        area_pixel_m2 = ( (D_l.value * pix_rad) ** 2 )
        # Σ_crit in kg/m^2
        sigma_crit_kg_m2, sigma_crit_Msun_kpc2 = self.sigma_crit()
        sigma_crit = sigma_crit_kg_m2.value
        # mask of pixels inside radius
        # radius_arcsec provided; convert to pixel units
        radius_pix = float(radius_arcsec / pix_arcsec)
        Y, X = np.indices((H, W))
        rpix = np.sqrt((X - x0_pix)**2 + (Y - y0_pix)**2)
        mask = rpix <= radius_pix
        if mask.sum() == 0:
            return {'mass_Msun': 0.0, 'n_pixels': 0}
        # compute mass (kg) = Σ_crit * Σ_kappa_pixel * area_pixel
        sum_kappa = np.sum(kappa[mask])
        mass_kg = sigma_crit * sum_kappa * area_pixel_m2
        mass_Msun = mass_kg / M_sun.to(u.kg).value
        return {'mass_Msun': float(mass_Msun), 'n_pixels': int(mask.sum()), 'sigma_crit_kg_m2': float(sigma_crit)}

    def radial_profile(self, kappa_map: np.ndarray, center_xy_norm: Tuple[float, float], n_bins: int = 40):
        """
        Compute radial profile (kappa vs radius in arcsec) returning:
          radii_arcsec (centers), kappa_mean, kappa_std, counts
        """
        kappa = ensure_finite_arr(kappa_map, fill=np.nan)
        H, W = kappa.shape
        x0_norm, y0_norm = center_xy_norm
        x0_pix = (x0_norm + 1.0) * W / 2.0
        y0_pix = (y0_norm + 1.0) * H / 2.0
        xs = np.linspace(0, W-1, W)
        ys = np.linspace(0, H-1, H)
        Yv, Xv = np.meshgrid(ys, xs, indexing='ij')
        rpix = np.sqrt((Xv - x0_pix)**2 + (Yv - y0_pix)**2)
        r_max = float(np.sqrt((max(x0_pix, W-1-x0_pix))**2 + (max(y0_pix, H-1-y0_pix))**2))
        bins = np.linspace(0.0, r_max, n_bins+1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        profile = np.zeros(n_bins)
        profile_std = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            mask = (rpix >= bins[i]) & (rpix < bins[i+1])
            vals = kappa[mask]
            vals = vals[~np.isnan(vals)]
            counts[i] = vals.size
            if vals.size > 0:
                profile[i] = float(np.mean(vals))
                profile_std[i] = float(np.std(vals))
            else:
                profile[i] = 0.0
                profile_std[i] = 0.0
        # convert centers pixels -> arcsec
        radii_arcsec = centers * self.pixel_scale
        valid = counts > 0
        return radii_arcsec[valid], profile[valid], profile_std[valid], counts[valid]

# Visualization

def stretch_image(img: np.ndarray, stretch='asinh', scale=5.0, clip_percentiles=(0.5, 99.9)):
    """
    Image stretch for display. Handles edge cases.
    """
    if img is None:
        return None
    arr = np.asarray(img, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, clip_percentiles)
    if hi - lo <= 0:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    clipped = np.clip(arr, lo, hi)
    normed = (clipped - lo) / (hi - lo + 1e-12)
    if stretch == 'asinh':
        return np.arcsinh(scale * normed) / np.arcsinh(scale)
    elif stretch == 'sqrt':
        return np.sqrt(normed)
    elif stretch == 'log':
        return np.log1p(normed * (np.e - 1.0))
    else:
        return normed

def create_comprehensive_analysis_figure(obs_img: np.ndarray,
                                         kappa_map: np.ndarray,
                                         lens_params: Dict[str, float],
                                         radial_profile_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                         mass_info_physical: Dict,
                                         mass_info_norm: Dict,
                                         halo_info: Dict,
                                         output_path: str,
                                         deflection_field: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                         model_pred: Optional[np.ndarray] = None,
                                         source_plane: Optional[np.ndarray] = None,
                                         residual_map: Optional[np.ndarray] = None,
                                         chi2_map: Optional[np.ndarray] = None,
                                         save_individual_panels: bool = False,
                                         include_3d: bool = False,
                                         interactive: bool = False,
                                         zoom_box: Optional[Tuple[float, float, float]] = None,
                                         photometry: Optional[np.ndarray] = None,
                                         bootstrap_profiles: Optional[Dict] = None):
    if interactive:
        plt.ion()
    else:
        plt.ioff()

    obs_img = ensure_finite_arr(obs_img, fill=0.0)
    kappa_map = ensure_finite_arr(kappa_map, fill=0.0)
    H, W = kappa_map.shape

    # Basic derived numbers
    try:
        pixel_scale_used = float(mass_info_norm.get('pixel_scale_used_arcsec_per_pix', 0.05))
    except Exception:
        pixel_scale_used = 0.05

    analyzer = MassProfileAnalyzer(pixel_scale=pixel_scale_used,
                                   cosmology=Planck18,
                                   z_lens=mass_info_physical.get('z_lens', 0.5),
                                   z_source=mass_info_physical.get('z_source', 2.0))

    # Einstein radius (arcsec) and sigma_v (km/s)
    try:
        einstein_arcsec = float(mass_info_norm.get('einstein_radius_arcsec', 
                          analyzer.einstein_radius_arcsec({'b': torch.tensor(lens_params.get('b', 0.0))}, (H, W))))
    except Exception:
        try:
            einstein_arcsec = analyzer.einstein_radius_arcsec({'b': torch.tensor(lens_params.get('b', 0.0))}, (H, W))
        except Exception:
            einstein_arcsec = 0.0
    
    try:
        sigma_v_kms = float(mass_info_physical.get('sigma_v_kms', 
                          analyzer.sigma_v_from_thetaE(einstein_arcsec, (H, W))))
    except Exception:
        sigma_v_kms = None

    # Mass within Einstein radius (physical)
    try:
        mass_einstein = float(mass_info_physical.get('mass_Msun', 
                            analyzer.mass_within_radius(kappa_map, einstein_arcsec, 
                            (lens_params.get('x0', 0.0), lens_params.get('y0', 0.0))).get('mass_Msun', 0.0)))
    except Exception:
        mass_einstein = 0.0
    fig = plt.figure(figsize=(28, 18), dpi=100)
    

    gs = gridspec.GridSpec(4, 6, figure=fig, 
                          height_ratios=[1.0, 1, 1, 1], 
                          hspace=1,  
                          wspace=1)   
    
    # Track which axes have colorbars to avoid overlap
    axes_with_cbars = []

    # Row 0: Main images
    # Observed image
    ax_obs = fig.add_subplot(gs[0, 0])
    try:
        obs_display = stretch_image(obs_img, stretch='asinh', scale=6.0)
        im_obs = ax_obs.imshow(obs_display, origin='lower', cmap='gray', aspect='auto')
        ax_obs.set_title('Observed Lensed Image', fontsize=11, weight='semibold', pad=8)
        ax_obs.axis('off')
        
        # Add Einstein radius circle
        try:
            x0_pix = (lens_params['x0'] + 1.0) * W / 2.0
            y0_pix = (lens_params['y0'] + 1.0) * H / 2.0
            theta_e_pix = einstein_arcsec / pixel_scale_used
            circle = plt.Circle((x0_pix, y0_pix), theta_e_pix, fill=False, 
                               color='cyan', linewidth=2, linestyle='--', alpha=0.8)
            ax_obs.add_patch(circle)
        except Exception:
            pass
        
        # Add colorbar with adjusted position
        cax_obs = fig.add_axes([ax_obs.get_position().x1 + 0.005, 
                               ax_obs.get_position().y0, 
                               0.01, 
                               ax_obs.get_position().height])
        plt.colorbar(im_obs, cax=cax_obs)
        axes_with_cbars.append(cax_obs)
    except Exception as e:
        ax_obs.text(0.5, 0.5, f"Observed image\nfailed", 
                   ha='center', va='center', transform=ax_obs.transAxes)

    # Model predicted image
    ax_model = fig.add_subplot(gs[0, 1])
    try:
        if model_pred is not None:
            model_disp = stretch_image(model_pred, stretch='asinh', scale=6.0)
            im_model = ax_model.imshow(model_disp, origin='lower', cmap='gray', aspect='auto')
            ax_model.set_title('Model Prediction', fontsize=11, pad=8)
            ax_model.axis('off')
            
            # Add colorbar
            cax_model = fig.add_axes([ax_model.get_position().x1 + 0.005,
                                     ax_model.get_position().y0,
                                     0.01,
                                     ax_model.get_position().height])
            plt.colorbar(im_model, cax=cax_model)
            axes_with_cbars.append(cax_model)
            
            if residual_map is not None:
                residual_snr = np.sqrt(np.nansum(residual_map**2)) / np.nanstd(obs_img) if np.nanstd(obs_img) > 0 else np.nan
                ax_model.text(0.02, 0.98, f'SNR: {residual_snr:.1f}', 
                            transform=ax_model.transAxes, fontsize=9,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax_model.text(0.5, 0.5, 'No model\nprediction', 
                         ha='center', va='center', transform=ax_model.transAxes)
            ax_model.axis('off')
    except Exception:
        ax_model.text(0.1, 0.5, "Model panel failed", transform=ax_model.transAxes)

    # Source plane reconstruction
    ax_source = fig.add_subplot(gs[0, 2])
    try:
        if source_plane is not None:
            src_disp = stretch_image(source_plane, stretch='asinh', scale=6.0)
            im_src = ax_source.imshow(src_disp, origin='lower', cmap='gray', aspect='auto')
            ax_source.set_title('Source Plane', fontsize=11, weight='semibold', pad=8)
            ax_source.axis('off')
            
            # Add colorbar
            cax_src = fig.add_axes([ax_source.get_position().x1 + 0.005,
                                   ax_source.get_position().y0,
                                   0.01,
                                   ax_source.get_position().height])
            plt.colorbar(im_src, cax=cax_src)
            axes_with_cbars.append(cax_src)
            
            src_stats = f"Max: {np.nanmax(source_plane):.2f}\nMean: {np.nanmean(source_plane):.3f}"
            ax_source.text(0.02, 0.98, src_stats, transform=ax_source.transAxes, fontsize=8,
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax_source.text(0.5, 0.5, 'No source\nplane', 
                          ha='center', va='center', transform=ax_source.transAxes)
            ax_source.axis('off')
    except Exception:
        ax_source.text(0.1, 0.5, "Source plane failed", transform=ax_source.transAxes)

    # Kappa map (spanning 3 columns)
    ax_kappa = fig.add_subplot(gs[0, 3:6])
    try:
        vmin_k = np.nanpercentile(kappa_map, 1) if np.isfinite(kappa_map).any() else 0
        vmax_k = np.nanpercentile(kappa_map, 99) if np.isfinite(kappa_map).any() else 1
        im_kappa = ax_kappa.imshow(kappa_map, origin='lower', cmap='inferno', 
                                  aspect='auto', vmin=vmin_k, vmax=vmax_k)
        ax_kappa.set_title('Convergence (κ)', fontsize=12, weight='semibold', pad=8)
        ax_kappa.axis('off')
        
        # Add critical curve (κ=1)
        try:
            if np.isfinite(kappa_map).any():
                ax_kappa.contour(kappa_map, levels=[1.0], colors='cyan', 
                               linewidths=2.0, origin='lower', linestyles='--', alpha=0.8)
        except Exception:
            pass
        
        # Add colorbar
        cax_kappa = fig.add_axes([ax_kappa.get_position().x1 + 0.005,
                                 ax_kappa.get_position().y0,
                                 0.01,
                                 ax_kappa.get_position().height])
        cbar = plt.colorbar(im_kappa, cax=cax_kappa)
        cbar.set_label('κ', fontsize=10)
        axes_with_cbars.append(cax_kappa)
    except Exception as e:
        ax_kappa.text(0.1, 0.5, f"κ panel failed", transform=ax_kappa.transAxes)

    # Row 1: Residuals and statistics
    # Residual map
    ax_res = fig.add_subplot(gs[1, 0:2])
    try:
        if residual_map is not None:
            vres = np.nanpercentile(np.abs(residual_map), 99.5) if np.isfinite(residual_map).any() else 1.0
            im_res = ax_res.imshow(residual_map, origin='lower', cmap='RdBu_r', 
                                  aspect='auto', vmin=-vres, vmax=vres)
            ax_res.set_title('Residual (Obs - Model)', fontsize=11, pad=8)
            ax_res.axis('off')
            
            rms_residual = np.sqrt(np.nanmean(residual_map**2))
            ax_res.text(0.02, 0.98, f'RMS: {rms_residual:.3f}', 
                       transform=ax_res.transAxes, fontsize=9,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add colorbar
            cax_res = fig.add_axes([ax_res.get_position().x1 + 0.005,
                                   ax_res.get_position().y0,
                                   0.01,
                                   ax_res.get_position().height])
            plt.colorbar(im_res, cax=cax_res)
            axes_with_cbars.append(cax_res)
        else:
            ax_res.text(0.5, 0.5, 'No residual\navailable', 
                       ha='center', va='center', transform=ax_res.transAxes)
            ax_res.axis('off')
    except Exception:
        ax_res.text(0.1, 0.5, "Residual failed", transform=ax_res.transAxes)

    # κ histogram
    ax_hist = fig.add_subplot(gs[1, 2])
    try:
        kdat = kappa_map.ravel()
        kdat = kdat[np.isfinite(kdat)]
        if kdat.size > 0:
            ax_hist.hist(kdat, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            ax_hist.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='κ=1 (critical)')
            ax_hist.legend(fontsize=8, loc='upper right')
            ax_hist.set_xlabel('κ', fontsize=9)
            ax_hist.set_ylabel('Density', fontsize=9)
            ax_hist.set_title('κ Distribution', fontsize=11, pad=8)
            ax_hist.tick_params(labelsize=8)
            ax_hist.grid(alpha=0.2)
        else:
            ax_hist.text(0.5, 0.5, 'No κ data', ha='center', va='center', transform=ax_hist.transAxes)
    except Exception:
        ax_hist.text(0.1, 0.5, "Histogram failed", transform=ax_hist.transAxes)

    # Chi² map
    ax_chi2 = fig.add_subplot(gs[1, 3])
    try:
        if chi2_map is not None:
            im_chi2 = ax_chi2.imshow(chi2_map, origin='lower', cmap='hot', aspect='auto')
            ax_chi2.set_title('χ² Map', fontsize=11, pad=8)
            ax_chi2.axis('off')
            
            total_chi2 = np.nansum(chi2_map)
            ax_chi2.text(0.02, 0.98, f'χ²: {total_chi2:.0f}', 
                        transform=ax_chi2.transAxes, fontsize=9,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add colorbar
            cax_chi2 = fig.add_axes([ax_chi2.get_position().x1 + 0.005,
                                    ax_chi2.get_position().y0,
                                    0.01,
                                    ax_chi2.get_position().height])
            plt.colorbar(im_chi2, cax=cax_chi2)
            axes_with_cbars.append(cax_chi2)
        else:
            ax_chi2.text(0.5, 0.5, 'No χ² map', 
                        ha='center', va='center', transform=ax_chi2.transAxes)
            ax_chi2.axis('off')
    except Exception:
        ax_chi2.text(0.1, 0.5, "χ² panel failed", transform=ax_chi2.transAxes)

    # Parameter table
    ax_table = fig.add_subplot(gs[1, 4:6])
    ax_table.axis('off')
    try:
        # Prepare parameter text
        lines = []
        lines.append("LENS PARAMETERS")
        lines.append("=" * 40)
        
        param_order = ['b', 'q', 'phi', 'x0', 'y0', 'gamma', 'kappa_s', 'rs']
        for k in param_order:
            if k in lens_params:
                val = lens_params[k]
                if isinstance(val, (np.ndarray, torch.Tensor)):
                    val = float(val)
                lines.append(f"{k:10s}: {val:8.4f}")
        
        lines.append("\nPHYSICAL QUANTITIES")
        lines.append("-" * 40)
        lines.append(f"θ_E         : {einstein_arcsec:8.3f} arcsec")
        if sigma_v_kms is not None:
            lines.append(f"σ_v (SIS)   : {sigma_v_kms:8.0f} km/s")
        lines.append(f"M(θ_E)      : {mass_einstein:8.2e} M⊙")
        lines.append(f"Pixel scale : {pixel_scale_used:8.4f} \"/pix")
        
        # Add source plane info if available
        if source_plane is not None:
            lines.append("\nSOURCE PLANE")
            lines.append("-" * 40)
            lines.append(f"Max flux  : {np.nanmax(source_plane):8.3f}")
            lines.append(f"Mean flux : {np.nanmean(source_plane):8.3f}")
        
        # Create table text
        table_text = "\n".join(lines)
        ax_table.text(0.02, 0.98, table_text, fontsize=9, va='top', 
                     family='monospace', transform=ax_table.transAxes,
                     bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    except Exception:
        ax_table.text(0.1, 0.5, "Params panel failed", transform=ax_table.transAxes)

    # Row 2: Radial profiles
    # Radial κ profile
    ax_rad = fig.add_subplot(gs[2, 0:2])
    try:
        radii, prof, prof_std, prof_counts = radial_profile_data
        if radii.size > 0:
            ax_rad.plot(radii, prof, '-', lw=2, color='darkblue', label='κ(r)', zorder=3)
            if prof_std.size == prof.size and np.any(prof_std > 0):
                ax_rad.fill_between(radii, prof - prof_std, prof + prof_std, 
                                   alpha=0.25, color='blue', label='±1σ', zorder=2)
            
            if bootstrap_profiles is not None:
                try:
                    mean_b = bootstrap_profiles.get('mean', None)
                    ci_low = bootstrap_profiles.get('ci_low', None)
                    ci_high = bootstrap_profiles.get('ci_high', None)
                    if mean_b is not None:
                        ax_rad.plot(radii, mean_b, '--', color='orange', 
                                   label='bootstrap mean', lw=1.5, zorder=4)
                        ax_rad.fill_between(radii, ci_low, ci_high, color='gray', 
                                           alpha=0.2, label='95% CI', zorder=1)
                except Exception:
                    pass
            
            if einstein_arcsec > 0:
                ax_rad.axvline(x=einstein_arcsec, color='red', linestyle='--', 
                              alpha=0.7, label=f'θ_E={einstein_arcsec:.2f}"', zorder=5)
            
            ax_rad.set_xlabel('Radius (arcsec)', fontsize=10)
            ax_rad.set_ylabel('κ', fontsize=10)
            ax_rad.set_title('Radial κ Profile', fontsize=11, pad=8)
            ax_rad.grid(alpha=0.3, linestyle='--')
            ax_rad.legend(fontsize=8, loc='upper right')
            ax_rad.tick_params(labelsize=9)
        else:
            ax_rad.text(0.5, 0.5, 'No radial data', 
                       ha='center', va='center', transform=ax_rad.transAxes)
    except Exception:
        ax_rad.text(0.1, 0.5, "Radial profile failed", transform=ax_rad.transAxes)

    # Cumulative mass profile
    ax_cum = fig.add_subplot(gs[2, 2:4])
    try:
        max_radius = min(H, W) * 0.4 * pixel_scale_used
        radii_arc = np.linspace(pixel_scale_used, max_radius, 40)
        massvals = []
        for r in radii_arc:
            try:
                m = analyzer.mass_within_radius(kappa_map, r, 
                                               (lens_params.get('x0', 0.0), 
                                                lens_params.get('y0', 0.0)))
                massvals.append(m.get('mass_Msun', 0.0))
            except Exception:
                massvals.append(0.0)
        
        ax_cum.plot(radii_arc, np.array(massvals), lw=2.5, color='darkgreen', 
                   label='M(<r)', zorder=3)
        
        if einstein_arcsec > 0 and einstein_arcsec < max_radius:
            ax_cum.axvline(x=einstein_arcsec, color='red', linestyle='--', 
                          alpha=0.7, label=f'θ_E={einstein_arcsec:.2f}"', zorder=4)
            idx = np.argmin(np.abs(radii_arc - einstein_arcsec))
            if idx < len(massvals):
                mass_at_thetaE = massvals[idx]
                ax_cum.plot(einstein_arcsec, mass_at_thetaE, 'ro', markersize=8, 
                           label=f'M(θ_E)={mass_at_thetaE:.2e}', zorder=5)
        
        ax_cum.set_xlabel('Radius (arcsec)', fontsize=10)
        ax_cum.set_ylabel('M(<r) [M⊙]', fontsize=10)
        ax_cum.set_title('Cumulative Mass Profile', fontsize=11, pad=8)
        ax_cum.grid(alpha=0.3, linestyle='--')
        ax_cum.set_yscale('log')
        ax_cum.legend(fontsize=8, loc='upper left')
        ax_cum.tick_params(labelsize=9)
    except Exception:
        ax_cum.text(0.1, 0.5, "Cumulative mass failed", transform=ax_cum.transAxes)

    # Shear maps (γ1, γ2)
    ax_g1 = fig.add_subplot(gs[2, 4])
    ax_g2 = fig.add_subplot(gs[2, 5])
    try:
        if deflection_field is not None:
            ax_map, ay_map = deflection_field
            d_ax_dx = np.gradient(ax_map, axis=1)
            d_ax_dy = np.gradient(ax_map, axis=0)
            d_ay_dx = np.gradient(ay_map, axis=1)
            d_ay_dy = np.gradient(ay_map, axis=0)
            gamma1 = 0.5 * (d_ax_dx - d_ay_dy)
            gamma2 = 0.5 * (d_ax_dy + d_ay_dx)
            
            # γ1 map
            vmax_g = max(np.nanmax(np.abs(gamma1)), np.nanmax(np.abs(gamma2)), 0.1)
            im_g1 = ax_g1.imshow(gamma1, origin='lower', cmap='coolwarm', 
                                aspect='auto', vmin=-vmax_g, vmax=vmax_g)
            ax_g1.set_title('γ₁', fontsize=11, pad=8)
            ax_g1.axis('off')
            
            # γ2 map
            im_g2 = ax_g2.imshow(gamma2, origin='lower', cmap='coolwarm', 
                                aspect='auto', vmin=-vmax_g, vmax=vmax_g)
            ax_g2.set_title('γ₂', fontsize=11, pad=8)
            ax_g2.axis('off')
            
            # Add colorbars
            cax_g1 = fig.add_axes([ax_g1.get_position().x1 + 0.005,
                                  ax_g1.get_position().y0,
                                  0.008,
                                  ax_g1.get_position().height])
            plt.colorbar(im_g1, cax=cax_g1)
            axes_with_cbars.append(cax_g1)
            
            cax_g2 = fig.add_axes([ax_g2.get_position().x1 + 0.005,
                                  ax_g2.get_position().y0,
                                  0.008,
                                  ax_g2.get_position().height])
            plt.colorbar(im_g2, cax=cax_g2)
            axes_with_cbars.append(cax_g2)
        else:
            ax_g1.text(0.5, 0.5, 'No deflection\nfield', 
                      ha='center', va='center', transform=ax_g1.transAxes)
            ax_g1.axis('off')
            ax_g2.axis('off')
    except Exception:
        ax_g1.text(0.1, 0.5, "Shear maps failed", transform=ax_g1.transAxes)
        ax_g2.axis('off')

    # Row 3: Additional analyses
    # Azimuthal κ profile
    ax_az = fig.add_subplot(gs[3, 0:2])
    try:
        radius_arcsec = einstein_arcsec if einstein_arcsec > 0 else (min(H, W) * 0.2 * pixel_scale_used)
        r_pix = max(2, int(radius_arcsec / pixel_scale_used))
        
        Yg, Xg = np.indices(kappa_map.shape)
        x0_pix = (lens_params.get('x0', 0.0) + 1.0) * W / 2.0
        y0_pix = (lens_params.get('y0', 0.0) + 1.0) * H / 2.0
        
        rmap = np.sqrt((Xg - x0_pix)**2 + (Yg - y0_pix)**2)
        ann_mask = (np.abs(rmap - r_pix) <= max(1, int(0.1 * r_pix)))
        
        thetas = np.arctan2(Yg[ann_mask] - y0_pix, Xg[ann_mask] - x0_pix)
        k_ann = kappa_map[ann_mask]
        
        if thetas.size > 10:
            order = np.argsort(thetas)
            thetas_s = thetas[order]
            k_s = k_ann[order]
            
            ax_az.scatter(np.degrees(thetas_s), k_s, s=10, alpha=0.6, color='blue', edgecolor='black')
            
            try:
                # Fit sinusoidal variation
                def sin_fit(theta, A, phi0, offset):
                    return A * np.cos(2*(theta - phi0)) + offset
                
                p0 = [0.1 * np.std(k_s), 0.0, np.mean(k_s)]
                popt, _ = curve_fit(sin_fit, thetas_s, k_s, p0=p0, maxfev=5000)
                fit_curve = sin_fit(thetas_s, *popt)
                ax_az.plot(np.degrees(thetas_s), fit_curve, 'r-', lw=2, 
                          label=f'A={abs(popt[0]):.3f}, φ={np.degrees(popt[1]):.1f}°')
                ax_az.legend(fontsize=8)
            except Exception:
                pass
            
            ax_az.set_xlabel('Azimuth (deg)', fontsize=10)
            ax_az.set_ylabel('κ', fontsize=10)
            ax_az.set_title(f'Azimuthal κ @ r ≈ {radius_arcsec:.2f}"', fontsize=11, pad=8)
            ax_az.grid(alpha=0.3, linestyle='--')
            ax_az.tick_params(labelsize=9)
        else:
            ax_az.text(0.5, 0.5, 'Insufficient data\nfor azimuthal profile', 
                      ha='center', va='center', transform=ax_az.transAxes)
    except Exception:
        ax_az.text(0.1, 0.5, "Azimuthal failed", transform=ax_az.transAxes)

    # Mass to light or source power spectrum
    ax_ml = fig.add_subplot(gs[3, 2:4])
    try:
        if photometry is not None and photometry.size == kappa_map.size:
            phot = photometry.reshape(kappa_map.shape)
            try:
                sigma_crit_kg_m2, _ = analyzer.sigma_crit()
                D_l, _, _ = analyzer.angular_diameter_distances()
                pix_rad = (pixel_scale_used * u.arcsec).to(u.rad).value
                pix_area = (D_l.value * pix_rad) ** 2
                mass_map_kg = kappa_map * sigma_crit_kg_m2.value * pix_area
                mass_map_msun = mass_map_kg / M_sun.to(u.kg).value
                with np.errstate(divide='ignore', invalid='ignore'):
                    ML_map = mass_map_msun / (phot + 1e-12)
                
                im_ml = ax_ml.imshow(np.clip(ML_map, 0, np.nanpercentile(ML_map[np.isfinite(ML_map)], 99.5)), 
                                    origin='lower', cmap='viridis', aspect='auto')
                ax_ml.set_title('Mass-to-Light Ratio', fontsize=11, pad=8)
                ax_ml.axis('off')
                
                cax_ml = fig.add_axes([ax_ml.get_position().x1 + 0.005,
                                      ax_ml.get_position().y0,
                                      0.01,
                                      ax_ml.get_position().height])
                cbar_ml = plt.colorbar(im_ml, cax=cax_ml)
                cbar_ml.set_label('M/L [M⊙/L⊙]', fontsize=9)
                axes_with_cbars.append(cax_ml)
            except Exception:
                ax_ml.text(0.5, 0.5, 'M/L calc failed', 
                          ha='center', va='center', transform=ax_ml.transAxes)
                ax_ml.axis('off')
        elif source_plane is not None:
            # Show source plane power spectrum
            try:
                src_fft = np.fft.fftshift(np.fft.fft2(source_plane))
                src_power = np.log1p(np.abs(src_fft)**2)
                im_src_pow = ax_ml.imshow(src_power, origin='lower', cmap='viridis', 
                                         aspect='auto')
                ax_ml.set_title('Source Power Spectrum', fontsize=11, pad=8)
                ax_ml.axis('off')
                
                cax_pow = fig.add_axes([ax_ml.get_position().x1 + 0.005,
                                       ax_ml.get_position().y0,
                                       0.01,
                                       ax_ml.get_position().height])
                plt.colorbar(im_src_pow, cax=cax_pow)
                axes_with_cbars.append(cax_pow)
            except Exception:
                ax_ml.text(0.5, 0.5, 'Source PS\nunavailable', 
                          ha='center', va='center', transform=ax_ml.transAxes)
                ax_ml.axis('off')
        else:
            ax_ml.text(0.5, 0.5, 'No photometry\nfor M/L', 
                      ha='center', va='center', transform=ax_ml.transAxes)
            ax_ml.axis('off')
    except Exception:
        ax_ml.text(0.1, 0.5, "M/L panel failed", transform=ax_ml.transAxes)

    # 3D surface or metrics
    ax_3d = fig.add_subplot(gs[3, 4:6], projection='3d') if include_3d else fig.add_subplot(gs[3, 4:6])
    try:
        if include_3d:
            step = max(1, min(H // 40, W // 40))
            X = np.arange(0, W, step)
            Y = np.arange(0, H, step)
            Xg, Yg = np.meshgrid(X, Y)
            Z = kappa_map[::step, ::step]
            
            ax_3d.plot_surface(Xg, Yg, Z, cmap='inferno', 
                              linewidth=0.5, antialiased=True, alpha=0.8)
            ax_3d.set_title('κ 3D Surface', fontsize=11, pad=8)
            ax_3d.set_xlabel('X (pix)', fontsize=9)
            ax_3d.set_ylabel('Y (pix)', fontsize=9)
            ax_3d.set_zlabel('κ', fontsize=9)
            ax_3d.view_init(elev=30, azim=45)
            ax_3d.tick_params(labelsize=8)
        else:
            # Show metrics summary
            ax_3d.axis('off')
            metrics_text = []
            metrics_text.append("RECONSTRUCTION METRICS")
            metrics_text.append("=" * 40)
            
            if residual_map is not None:
                rms_residual = np.sqrt(np.nanmean(residual_map**2))
                snr = np.sqrt(np.nansum(obs_img**2)) / np.sqrt(np.nansum(residual_map**2)) if np.nansum(residual_map**2) > 0 else np.inf
                metrics_text.append(f"RMS residual : {rms_residual:.4f}")
                metrics_text.append(f"SNR          : {snr:.2f}")
            
            metrics_text.append(f"θ_E          : {einstein_arcsec:.3f}\"")
            metrics_text.append(f"M(θ_E)       : {mass_einstein:.2e} M⊙")
            
            if sigma_v_kms is not None:
                metrics_text.append(f"σ_v (SIS)    : {sigma_v_kms:.0f} km/s")
            
            metrics_text.append(f"Pixel scale  : {pixel_scale_used:.4f}\"/pix")
            metrics_text.append(f"z_lens       : {analyzer.z_lens}")
            metrics_text.append(f"z_source     : {analyzer.z_source}")
            
            if source_plane is not None:
                metrics_text.append("\nSOURCE PLANE")
                metrics_text.append("-" * 40)
                metrics_text.append(f"Max flux     : {np.nanmax(source_plane):.3f}")
                metrics_text.append(f"Mean flux    : {np.nanmean(source_plane):.3f}")
            
            ax_3d.text(0.02, 0.98, "\n".join(metrics_text), fontsize=9, va='top', 
                      family='monospace', transform=ax_3d.transAxes,
                      bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    except Exception:
        ax_3d.text(0.1, 0.5, "3D/metrics panel failed", transform=ax_3d.transAxes)

    # Main title and footer
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    fig.suptitle(f"Gravitational Lens Mass Profile Analysis with Source Reconstruction", 
                fontsize=16, weight='bold', y=0.99)
    
    footer_text = (f"θ_E={einstein_arcsec:.3f}\" | M(θ_E)={mass_einstein:.2e} M⊙ | "
                  f"Scale={pixel_scale_used:.4f}\"/pix | z_lens={analyzer.z_lens}, z_source={analyzer.z_source}")
    fig.text(0.5, 0.01, footer_text, fontsize=10, ha='center', color='darkblue')

    # Save with tight layout
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        logger.info(f"Saved comprehensive analysis to: {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save figure: {e}")

    # Save individual panels if requested
    if save_individual_panels:
        try:
            base, ext = os.path.splitext(output_path)
            for i, ax in enumerate(fig.axes):
                if ax in axes_with_cbars:
                    continue  # Skip colorbar axes
                fn = f"{base}_panel{i+1:02d}.png"
                try:
                    extent = ax.get_tightbbox(fig.canvas.get_renderer())
                    if extent is not None:
                        fig.savefig(fn, bbox_inches=extent.transformed(fig.dpi_scale_trans.inverted()))
                except Exception:
                    pass
            logger.info("Saved individual panels.")
        except Exception as e:
            logger.warning(f"Saving individual panels failed: {e}")

    if not interactive:
        plt.close(fig)
    else:
        plt.show()

    return fig

# I/O helpers for loading FITS data

def load_fits_pair(fits_path, source_hdu_names=None):
    """
    Return (obs_tensor, src_tensor) where tensors are normalized torch float tensors
    with shape (1,1,H,W). Now also looks for source plane reconstruction.
    
    Parameters:
    -----------
    fits_path : str
        Path to FITS file
    source_hdu_names : list, optional
        List of HDU names to try for source plane (default: ['RECON', 'SRC', 'SOURCE', 'SOURCE_PLANE'])
    
    Returns:
    --------
    obs_tensor : torch.Tensor
        Observed lensed image
    src_tensor : torch.Tensor or None
        Source plane reconstruction if found
    """
    if source_hdu_names is None:
        source_hdu_names = ['RECON', 'SRC', 'SOURCE', 'SOURCE_PLANE']
    
    hdul = fits.open(fits_path, memmap=False)
    lensed = None
    source = None
    
    for i, h in enumerate(hdul):
        name = (getattr(h, 'name', '') or h.header.get('EXTNAME', '')).upper()
        data = h.data
        if data is None:
            continue
        try:
            d = np.array(data).astype(np.float32)
        except Exception:
            continue
        
        # Check for source plane reconstruction
        if name in source_hdu_names:
            if source is None:  # Take first source plane found
                source = d
                logger.debug(f"Found source plane in HDU '{name}'")
        
        # Check for observed lensed image
        elif name in ('LENSED', 'OBS', 'OBSERVED', 'DATA'):
            if lensed is None:  # Take first lensed image found
                lensed = d
    
    # Fallback: if no explicit lensed HDU found, check primary and secondary
    if lensed is None and hdul[0].data is not None:
        try:
            # Check if primary is likely lensed (not source)
            primary_data = np.array(hdul[0].data).astype(np.float32)
            # Simple heuristic: lensed images typically have more complex structure
            # but for now just use it as lensed
            lensed = primary_data
            logger.debug("Using primary HDU as lensed image")
        except Exception:
            pass
    
    # Fallback for source plane
    if source is None and len(hdul) > 1 and hdul[1].data is not None:
        try:
            # Check if secondary is source plane (not already used)
            sec_data = np.array(hdul[1].data).astype(np.float32)
            if lensed is not None and not np.array_equal(lensed, sec_data):
                source = sec_data
                logger.debug("Using secondary HDU as source plane")
        except Exception:
            pass
    
    hdul.close()
    
    if lensed is None:
        raise RuntimeError(f"{fits_path} missing LENSED/OBS/DATA HDU.")
    
    # Clean and normalize lensed image
    lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    while lensed.ndim > 2:
        lensed = np.squeeze(lensed, axis=0)
    if lensed.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {lensed.shape} from {fits_path}")
    
    # Clean and normalize source plane if available
    if source is not None:
        source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        while source.ndim > 2:
            source = np.squeeze(source, axis=0)
        if source.ndim != 2:
            logger.warning(f"Source plane has shape {source.shape}, ignoring")
            source = None
        else:
            # Ensure source plane has same dimensions as lensed (resize if needed)
            if source.shape != lensed.shape:
                from scipy.ndimage import zoom
                try:
                    zoom_factors = (lensed.shape[0] / source.shape[0], 
                                   lensed.shape[1] / source.shape[1])
                    source = zoom(source, zoom_factors, order=1)
                    logger.info(f"Resized source plane from {source.shape} to {lensed.shape}")
                except Exception as e:
                    logger.warning(f"Could not resize source plane: {e}")
                    source = None
    
    # Normalize by robust maximum
    eps = 1e-12
    m = max(np.nanmax(lensed) if lensed.size else eps, 
            np.nanmax(source) if source is not None and source.size else eps, eps)
    
    lensed = (lensed / m).astype(np.float32)
    obs_tensor = torch.from_numpy(lensed).unsqueeze(0).unsqueeze(0).float()
    
    src_tensor = None
    if source is not None:
        source = (source / m).astype(np.float32)
        src_tensor = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float()
    
    logger.info(f"Loaded FITS: lensed shape {lensed.shape}, source {'found' if src_tensor is not None else 'not found'}")
    return obs_tensor, src_tensor

# Analysis helpers: single lens analysis

def compute_bootstrap_radial(kappa_map: np.ndarray, center_xy_norm: Tuple[float, float], analyzer: MassProfileAnalyzer, n_boot: int = 200, n_bins: int = 40, ci=0.95):
    """
    Bootstrap resampling of pixels within annuli to compute profile uncertainty.
    Returns dict: mean, ci_low, ci_high arrays (same length as profile).
    """
    radii, prof, prof_std, counts = analyzer.radial_profile(kappa_map, center_xy_norm, n_bins=n_bins)
    if radii.size == 0:
        return None
    H, W = kappa_map.shape
    x0_pix = (center_xy_norm[0] + 1.0) * W / 2.0
    y0_pix = (center_xy_norm[1] + 1.0) * H / 2.0
    # build per-bin pixel lists
    Yg, Xg = np.indices(kappa_map.shape)
    rmap = np.sqrt((Xg - x0_pix)**2 + (Yg - y0_pix)**2)
    r_max = rmap.max()
    bins = np.linspace(0.0, r_max, n_bins+1)
    bin_pixel_vals = []
    for i in range(n_bins):
        mask = (rmap >= bins[i]) & (rmap < bins[i+1])
        vals = kappa_map[mask]
        vals = vals[np.isfinite(vals)]
        bin_pixel_vals.append(vals if vals.size > 0 else np.array([0.0]))
    # bootstrap
    boot_profiles = []
    rng = np.random.RandomState(42)
    for b in range(n_boot):
        prof_b = []
        for vals in bin_pixel_vals:
            # resample with replacement
            idx = rng.randint(0, vals.size, size=vals.size)
            prof_b.append(np.mean(vals[idx]) if vals.size > 0 else 0.0)
        boot_profiles.append(prof_b)
    boot = np.array(boot_profiles)  # shape (n_boot, n_bins)
    mean = np.mean(boot, axis=0)
    low = np.percentile(boot, (1.0-ci)/2*100, axis=0)
    high = np.percentile(boot, (1.0+ci)/2*100, axis=0)
    # return only bins with counts > 0 (consistent with analyzer.radial_profile)
    valid = counts > 0
    return {'radii': radii, 'mean': mean[valid], 'ci_low': low[valid], 'ci_high': high[valid]}

def compute_residual_metrics(obs_img, model_pred, source_plane=None):
    """
    Compute various residual metrics and χ² map.
    
    Returns:
    --------
    residual_map : np.ndarray
        Observed - Model
    chi2_map : np.ndarray
        χ² per pixel (assuming Poisson + read noise)
    metrics : dict
        Various quality metrics
    """
    residual = obs_img - model_pred
    
    # Simple χ² calculation (assuming Gaussian errors)
    noise_estimate = np.sqrt(np.abs(model_pred) + 0.1**2)  # Poisson + read noise
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_map = (residual / noise_estimate)**2
    chi2_map = np.nan_to_num(chi2_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    metrics = {
        'rms_residual': np.sqrt(np.nanmean(residual**2)),
        'max_residual': np.nanmax(np.abs(residual)),
        'total_chi2': np.nansum(chi2_map),
        'reduced_chi2': np.nansum(chi2_map) / (obs_img.size - 10),  # Approximate DOF
        'snr': np.sqrt(np.nansum(obs_img**2)) / np.sqrt(np.nansum(residual**2)) if np.nansum(residual**2) > 0 else np.inf
    }
    
    # Add source plane metrics if available
    if source_plane is not None:
        metrics.update({
            'source_max': np.nanmax(source_plane),
            'source_mean': np.nanmean(source_plane),
            'source_total': np.nansum(source_plane),
            'source_std': np.nanstd(source_plane)
        })
    
    return residual, chi2_map, metrics

def analyze_single_lens_worker(forward_model_path: str, fits_path: str, output_dir: str, lens_id: str, args):
    """
    Worker invoked in parallel processes: loads forward_model afresh to avoid pickling issues.
    Returns summary dict (or raises).
    NOTE: this function mirrors analyze_single_lens but is self-contained for MP.
    """
    # Load the forward model (using the complete class definition from this script)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Instantiate the forward model
    forward_model = ConditionalPhysicalForward(
        kernel_size=args.kernel_size,
        enforce_nonneg=True,
        init_fwhm=3.0,
        init_beta=4.5,
        pixel_scale=0.05,  # Fixed for the model
        learn_residual_psf=args.learn_residual_psf,
        residual_scale=1e-2,
        use_fft_conv=True,
        use_nfw=args.use_nfw,
        use_subhalos=args.use_subhalos,
        n_subhalos=args.n_subhalos
    )
    
    try:
        st = torch.load(forward_model_path, map_location='cpu')
        # try common key patterns
        if isinstance(st, dict) and ('forward_state' in st or 'state_dict' in st):
            sd = st.get('forward_state', st.get('state_dict', st))
            forward_model.load_state_dict(sd)
        else:
            forward_model.load_state_dict(st)
        forward_model.to(device)
        forward_model.eval()
    except Exception as e:
        logger.error(f"[{lens_id}] Failed to load forward model in worker: {e}")
        raise

    try:
        obs_tensor, src_tensor = load_fits_pair(fits_path, 
                                               source_hdu_names=getattr(args, 'source_hdu_names', 
                                                                       ['RECON', 'SRC', 'SOURCE', 'SOURCE_PLANE']))
    except Exception as e:
        logger.error(f"[{lens_id}] Failed to load FITS {fits_path}: {e}")
        raise

    return analyze_single_lens(forward_model, obs_tensor, output_dir, lens_id, args=args, src_tensor=src_tensor)

def analyze_single_lens(forward_model, obs_tensor, output_dir, lens_id="lens_001", args=None, src_tensor=None):
    """
    Analyze one lens with source plane reconstruction support:
      - compute lens params and kappa map
      - generate model-predicted image from source plane (if available)
      - compute residuals and χ²
      - compute cosmology-based masses and σ_v
      - compute radial profile and bootstrap uncertainties (if requested)
      - produce enhanced visualizations and outputs
    
    Returns:
      summary dict
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(forward_model.parameters()).device if hasattr(forward_model, 'parameters') else torch.device('cpu')
    obs_tensor = obs_tensor.to(device)
    
    # Use source plane if provided, otherwise create synthetic source
    if src_tensor is not None:
        src_tensor = src_tensor.to(device)
        logger.info(f"[{lens_id}] Using provided source plane reconstruction")
    else:
        # Create synthetic Gaussian source as fallback
        H = obs_tensor.shape[-2]; W = obs_tensor.shape[-1]
        yv, xv = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H), indexing='xy')
        sigma = 0.08
        gauss = np.exp(-0.5 * (xv**2 + yv**2) / (sigma**2)).astype(np.float32)
        gauss = gauss / (gauss.max() + 1e-12)
        src_tensor = torch.from_numpy(gauss).unsqueeze(0).unsqueeze(0).to(device)
        logger.info(f"[{lens_id}] Using synthetic Gaussian source")

    # encode + compute kappa
    forward_model.eval()
    with torch.no_grad():
        try:
            lens_params_torch, _ = forward_model.encode_lens_parameters(obs_tensor)
        except Exception as e:
            # Fallback for older model versions
            try:
                lens_params_torch = forward_model.encode_lens_parameters(obs_tensor)
            except Exception as e2:
                logger.error(f"[{lens_id}] encode_lens_parameters failed: {e2}")
                raise
        
        try:
            kappa_t, _ = forward_model.compute_convergence_map(obs_tensor)
            kappa_map = kappa_t.cpu().numpy()[0, 0]
        except Exception as e:
            logger.error(f"[{lens_id}] compute_convergence_map failed: {e}")
            raise
        
        # Generate model-predicted image from source plane
        try:
            model_pred, _, _ = forward_model(src_tensor, obs_tensor, add_noise=False, return_params=True)
            model_pred_np = model_pred.cpu().numpy()[0, 0]
            logger.info(f"[{lens_id}] Generated model prediction from source plane")
        except Exception as e:
            logger.warning(f"[{lens_id}] Failed to generate model prediction: {e}")
            model_pred_np = None
        
        # attempt deflection
        try:
            H, W = kappa_map.shape
            ax_t, ay_t = forward_model._compute_deflection(H, W, device, lens_params_torch)
            ax_map = ax_t.cpu().numpy()[0, 0]
            ay_map = ay_t.cpu().numpy()[0, 0]
            deflection_field = (ax_map, ay_map)
        except Exception:
            deflection_field = None

    lens_params_dict = {k: float(v.cpu().item()) for k, v in lens_params_torch.items()}

    # Convert tensors to numpy
    obs_np = obs_tensor.cpu().numpy()
    if obs_np.ndim == 4:
        obs_img = obs_np[0, 0]
    elif obs_np.ndim == 3:
        obs_img = obs_np[0]
    else:
        obs_img = np.squeeze(obs_np)
    
    src_np = src_tensor.cpu().numpy()
    if src_np.ndim == 4:
        src_img = src_np[0, 0]
    elif src_np.ndim == 3:
        src_img = src_np[0]
    else:
        src_img = np.squeeze(src_np)
    
    H, W = obs_img.shape

    # Compute residuals and metrics if we have model prediction
    residual_map = None
    chi2_map = None
    residual_metrics = {}
    if model_pred_np is not None:
        residual_map, chi2_map, residual_metrics = compute_residual_metrics(obs_img, model_pred_np, src_img)
        logger.info(f"[{lens_id}] Residual RMS: {residual_metrics['rms_residual']:.4f}, SNR: {residual_metrics['snr']:.2f}")

    # pixel scale determination
    pixscale_method = getattr(args, 'pixscale_method', 'assume') if args is not None else 'assume'
    pixel_scale_eff = float(getattr(args, 'pixel_scale', 0.05))
    
    if pixscale_method == 'psf_fwhm':
        assumed_seeing = getattr(args, 'assumed_seeing', 0.8)
        est = estimate_pixel_scale_from_psf(
            obs_img, 
            assumed_seeing_arcsec=assumed_seeing,
            verbose=getattr(args, 'verbose', False)
        )
        if est is not None:
            pixel_scale_eff = est
        else:
            logger.warning(f"[{lens_id}] PSF-based pixel-scale failed; using fallback {pixel_scale_eff:.5f} arcsec/pix")
    elif pixscale_method == 'calibrate':
        target_med = getattr(args, 'target_median_theta', None)
        if target_med is None:
            logger.warning("Calibrate selected but --target-median-theta not provided; using assume.")
        else:
            b_norm = lens_params_dict['b']
            arcsec_per_norm_req = target_med / max(1e-12, b_norm)
            pixel_scale_eff = arcsec_per_norm_req / (W/2.0)
            logger.info(f"[{lens_id}] Calibrated pixel scale: {pixel_scale_eff:.5f} arcsec/pix")

    # instantiate analyzer with cosmology settings
    cosmo_name = getattr(args, 'cosmology', None) if args is not None else None
    if cosmo_name:
        try:
            cosmo = getattr(astro_cosmo, cosmo_name)
        except AttributeError:
            logger.warning(f"Unknown cosmology '{cosmo_name}', using Planck18")
            cosmo = Planck18
    else:
        cosmo = Planck18
        
    analyzer = MassProfileAnalyzer(
        pixel_scale=pixel_scale_eff, 
        cosmology=cosmo, 
        z_lens=getattr(args, 'z_lens', 0.5), 
        z_source=getattr(args, 'z_source', 2.0)
    )

    # radial profile
    radial_r_arcsec, radial_kappa, radial_std, radial_counts = analyzer.radial_profile(
        kappa_map, (lens_params_dict['x0'], lens_params_dict['y0']), n_bins=50
    )
    radial_profile_data = (radial_r_arcsec, radial_kappa, radial_std, radial_counts)

    # physical mass at Einstein radius
    einstein_arcsec = analyzer.einstein_radius_arcsec(lens_params_torch, img_shape=(H, W))
    mass_phys = analyzer.mass_within_radius(kappa_map, einstein_arcsec, (lens_params_dict['x0'], lens_params_dict['y0']))
    
    # additional mass returned in normalized version
    mass_norm = {
        'sis_mass': float(lens_params_dict['b'] * einstein_arcsec),
        'nfw_mass': float(lens_params_dict.get('kappa_s', 0.0) * (lens_params_dict.get('rs', 0.0) * (W/2.0) * pixel_scale_eff)**2),
        'total_mass': None,
        'radius_arcsec': float(einstein_arcsec),
        'einstein_radius_arcsec': float(einstein_arcsec),
        'pixel_scale_used_arcsec_per_pix': float(pixel_scale_eff)
    }
    mass_norm['total_mass'] = mass_norm['sis_mass'] + mass_norm['nfw_mass']

    mass_phys_out = {
        'mass_Msun': mass_phys['mass_Msun'],
        'n_pixels': mass_phys['n_pixels'],
        'einstein_radius_arcsec': einstein_arcsec,
        'sigma_crit_kg_m2': mass_phys.get('sigma_crit_kg_m2', None),
        'pixel_scale_used_arcsec_per_pix': pixel_scale_eff,
        'z_lens': analyzer.z_lens,
        'z_source': analyzer.z_source
    }

    # velocity dispersion estimate
    try:
        sigma_v_kms = analyzer.sigma_v_from_thetaE(einstein_arcsec, (H, W))
    except Exception:
        sigma_v_kms = None

    # bootstrap radial uncertainties if requested
    bootstrap_profiles = None
    n_boot = max(0, int(getattr(args, 'n_bootstrap', 0))) if args is not None else 0
    if n_boot > 0:
        try:
            bootstrap_profiles = compute_bootstrap_radial(
                kappa_map, (lens_params_dict['x0'], lens_params_dict['y0']), 
                analyzer, n_boot=n_boot, n_bins=50
            )
            logger.info(f"[{lens_id}] Completed bootstrap (n={n_boot}).")
        except Exception as e:
            logger.warning(f"[{lens_id}] Bootstrap failed: {e}")
            bootstrap_profiles = None

    # assemble halo_info with source plane metrics
    halo_info = {
        'kappa_s': float(lens_params_dict.get('kappa_s', 0.0)),
        'rs_norm': float(lens_params_dict.get('rs', 0.0)),
        'rs_arcsec': float(lens_params_dict.get('rs', 0.0) * (W / 2.0) * pixel_scale_eff),
        'sigma_v_kms': sigma_v_kms,
        'source_available': src_tensor is not None
    }
    
    # Add residual metrics to halo_info
    halo_info.update(residual_metrics)

    # Output files: PNG figure, JSON, optional HDF5, CSV radial profile, LaTeX table, PDF summary
    png_path = os.path.join(output_dir, f"{lens_id}_mass_profile.png")
    try:
        create_comprehensive_analysis_figure(
            obs_img, kappa_map, lens_params_dict, radial_profile_data,
            mass_phys_out, mass_norm, halo_info, png_path,
            deflection_field=deflection_field, 
            model_pred=model_pred_np,
            source_plane=src_img,
            residual_map=residual_map,
            chi2_map=chi2_map,
            save_individual_panels=getattr(args, 'save_individual_panels', False),
            include_3d=getattr(args, 'include_3d', False),
            interactive=getattr(args, 'interactive', False),
            zoom_box=getattr(args, 'zoom', None),
            photometry=None,
            bootstrap_profiles=bootstrap_profiles
        )
    except Exception as e:
        logger.warning(f"[{lens_id}] Figure creation failed: {e}")

    # JSON params with enhanced information
    json_path = os.path.join(output_dir, f"{lens_id}_parameters.json")
    outdict = {
        'lens_id': lens_id,
        'lens_parameters': lens_params_dict,
        'mass_physical': mass_phys_out,
        'mass_normalized': mass_norm,
        'halo_info': halo_info,
        'pixel_scale_used_arcsec_per_pix': pixel_scale_eff,
        'residual_metrics': residual_metrics,
        'source_plane_info': {
            'available': src_tensor is not None,
            'shape': src_img.shape if src_tensor is not None else None,
            'max_value': float(np.nanmax(src_img)) if src_tensor is not None else None,
            'mean_value': float(np.nanmean(src_img)) if src_tensor is not None else None
        },
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    with open(json_path, 'w') as f:
        json.dump(outdict, f, indent=2)
    logger.info(f"[{lens_id}] Saved parameters JSON: {json_path}")

    # optional HDF5 with source plane
    if getattr(args, 'hdf5_output', False) and HDF5_AVAILABLE:
        try:
            h5_path = os.path.join(output_dir, f"{lens_id}_data.h5")
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset('kappa', data=kappa_map, compression='gzip')
                hf.create_dataset('obs', data=obs_img, compression='gzip')
                if src_tensor is not None:
                    hf.create_dataset('source', data=src_img, compression='gzip')
                if model_pred_np is not None:
                    hf.create_dataset('model_pred', data=model_pred_np, compression='gzip')
                if residual_map is not None:
                    hf.create_dataset('residual', data=residual_map, compression='gzip')
                if chi2_map is not None:
                    hf.create_dataset('chi2', data=chi2_map, compression='gzip')
                
                hf.attrs['pixel_scale_arcsec_per_pix'] = pixel_scale_eff
                hf.attrs['z_lens'] = analyzer.z_lens
                hf.attrs['z_source'] = analyzer.z_source
                hf.attrs['einstein_radius_arcsec'] = einstein_arcsec
                hf.attrs['mass_Msun'] = mass_phys['mass_Msun']
                
                # Store lens parameters as attributes
                for k, v in lens_params_dict.items():
                    hf.attrs[f'lens_{k}'] = v
                
            logger.info(f"[{lens_id}] Saved HDF5: {h5_path}")
        except Exception as e:
            logger.warning(f"[{lens_id}] HDF5 save failed: {e}")

    # optional CSV radial profile
    if getattr(args, 'save_csv', False) and PANDAS_AVAILABLE:
        try:
            csv_path = os.path.join(output_dir, f"{lens_id}_radial_profile.csv")
            df = pd.DataFrame({
                'radius_arcsec': radial_r_arcsec,
                'kappa_mean': radial_kappa,
                'kappa_std': radial_std,
                'counts': radial_counts
            })
            df.to_csv(csv_path, index=False)
            logger.info(f"[{lens_id}] Saved radial profile CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"[{lens_id}] CSV save failed: {e}")

    # optional LaTeX table of parameters
    if getattr(args, 'latex_table', False) and PANDAS_AVAILABLE:
        try:
            # Create enhanced parameter table
            params_table = lens_params_dict.copy()
            params_table['theta_E'] = einstein_arcsec
            params_table['sigma_v'] = sigma_v_kms
            params_table['mass_Msun'] = mass_phys['mass_Msun']
            if residual_metrics:
                params_table['rms_residual'] = residual_metrics.get('rms_residual', np.nan)
                params_table['reduced_chi2'] = residual_metrics.get('reduced_chi2', np.nan)
            
            dfp = pd.DataFrame([params_table])
            tex_path = os.path.join(output_dir, f"{lens_id}_params.tex")
            with open(tex_path, 'w') as f:
                f.write(dfp.to_latex(index=False, float_format="%.4g"))
            logger.info(f"[{lens_id}] Saved LaTeX params table: {tex_path}")
        except Exception as e:
            logger.warning(f"[{lens_id}] LaTeX table failed: {e}")

    # return summary for batch aggregations
    summary = {
        'lens_id': lens_id,
        'params': lens_params_dict,
        'mass_physical': mass_phys_out,
        'mass_normalized': mass_norm,
        'halo_info': halo_info,
        'residual_metrics': residual_metrics,
        'source_available': src_tensor is not None,
        'png': png_path,
        'json': json_path
    }
    return summary

# Batch analysis manager 

def batch_analyze(forward_model_path, fits_files: List[str], output_dir: str, args):
    os.makedirs(output_dir, exist_ok=True)
    n_procs = max(1, int(getattr(args, 'n_procs', 1)))
    results = []
    
    if n_procs == 1:
        # single process: load forward model once
        device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        
        # Instantiate forward model with parameters from args
        forward_model = ConditionalPhysicalForward(
            kernel_size=args.kernel_size,
            enforce_nonneg=True,
            init_fwhm=3.0,
            init_beta=4.5,
            pixel_scale=0.05,  # Fixed for the model
            learn_residual_psf=args.learn_residual_psf,
            residual_scale=1e-2,
            use_fft_conv=True,
            use_nfw=args.use_nfw,
            use_subhalos=args.use_subhalos,
            n_subhalos=args.n_subhalos
        ).to(device)
        
        try:
            st = torch.load(forward_model_path, map_location='cpu')
            if isinstance(st, dict) and ('forward_state' in st or 'state_dict' in st):
                sd = st.get('forward_state', st.get('state_dict', st))
                forward_model.load_state_dict(sd)
            else:
                forward_model.load_state_dict(st)
            logger.info("Loaded forward model for batch (single-process).")
        except Exception as e:
            logger.warning(f"Forward model load failed (single-process): {e}. Attempting loose load.")
            try:
                forward_model.load_state_dict(torch.load(forward_model_path, map_location='cpu'), strict=False)
                logger.info("Loaded forward model (strict=False).")
            except Exception as e2:
                logger.error(f"Cannot load forward model: {e2}")
                return []

        forward_model.to(device)
        forward_model.eval()
        
        for idx, f in enumerate(tqdm(fits_files, desc="Batch analyze")):
            if getattr(args, 'max_samples', None) is not None and idx >= int(args.max_samples):
                break
            lens_id = f"lens_{idx+1:03d}"
            try:
                obs_tensor, src_tensor = load_fits_pair(f, 
                                                       source_hdu_names=getattr(args, 'source_hdu_names', 
                                                                               ['RECON', 'SRC', 'SOURCE', 'SOURCE_PLANE']))
                res = analyze_single_lens(forward_model, obs_tensor, output_dir, lens_id, args=args, src_tensor=src_tensor)
                results.append(res)
            except Exception as e:
                logger.warning(f"[{lens_id}] Analysis failed: {e}")
    else:
        # multiprocessing: spawn workers that each load the forward model 
        logger.info(f"Starting multiprocessing pool with {n_procs} workers...")
        pool = mp.Pool(processes=n_procs)
        worker = partial(analyze_single_lens_worker, forward_model_path)
        tasks = []
        for idx, f in enumerate(fits_files):
            if getattr(args, 'max_samples', None) is not None and idx >= int(args.max_samples):
                break
            lens_id = f"lens_{idx+1:03d}"
            tasks.append(pool.apply_async(worker, args=(f, output_dir, lens_id, args)))
        pool.close()
        for t in tqdm(tasks, desc="Batch analyze (MP)"):
            try:
                res = t.get()
                results.append(res)
            except Exception as e:
                logger.warning(f"Worker failed: {e}")
        pool.join()

    # summary statistics across batch
    if results:
        # compute means/stds for selected parameters
        param_names = ['b','q','phi','gamma','kappa_s','rs']
        stats = {}
        for p in param_names:
            vals = []
            for r in results:
                v = r['params'].get(p, np.nan)
                vals.append(float(v))
            vals = np.array(vals, dtype=float)
            stats[p] = {'mean': np.nanmean(vals), 'std': np.nanstd(vals), 'min': np.nanmin(vals), 'max': np.nanmax(vals)}
        
        # Add source plane statistics
        source_available = [r.get('source_available', False) for r in results]
        stats['source_available'] = {'count': sum(source_available), 'fraction': np.mean(source_available)}
        
        # Add residual statistics
        if all('residual_metrics' in r for r in results):
            rms_vals = [r['residual_metrics'].get('rms_residual', np.nan) for r in results]
            chi2_vals = [r['residual_metrics'].get('reduced_chi2', np.nan) for r in results]
            stats['residual_rms'] = {'mean': np.nanmean(rms_vals), 'std': np.nanstd(rms_vals)}
            stats['reduced_chi2'] = {'mean': np.nanmean(chi2_vals), 'std': np.nanstd(chi2_vals)}
        
        # save summary JSON
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({'n': len(results), 'stats': stats, 'source_plane_fraction': stats['source_available']['fraction']}, f, indent=2)
        logger.info(f"Saved batch summary: {summary_path}")
    return results

# CLI 

def main():
    parser = argparse.ArgumentParser(description="Mass Profile Analysis for Conditional Lensing Models with Source Plane Support")
    parser.add_argument('--forward-model', type=str, default="PATH TO YOUR FORWARD MODEL",)
    parser.add_argument('--input', type=str, default="PATH TO YOUR INPUT FITS OR DIRECTORY")
    parser.add_argument('--output-dir', type=str, default='PATH TO YOUR OUTPUT DIRECTORY')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--pixel-scale', type=float, default=0.214, help='Fallback pixel scale (arcsec/pixel)')
    parser.add_argument('--pixscale-method', choices=['assume', 'psf_fwhm', 'calibrate'], default='psf_fwhm')
    parser.add_argument('--assumed-seeing', type=float, default=0.8, help='Assumed seeing FWHM in arcsec for PSF-based pixel scale estimation')
    parser.add_argument('--target-median-theta', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda','cpu'])
    parser.add_argument('--z-lens', type=float, default=0.68)
    parser.add_argument('--z-source', type=float, default=1.73)
    parser.add_argument('--cosmology', type=str, default='Planck18', help='Astropy cosmology name (e.g., Planck18, WMAP9)')
    parser.add_argument('--interactive', action='store_true', help='Enable matplotlib interactive mode')
    parser.add_argument('--include-3d', action='store_true', help='Include optional 3D κ surface panel')
    parser.add_argument('--save-individual-panels', action='store_true', help='Save each panel as a separate PNG')
    parser.add_argument('--n-bootstrap', type=int, default=0, help='Number of bootstrap resamples for profile uncertainties')
    parser.add_argument('--n-procs', type=int, default=1, help='Number of parallel processes for batch processing')
    parser.add_argument('--hdf5-output', action='store_true', help='Save HDF5 outputs (requires h5py)')
    parser.add_argument('--save-csv', action='store_true', help='Save radial profile CSV (requires pandas)')
    parser.add_argument('--latex-table', action='store_true', help='Save parameters as LaTeX table (requires pandas)')
    parser.add_argument('--summary-pdf', action='store_true', help='Create summary PDF combining key figures')
    parser.add_argument('--zoom', nargs=3, type=float, metavar=('X0','Y0','R'), help='Zoom region center (norm coords x0,y0 in [-1,1]) and radius in arcsec')
    parser.add_argument('--config-file', type=str, default=None, help='YAML/JSON config file with settings')
    parser.add_argument('--verbose', action='store_true')
    
    # Forward model parameters
    parser.add_argument('--use-nfw', action='store_true', default=True, help='Enable NFW halo in forward operator')
    parser.add_argument('--use-subhalos', action='store_true', default=False, help='Enable subhalos in forward operator')
    parser.add_argument('--n-subhalos', type=int, default=2, help='Number of subhalos')
    parser.add_argument('--kernel-size', type=int, default=21, help='PSF kernel size')
    parser.add_argument('--learn-residual-psf', action='store_true', default=True, help='Learn residual PSF corrections')
    
    # Source plane parameters (NEW)
    parser.add_argument('--source-hdu-names', nargs='+', default=['RECON', 'SRC', 'SOURCE', 'SOURCE_PLANE'],
                       help='HDU names to search for source plane reconstruction')
    parser.add_argument('--force-synthetic-source', action='store_true', 
                       help='Force use of synthetic Gaussian source even if source plane is available')
    parser.add_argument('--compare-source-plane', action='store_true',
                       help='Compare results with and without source plane when available')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # load config overrides if provided
    if args.config_file and YAML_AVAILABLE:
        try:
            with open(args.config_file, 'r') as cf:
                cfg = yaml.safe_load(cf)
                # apply some keys
                for k,v in (cfg or {}).items():
                    if hasattr(args, k):
                        setattr(args, k, v)
            logger.info(f"Loaded config {args.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")

    if os.path.isfile(args.input):
        fits_files = [args.input]
    elif os.path.isdir(args.input):
        fits_files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.fits')])
    else:
        logger.error("Input not found")
        return

    if len(fits_files) == 0:
        logger.error("No FITS files found")
        return

    logger.info(f"Found {len(fits_files)} FITS file(s)")
    if args.force_synthetic_source:
        logger.info("Forcing synthetic source (ignoring any source plane reconstructions)")

    results = batch_analyze(args.forward_model, fits_files, args.output_dir, args)

    # optional summary PDF: collect PNGs and combine
    if args.summary_pdf and results:
        pdf_path = os.path.join(args.output_dir, 'summary_report.pdf')
        try:
            with PdfPages(pdf_path) as pdf:
                for r in results:
                    # each r may be summary dict with 'png' or be tuple
                    png = r['png'] if isinstance(r, dict) and 'png' in r else None
                    if png and os.path.exists(png):
                        img = plt.imread(png)
                        fig = plt.figure(figsize=(8.27, 11.69))  # A4
                        ax = fig.add_subplot(111)
                        ax.imshow(img)
                        ax.axis('off')
                        
                        # Add title with lens info
                        lens_id = r.get('lens_id', 'Unknown')
                        source_avail = r.get('source_available', False)
                        title = f"Lens: {lens_id} - Source plane: {'Yes' if source_avail else 'No'}"
                        ax.set_title(title, fontsize=10, pad=20)
                        
                        pdf.savefig(fig)
                        plt.close(fig)
            logger.info(f"Saved summary PDF: {pdf_path}")
        except Exception as e:
            logger.warning(f"Summary PDF creation failed: {e}")

    logger.info("All done.")

if __name__ == '__main__':
    main()