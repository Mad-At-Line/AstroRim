from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astropy.io import fits

CORE_VERSION = "2.1.0"

TRUTH_ORDER: List[str] = [
    'b', 'q', 'phi', 'x0', 'y0', 'gamma', 'gamma_phi', 'kappa_s', 'rs',
    'psf_fwhm', 'psf_beta', 'psf_e1', 'psf_e2',
    'lens_flux', 'lens_Re', 'lens_n',
]

TRUTH_FITS_KEYS: Dict[str, str] = {
    'b': 'TR_B', 'q': 'TR_Q', 'phi': 'TR_PHI', 'x0': 'TR_X0', 'y0': 'TR_Y0',
    'gamma': 'TR_GAM', 'gamma_phi': 'TR_GPHI', 'kappa_s': 'TR_KS', 'rs': 'TR_RS',
    'psf_fwhm': 'TR_PSFW', 'psf_beta': 'TR_PSFB',
    'psf_e1': 'TR_PSFE1', 'psf_e2': 'TR_PSFE2',
    'lens_flux': 'TR_LFLX', 'lens_Re': 'TR_LRE', 'lens_n': 'TR_LN',
}
TRUTH_MASK_KEY = 'TR_MASK'
TRUTH_CLEAN_KEY = 'TR_CLEAN'
TRUTH_SET_KEY = 'TR_SET'        # simulator variant name
TRUTH_PIXSCALE_KEY = 'TR_PS'    # arcsec / detector pixel used by the simulator

# Axis-type angles (period pi): supervised losses must compare them on the
# (cos 2x, sin 2x) circle, never by raw difference.
PERIODIC_PI_PARAMS = ('phi', 'gamma_phi')


def pack_truth_from_header(header) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read TR_* keywords into (vec[16], mask[16], info). Missing -> mask 0."""
    n = len(TRUTH_ORDER)
    vec = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    declared = header.get(TRUTH_MASK_KEY, None)
    for i, name in enumerate(TRUTH_ORDER):
        key = TRUTH_FITS_KEYS[name]
        if key in header:
            try:
                v = float(header[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vec[i] = v
                bit_ok = True
                if declared is not None:
                    try:
                        bit_ok = bool((int(declared) >> i) & 1)
                    except (TypeError, ValueError):
                        bit_ok = True
                mask[i] = 1.0 if bit_ok else 0.0
    info = {
        'clean': int(header.get(TRUTH_CLEAN_KEY, 0) or 0),
        'sim_set': str(header.get(TRUTH_SET_KEY, '') or ''),
        'pixscale': float(header[TRUTH_PIXSCALE_KEY]) if TRUTH_PIXSCALE_KEY in header else float('nan'),
    }
    return vec, mask, info


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


DEFAULT_PSF_FWHM_RANGE = (0.3, 25.0)   # [FIX C2] was (1.0, 25.0)
LEGACY_PSF_FWHM_RANGE = (1.0, 25.0)

BASE_PARAM_SPECS: Dict[str, Tuple[str, float, float]] = {
    'b':         ('sigmoid', 0.01, 1.00),
    'q':         ('sigmoid', 0.2, 1.0),
    'phi':       ('tanh', -math.pi, math.pi),
    'x0':        ('tanh', -0.2, 0.2),
    'y0':        ('tanh', -0.2, 0.2),
    'gamma':     ('sigmoid', 0.0, 0.2),
    'gamma_phi': ('tanh', -math.pi, math.pi),
    'kappa_s':   ('sigmoid', 0.005, 0.5),
    'rs':        ('sigmoid', 0.05, 0.5),
    'psf_beta':  ('sigmoid', 2.5, 5.0),
    'psf_e1':    ('tanh', -0.15, 0.15),
    'psf_e2':    ('tanh', -0.15, 0.15),
    'lens_flux': ('sigmoid', 0.0, 0.5),
    'lens_Re':   ('sigmoid', 0.02, 0.60),
    'lens_n':    ('sigmoid', 1.0, 6.0),
}


def squash_param(name: str, raw: torch.Tensor,
                 psf_fwhm_range: Tuple[float, float] = DEFAULT_PSF_FWHM_RANGE) -> torch.Tensor:
    if name == 'psf_fwhm':
        lo, hi = psf_fwhm_range
        lf = math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw)
        return torch.exp(lf)
    kind, lo, hi = BASE_PARAM_SPECS[name]
    if kind == 'sigmoid':
        return lo + (hi - lo) * torch.sigmoid(raw)
    if kind == 'tanh':
        return hi * torch.tanh(raw)
    raise KeyError(name)


def unsquash_param(name: str, value: torch.Tensor,
                   psf_fwhm_range: Tuple[float, float] = DEFAULT_PSF_FWHM_RANGE) -> torch.Tensor:
    """Inverse of squash_param (clamped away from the bounds for stability).
    Used by MAP refinement and the global-parameter ablation initializer."""
    eps = 1e-4
    if name == 'psf_fwhm':
        lo, hi = psf_fwhm_range
        t = (torch.log(value.clamp(lo * (1 + 1e-6), hi * (1 - 1e-6))) - math.log(lo)) \
            / (math.log(hi) - math.log(lo))
        t = t.clamp(eps, 1 - eps)
        return torch.log(t / (1 - t))
    kind, lo, hi = BASE_PARAM_SPECS[name]
    if kind == 'sigmoid':
        t = ((value - lo) / (hi - lo)).clamp(eps, 1 - eps)
        return torch.log(t / (1 - t))
    if kind == 'tanh':
        t = (value / hi).clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(t) - torch.log1p(-t))
    raise KeyError(name)


def param_range_width(name: str,
                      psf_fwhm_range: Tuple[float, float] = DEFAULT_PSF_FWHM_RANGE) -> float:
    """Range width used to normalize supervised parameter losses."""
    if name == 'psf_fwhm':
        return psf_fwhm_range[1] - psf_fwhm_range[0]
    kind, lo, hi = BASE_PARAM_SPECS[name]
    if kind == 'tanh':
        return 2.0 * hi
    return hi - lo

class LensParameterEncoder(nn.Module):
    """Encodes observed image to lens, PSF, and lens-light parameters."""

    def __init__(self, input_channels=1, latent_dim=128,
                 psf_fwhm_range: Tuple[float, float] = DEFAULT_PSF_FWHM_RANGE):
        super().__init__()
        self.psf_fwhm_range = tuple(float(v) for v in psf_fwhm_range)
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

        # Lens mass heads
        self.b_head = nn.Linear(latent_dim // 2, 1)
        self.q_head = nn.Linear(latent_dim // 2, 1)
        self.phi_head = nn.Linear(latent_dim // 2, 1)
        self.x0_head = nn.Linear(latent_dim // 2, 1)
        self.y0_head = nn.Linear(latent_dim // 2, 1)
        self.gamma_head = nn.Linear(latent_dim // 2, 1)
        self.gamma_phi_head = nn.Linear(latent_dim // 2, 1)
        self.kappa_s_head = nn.Linear(latent_dim // 2, 1)
        self.rs_head = nn.Linear(latent_dim // 2, 1)

        # PSF heads (per-observation). FWHM in detector pixels; beta = Moffat index.
        self.psf_fwhm_head = nn.Linear(latent_dim // 2, 1)
        self.psf_beta_head = nn.Linear(latent_dim // 2, 1)
        self.psf_e_head = nn.Linear(latent_dim // 2, 2)  # (e1, e2)

        # Lens-light Sersic heads. Center shared with the mass center (x0, y0).
        self.lens_flux_head = nn.Linear(latent_dim // 2, 1)
        self.lens_Re_head = nn.Linear(latent_dim // 2, 1)
        self.lens_n_head = nn.Linear(latent_dim // 2, 1)

        self.apply(init_weights)

    def features(self, x):
        h = F.relu(self.gn1(self.conv1(x)))
        h = self.pool(h)
        h = F.relu(self.gn2(self.conv2(h)))
        h = self.pool(h)
        h = F.relu(self.gn3(self.conv3(h)))
        h = self.pool(h)
        h = F.relu(self.gn4(self.conv4(h)))
        h = self.pool(h)
        return self.adaptive_pool(h)

    def forward(self, x):
        z = self.features(x)
        flat = z.view(z.size(0), -1)
        f1 = F.relu(self.fc1(flat))
        f2 = F.relu(self.fc2(f1))

        out = {}
        heads = {
            'b': self.b_head, 'q': self.q_head, 'phi': self.phi_head,
            'x0': self.x0_head, 'y0': self.y0_head,
            'gamma': self.gamma_head, 'gamma_phi': self.gamma_phi_head,
            'kappa_s': self.kappa_s_head, 'rs': self.rs_head,
            'psf_fwhm': self.psf_fwhm_head, 'psf_beta': self.psf_beta_head,
            'lens_flux': self.lens_flux_head, 'lens_Re': self.lens_Re_head,
            'lens_n': self.lens_n_head,
        }
        for name, head in heads.items():
            out[name] = squash_param(name, head(f2),
                                     psf_fwhm_range=self.psf_fwhm_range).squeeze(-1)
        psf_e_raw = self.psf_e_head(f2)
        out['psf_e1'] = squash_param('psf_e1', psf_e_raw[:, 0])
        out['psf_e2'] = squash_param('psf_e2', psf_e_raw[:, 1])
        return out


class ConditionalNFWDeflection(nn.Module):
    def __init__(self):
        super().__init__()

    def nfw_deflection_angle(self, x_grid, y_grid, kappa_s, rs, x0, y0):
        eps = 1e-8
        dx = x_grid - x0.unsqueeze(-1).unsqueeze(-1)
        dy = y_grid - y0.unsqueeze(-1).unsqueeze(-1)
        r = torch.sqrt(dx ** 2 + dy ** 2 + eps)

        x_norm = r / (rs.unsqueeze(-1).unsqueeze(-1) + eps)
        x_norm = torch.clamp(x_norm, min=1e-4, max=1e4)

        h = torch.zeros_like(x_norm)
        band = 1e-3
        mask_lt1 = x_norm < (1.0 - band)
        mask_gt1 = x_norm > (1.0 + band)
        mask_eq1 = ~(mask_lt1 | mask_gt1)

        if mask_lt1.any():
            xs = x_norm[mask_lt1]
            inv_xs = 1.0 / xs
            acosh_term = torch.log(inv_xs + torch.sqrt(torch.clamp(inv_xs ** 2 - 1.0, min=1e-12)))
            sqrt_term = torch.sqrt(torch.clamp(1.0 - xs ** 2, min=1e-12))
            h[mask_lt1] = torch.log(xs / 2.0) + acosh_term / sqrt_term

        if mask_gt1.any():
            xl = x_norm[mask_gt1]
            inv_xl = 1.0 / xl
            acos_arg = torch.clamp(inv_xl, min=-1.0 + 1e-7, max=1.0 - 1e-7)
            acos_term = torch.acos(acos_arg)
            sqrt_term = torch.sqrt(torch.clamp(xl ** 2 - 1.0, min=1e-12))
            h[mask_gt1] = torch.log(xl / 2.0) + acos_term / sqrt_term

        if mask_eq1.any():
            h[mask_eq1] = 1.0 + math.log(0.5)

        alpha_magnitude = (4.0 / (x_norm + 1e-12)) * h
        alpha_r = alpha_magnitude * kappa_s.unsqueeze(-1).unsqueeze(-1) * rs.unsqueeze(-1).unsqueeze(-1)

        cos_theta = dx / (r + eps)
        sin_theta = dy / (r + eps)
        return alpha_r * cos_theta, alpha_r * sin_theta

    def nfw_convergence(self, x_grid, y_grid, kappa_s, rs, x0, y0):
        """Wright & Brainerd (2000) NFW convergence (batch-safe torch.where form)."""
        eps = 1e-8
        dx = x_grid - x0.unsqueeze(-1).unsqueeze(-1)
        dy = y_grid - y0.unsqueeze(-1).unsqueeze(-1)
        r = torch.sqrt(dx ** 2 + dy ** 2 + eps)
        x_norm = r / (rs.unsqueeze(-1).unsqueeze(-1) + eps)
        x_norm = torch.clamp(x_norm, min=1e-4, max=1e4)

        def safe_atanh(t):
            t = torch.clamp(t, min=-1.0 + 1e-6, max=1.0 - 1e-6)
            return 0.5 * (torch.log1p(t) - torch.log1p(-t))

        ks = kappa_s.unsqueeze(-1).unsqueeze(-1)

        x_lt = torch.clamp(x_norm, min=1e-4, max=1.0 - 1e-4)
        denom_lt = x_lt ** 2 - 1.0
        root_lt = torch.sqrt(torch.clamp(1.0 - x_lt ** 2, min=1e-12))
        u_lt = torch.sqrt(torch.clamp((1.0 - x_lt) / (1.0 + x_lt), min=0.0, max=1.0 - 1e-6))
        kappa_lt = (2.0 * ks / (denom_lt + 1e-12)) * \
                   (1.0 - (2.0 / (root_lt + 1e-12)) * safe_atanh(u_lt))

        x_gt = torch.clamp(x_norm, min=1.0 + 1e-4, max=1e4)
        denom_gt = x_gt ** 2 - 1.0
        root_gt = torch.sqrt(torch.clamp(x_gt ** 2 - 1.0, min=1e-12))
        v_gt = torch.sqrt(torch.clamp((x_gt - 1.0) / (1.0 + x_gt), min=0.0))
        kappa_gt = (2.0 * ks / (denom_gt + 1e-12)) * \
                   (1.0 - (2.0 / (root_gt + 1e-12)) * torch.atan(v_gt))

        kappa_eq = 2.0 * ks / 3.0 * torch.ones_like(x_norm)

        band = 1e-3
        kappa = torch.where(x_norm < (1.0 - band), kappa_lt, kappa_eq)
        kappa = torch.where(x_norm > (1.0 + band), kappa_gt, kappa)
        return kappa

class ConditionalSubhaloComponent(nn.Module):
    def __init__(self, n_subhalos=3):
        super().__init__()
        self.n_subhalos = n_subhalos

    def compute_deflection(self, x_grid, y_grid, subhalo_params):
        B = x_grid.shape[0]
        H, W = x_grid.shape[1], x_grid.shape[2]
        alpha_x_total = torch.zeros(B, H, W, device=x_grid.device)
        alpha_y_total = torch.zeros(B, H, W, device=x_grid.device)
        params = subhalo_params.view(B, self.n_subhalos, 4)
        for i in range(self.n_subhalos):
            x_sub = params[:, i, 0].unsqueeze(-1).unsqueeze(-1)
            y_sub = params[:, i, 1].unsqueeze(-1).unsqueeze(-1)
            theta_e = params[:, i, 2].unsqueeze(-1).unsqueeze(-1)
            rc = params[:, i, 3].unsqueeze(-1).unsqueeze(-1)
            dx = x_grid - x_sub
            dy = y_grid - y_sub
            r = torch.sqrt(dx ** 2 + dy ** 2 + rc ** 2 + 1e-12)
            alpha_r = theta_e * (1.0 - rc / (r + rc + 1e-12))
            alpha_x_total += alpha_r * dx / (r + 1e-12)
            alpha_y_total += alpha_r * dy / (r + 1e-12)
        return alpha_x_total, alpha_y_total


class ConditionalPhysicalForward(nn.Module):
    """Differentiable lensing forward operator with per-observation conditioning.

    v2.1 additions:
      * param_mode 'encoder' (default, v2.0 behavior) or 'global' [NEW C6].
      * forward()/adjoint() accept a precomputed lens_params dict so the RIM
        can encode once per reconstruction [NEW C8] and so test-time MAP
        refinement can inject refined parameters.
      * _conv_per_obs implements real FFT convolution [FIX C3].
    """

    def __init__(self,
                 kernel_size: int = 21,
                 enforce_nonneg: bool = True,
                 pixel_scale: float = 0.05,
                 learn_residual_psf: bool = True,
                 residual_scale: float = 1e-2,
                 use_fft_conv: bool = True,
                 use_nfw: bool = True,
                 use_subhalos: bool = False,
                 n_subhalos: int = 2,
                 use_per_obs_psf: bool = True,
                 use_lens_light: bool = True,
                 psf_fwhm_range: Tuple[float, float] = DEFAULT_PSF_FWHM_RANGE,
                 param_mode: str = 'encoder'):
        super().__init__()
        assert param_mode in ('encoder', 'global'), param_mode
        self.kernel_size = kernel_size
        self.enforce_nonneg = enforce_nonneg
        self.pixel_scale = pixel_scale
        self.use_fft_conv = use_fft_conv
        self.use_nfw = use_nfw
        self.use_subhalos = use_subhalos
        self.n_subhalos = n_subhalos
        self.use_per_obs_psf = use_per_obs_psf
        self.use_lens_light = use_lens_light
        self.psf_fwhm_range = tuple(float(v) for v in psf_fwhm_range)
        self.param_mode = param_mode

        self.encoder = LensParameterEncoder(latent_dim=128,
                                            psf_fwhm_range=self.psf_fwhm_range)

        # [NEW C6] One learnable raw value per parameter, squashed through the
        # same bounds as the encoder heads. Only exists in 'global' mode so
        # 'encoder'-mode state dicts are unchanged from v2.0.
        if self.param_mode == 'global':
            self._global_param_names = list(TRUTH_ORDER)
            self.global_raw = nn.Parameter(torch.zeros(len(self._global_param_names)))

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

        self.log_background = nn.Parameter(torch.log(torch.tensor(1e-2)))
        self.log_gain = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_sigma_read = nn.Parameter(torch.log(torch.tensor(1e-3)))

        self.apply(init_weights)

        if self.raw_res_psf is not None:
            with torch.no_grad():
                self.raw_res_psf.mul_(1e-2)

    @property
    def background(self):
        return torch.exp(self.log_background)

    @property
    def gain(self):
        return torch.exp(self.log_gain)

    @property
    def sigma_read(self):
        return torch.exp(self.log_sigma_read)

    # PSF

    def _build_moffat_kernel_batched(self, fwhm_px, beta, e1, e2, size, device):
        B = fwhm_px.shape[0]
        k = size
        ax = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        xx = xx.unsqueeze(0).expand(B, -1, -1)
        yy = yy.unsqueeze(0).expand(B, -1, -1)

        e1 = e1.view(B, 1, 1)
        e2 = e2.view(B, 1, 1)
        x_ell = xx * (1.0 + e1) + yy * e2
        y_ell = xx * e2 + yy * (1.0 - e1)
        r2 = x_ell ** 2 + y_ell ** 2

        fwhm_px = fwhm_px.view(B, 1, 1)
        beta = beta.view(B, 1, 1)
        alpha = fwhm_px / (2.0 * torch.sqrt(2.0 ** (1.0 / beta) - 1.0) + 1e-12)
        kern = (1.0 + (r2 / (alpha ** 2 + 1e-12))) ** (-beta)
        kern = kern / (kern.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return kern.unsqueeze(1)

    def _build_global_moffat_kernel(self, size, device):
        """Fallback (use_per_obs_psf=False): fixed global Moffat (v1 reproduction)."""
        k = size
        fwhm = 3.0
        beta = 4.5
        alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0) + 1e-12)
        ax = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        r2 = xx ** 2 + yy ** 2
        moff = (1.0 + (r2 / (alpha ** 2))) ** (-beta)
        moff = moff / (moff.sum() + 1e-12)
        return moff.unsqueeze(0).unsqueeze(0)

    def _get_psf_batched(self, lens_params, device):
        if self.use_per_obs_psf:
            kern = self._build_moffat_kernel_batched(
                lens_params['psf_fwhm'], lens_params['psf_beta'],
                lens_params['psf_e1'], lens_params['psf_e2'],
                self.kernel_size, device)
        else:
            B = lens_params['b'].shape[0]
            base = self._build_global_moffat_kernel(self.kernel_size, device)
            kern = base.expand(B, -1, -1, -1).contiguous()

        if self.raw_res_psf is not None:
            kern = kern + self.raw_res_psf.to(device)

        if self.enforce_nonneg:
            kern = torch.relu(kern)
        s = kern.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        kern = kern / s
        kern = torch.clamp(kern, min=0.0, max=1.0)
        kern = kern / (kern.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return kern

    #  geometry 

    def _build_mesh(self, B, H, W, device):
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((xv, yv), dim=-1)
        return grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # SIE deflection (Kormann et al. 1994) -- unchanged, verified [KEPT C9]
    def _sie_deflection(self, x_rot, y_rot, b, q, rc):
        eps = 1e-8
        q_safe = torch.clamp(q, min=0.2, max=0.9999)
        psi = torch.sqrt(q_safe ** 2 * (x_rot ** 2 + rc ** 2) + y_rot ** 2 + eps)
        qsq_comp = torch.sqrt(torch.clamp(1.0 - q_safe ** 2, min=eps))

        prefactor = b * torch.sqrt(q_safe) / qsq_comp

        arg_x = qsq_comp * x_rot / (psi + rc + eps)
        arg_y = qsq_comp * y_rot / (psi + q_safe ** 2 * rc + eps)
        arg_y = torch.clamp(arg_y, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        alpha_x = prefactor * torch.atan(arg_x)
        alpha_y = prefactor * 0.5 * (torch.log1p(arg_y) - torch.log1p(-arg_y))

        q_is_one = (q > 0.999).expand_as(x_rot)
        if q_is_one.any():
            r_sis = torch.sqrt(x_rot ** 2 + y_rot ** 2 + rc ** 2 + eps)
            alpha_x = torch.where(q_is_one, b * x_rot / r_sis, alpha_x)
            alpha_y = torch.where(q_is_one, b * y_rot / r_sis, alpha_y)
        return alpha_x, alpha_y

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

        ax_lens, ay_lens = self._sie_deflection(x_rot, y_rot, b, q, rc)

        c2 = torch.cos(phi)
        s2 = torch.sin(phi)
        ax = c2 * ax_lens - s2 * ay_lens
        ay = s2 * ax_lens + c2 * ay_lens

        cos2 = torch.cos(2.0 * gamma_phi)
        sin2 = torch.sin(2.0 * gamma_phi)
        ax = ax + gamma * (xv * cos2 + yv * sin2)
        ay = ay + gamma * (xv * sin2 - yv * cos2)

        if self.use_nfw:
            kappa_s = lens_params['kappa_s'].view(B)
            rs = lens_params['rs'].view(B)
            ax_nfw, ay_nfw = self.nfw.nfw_deflection_angle(
                xv, yv, kappa_s, rs,
                x0.squeeze(-1).squeeze(-1), y0.squeeze(-1).squeeze(-1))
            ax = ax + ax_nfw
            ay = ay + ay_nfw

        if self.use_subhalos and subhalo_params is not None:
            ax_sub, ay_sub = self.subhalos.compute_deflection(xv, yv, subhalo_params)
            ax = ax + ax_sub
            ay = ay + ay_sub

        return ax.unsqueeze(1), ay.unsqueeze(1)

    # encoding 

    def global_params(self, B: int, device) -> Dict[str, torch.Tensor]:
        """[NEW C6] One shared learned parameter set, expanded to batch size."""
        out = {}
        for i, name in enumerate(self._global_param_names):
            raw = self.global_raw[i]
            val = squash_param(name, raw, psf_fwhm_range=self.psf_fwhm_range)
            out[name] = val.to(device).expand(B)
        return out

    def encode_lens_parameters(self, obs):
        B = obs.shape[0]
        if self.param_mode == 'global':
            return self.global_params(B, obs.device), None

        lens_params = self.encoder(obs)

        subhalo_params = None
        if self.use_subhalos:
            z = self.encoder.features(obs)
            features_flat = z.view(B, -1)
            sp = self.subhalo_encoder(features_flat).view(B, self.n_subhalos, 4)
            subhalo_params = torch.stack([
                0.5 * torch.tanh(sp[:, :, 0]),
                0.5 * torch.tanh(sp[:, :, 1]),
                0.05 * torch.sigmoid(sp[:, :, 2]),
                0.01 + 0.09 * torch.sigmoid(sp[:, :, 3]),
            ], dim=-1)
        return lens_params, subhalo_params

    # lens light 

    def _sersic_lens_light(self, H, W, device, lens_params):
        B = lens_params['b'].shape[0]
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='ij')
        xv = xv.unsqueeze(0).repeat(B, 1, 1)
        yv = yv.unsqueeze(0).repeat(B, 1, 1)

        x0 = lens_params['x0'].view(B, 1, 1)
        y0 = lens_params['y0'].view(B, 1, 1)
        Re = lens_params['lens_Re'].view(B, 1, 1)
        n = lens_params['lens_n'].view(B, 1, 1)
        flux = lens_params['lens_flux'].view(B, 1, 1)

        dx = xv - x0
        dy = yv - y0
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        bn = 2.0 * n - 0.331  # Graham & Driver (2005), n > 0.5
        intensity = flux * torch.exp(-bn * ((r / (Re + 1e-6)) ** (1.0 / n) - 1.0))
        return intensity.unsqueeze(1)

    # convolution 

    def _conv_per_obs(self, img, psf_batch):
        
        B, C, H, W = img.shape
        kH, kW = psf_batch.shape[-2], psf_batch.shape[-1]
        pad_h = kH // 2
        pad_w = kW // 2

        if self.use_fft_conv and max(H, W) > 64:
            kern = torch.flip(psf_batch, dims=(-2, -1)).reshape(B * C, 1, kH, kW)
            x = img.reshape(B * C, 1, H, W)
            fh, fw = H + kH - 1, W + kW - 1
            Xf = torch.fft.rfft2(x.float(), s=(fh, fw))
            Kf = torch.fft.rfft2(kern.float(), s=(fh, fw))
            full = torch.fft.irfft2(Xf * Kf, s=(fh, fw))
            out = full[..., pad_h:pad_h + H, pad_w:pad_w + W]
            return out.reshape(B, C, H, W).to(img.dtype)

        img_g = img.view(1, B * C, H, W)
        kern_g = psf_batch.view(B * C, 1, kH, kW)
        out = F.conv2d(img_g, kern_g, padding=(pad_h, pad_w), groups=B * C)
        return out.view(B, C, H, W)

    # forward 

    def forward(self, src, obs, add_noise: bool = False, return_params: bool = False,
                lens_params: Optional[Dict[str, torch.Tensor]] = None,
                subhalo_params: Optional[torch.Tensor] = None):
        B, C, H, W = src.shape
        device = src.device
        if lens_params is None:
            lens_params, subhalo_params = self.encode_lens_parameters(obs)
        ax, ay = self._compute_deflection(H, W, device, lens_params, subhalo_params)

        grid = self._build_mesh(B, H, W, device)
        grid_x = grid[..., 0] - ax.squeeze(1)
        grid_y = grid[..., 1] - ay.squeeze(1)
        samp_grid = torch.stack((grid_x, grid_y), dim=-1)
        # [FIX T13] Do NOT clamp sampling coordinates into [-1, 1]. padding_mode
        # 'zeros' already maps out-of-FOV coordinates to 0 (physically correct:
        # no source flux maps there). The old component-wise clamp instead folded
        # every out-of-bounds coordinate onto the source-plane edge row/column,
        # producing the square "frame" + axis-aligned cross seen in the
        # reconstructions. Only sanitise non-finite coordinates (-> out of bounds
        # -> 0); the clamp never actually guarded against those (clamp(nan)=nan).
        samp_grid = torch.nan_to_num(samp_grid, nan=2.0, posinf=2.0, neginf=-2.0)
        y_warp = F.grid_sample(src, samp_grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)

        psf_batch = self._get_psf_batched(lens_params, device)
        y_conv = self._conv_per_obs(y_warp, psf_batch)

        if self.use_lens_light:
            lens_light = self._sersic_lens_light(H, W, device, lens_params)
            y_conv = y_conv + self._conv_per_obs(lens_light, psf_batch)

        y_conv = y_conv + self.background

        if add_noise:
            gain = self.gain
            counts = torch.clamp(y_conv * gain, min=0.0)
            counts_noisy = torch.poisson(counts)
            counts_noisy = counts_noisy + torch.randn_like(counts_noisy) * self.sigma_read
            y_out = counts_noisy / (gain + 1e-12)
        else:
            y_out = y_conv

        if return_params:
            return y_out, lens_params, subhalo_params
        return y_out

    def adjoint(self, residual, obs,
                lens_params: Optional[Dict[str, torch.Tensor]] = None,
                subhalo_params: Optional[torch.Tensor] = None):
        """Adjoint of warp+PSF (lens light is additive: zero source-gradient)."""
        if lens_params is None:
            lens_params, subhalo_params = self.encode_lens_parameters(obs)
        device = residual.device

        psf_batch = self._get_psf_batched(lens_params, device)
        psf_flipped = torch.flip(psf_batch, dims=(-2, -1))
        r_conv = self._conv_per_obs(residual, psf_flipped)

        B, C, H, W = residual.shape
        grid = self._build_mesh(B, H, W, device)
        ax, ay = self._compute_deflection(H, W, device, lens_params, subhalo_params)

        grid_x = grid[..., 0] + ax.squeeze(1)
        grid_y = grid[..., 1] + ay.squeeze(1)
        adj_grid = torch.stack((grid_x, grid_y), dim=-1)
        # [FIX T13] See forward(): drop the edge-folding clamp; rely on
        # padding_mode='zeros' for out-of-FOV, sanitise only non-finite coords.
        adj_grid = torch.nan_to_num(adj_grid, nan=2.0, posinf=2.0, neginf=-2.0)
        return F.grid_sample(r_conv, adj_grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    #  convergence 

    def compute_convergence_map(self, obs, H=None, W=None,
                                lens_params: Optional[Dict[str, torch.Tensor]] = None):
        device = obs.device
        B = obs.shape[0]
        if H is None or W is None:
            H, W = obs.shape[-2], obs.shape[-1]
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='ij')
        xv = xv.unsqueeze(0).repeat(B, 1, 1)
        yv = yv.unsqueeze(0).repeat(B, 1, 1)

        if lens_params is None:
            lens_params, _ = self.encode_lens_parameters(obs)

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

        r_ell = torch.sqrt(q ** 2 * (x_rot ** 2 + rc ** 2) + y_rot ** 2 + 1e-12)
        kappa = 0.5 * b * torch.sqrt(q) / r_ell

        if self.use_nfw:
            kappa = kappa + self.nfw.nfw_convergence(
                xv, yv, lens_params['kappa_s'].view(B), lens_params['rs'].view(B),
                lens_params['x0'].view(B), lens_params['y0'].view(B))
        return kappa

    #  regularization 

    def compute_regularization_loss(self, lens_params, subhalo_params=None):
        device = next(self.parameters()).device
        reg = torch.tensor(0.0, device=device)

        b = lens_params['b']
        q = lens_params['q']
        x0 = lens_params['x0']
        y0 = lens_params['y0']
        gamma = lens_params['gamma']

        reg = reg + torch.mean(torch.relu(-b) ** 2)
        reg = reg + torch.mean(torch.relu(b - 1.0) ** 2)
        reg = reg + 0.1 * torch.mean(x0 ** 2)
        reg = reg + 0.1 * torch.mean(y0 ** 2)
        reg = reg + 0.1 * torch.mean((gamma - 0.05) ** 2)

        reg = reg + 0.05 * torch.mean(torch.relu(torch.abs(b - 0.15) - 0.30) ** 2)
        reg = reg + 0.05 * torch.mean(torch.relu(torch.abs(q - 0.70) - 0.35) ** 2)

        if self.use_nfw:
            kappa_s = lens_params['kappa_s']
            rs = lens_params['rs']
            reg = reg + 0.02 * torch.mean((kappa_s - 0.05) ** 2)
            reg = reg + 0.02 * torch.mean((rs - 0.20) ** 2)

        if self.use_lens_light:
            lens_flux = lens_params['lens_flux']
            reg = reg + 0.01 * torch.mean(lens_flux ** 2)

        if self.use_subhalos and subhalo_params is not None:
            B = subhalo_params.shape[0]
            sub_params = subhalo_params.view(B, self.n_subhalos, 4)
            theta_e = sub_params[:, :, 2]
            reg = reg + 0.1 * torch.mean(theta_e ** 2)
            x_sub = sub_params[:, :, 0]
            y_sub = sub_params[:, :, 1]
            reg = reg + 0.01 * torch.mean(torch.relu(torch.abs(x_sub) - 0.9) ** 2)
            reg = reg + 0.01 * torch.mean(torch.relu(torch.abs(y_sub) - 0.9) ** 2)
            if self.n_subhalos > 1:
                for i in range(self.n_subhalos):
                    for j in range(i + 1, self.n_subhalos):
                        dxs = x_sub[:, i] - x_sub[:, j]
                        dys = y_sub[:, i] - y_sub[:, j]
                        dist = torch.sqrt(dxs ** 2 + dys ** 2 + 1e-6)
                        reg = reg + 0.1 * torch.mean(torch.relu(0.1 - dist) ** 2)
        return reg


# RIM 

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
        update_gate = torch.sigmoid(self.gates(combined))
        cand = torch.tanh(self.candidate(combined))
        h_new = (1 - update_gate) * h + update_gate * cand
        delta = self.step * self.to_image(h_new)
        x_new = x - delta
        return x_new, h_new


class ConditionalRIM(nn.Module):
    

    def __init__(self, n_iter=10, hidden_dim=128):
        super().__init__()
        self.n_iter = n_iter
        self.hidden_dim = hidden_dim
        self.cell = ConditionalRIMCell(hidden_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    def forward(self, y, obs, forward_operator, return_intermediates: bool = False,
                lens_params: Optional[Dict[str, torch.Tensor]] = None,
                subhalo_params: Optional[torch.Tensor] = None):
        B, C, H, W = y.shape
        device = y.device
        h = torch.zeros(B, self.hidden_dim, H, W, device=device)

        if lens_params is None:
            lens_params, subhalo_params = forward_operator.encode_lens_parameters(obs)

        x = forward_operator.adjoint(y, obs, lens_params=lens_params,
                                     subhalo_params=subhalo_params)

        intermediates = []
        for _ in range(self.n_iter):
            y_sim = forward_operator.forward(x, obs, lens_params=lens_params,
                                             subhalo_params=subhalo_params)
            residual = y_sim - y
            grad = forward_operator.adjoint(residual, obs, lens_params=lens_params,
                                            subhalo_params=subhalo_params)
            x, h = self.cell(x, grad, h)
            if return_intermediates:
                intermediates.append(x)

        x = x + self.refine(h)
        if return_intermediates:
            return x, intermediates
        return x



# Shadow EMA 


class ShadowEMA:
    def __init__(self, modules: List[nn.Module], decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self._modules = list(modules)
        for m in self._modules:
            for p in m.parameters():
                if p.requires_grad:
                    self.shadow[id(p)] = p.detach().clone()

    @torch.no_grad()
    def update(self):
        d = self.decay
        for m in self._modules:
            for p in m.parameters():
                if p.requires_grad and id(p) in self.shadow:
                    self.shadow[id(p)].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, modules: List[nn.Module]):
        backup = {}
        for m in modules:
            for p in m.parameters():
                if id(p) in self.shadow:
                    backup[id(p)] = p.detach().clone()
                    p.data.copy_(self.shadow[id(p)])
        return backup

    @torch.no_grad()
    def restore(self, modules: List[nn.Module], backup: dict):
        for m in modules:
            for p in m.parameters():
                if id(p) in backup:
                    p.data.copy_(backup[id(p)])

    def state_dict(self):
        return {str(k): v.cpu() for k, v in self.shadow.items()}



def _wrap_axis_angle(a: float) -> float:
    """Wrap an axis-type angle into (-pi/2, pi/2]."""
    a = (a + math.pi / 2.0) % math.pi - math.pi / 2.0
    if a <= -math.pi / 2.0:
        a += math.pi
    return a


def transform_truth_fliplr(t: dict) -> dict:
    t = dict(t)
    t['x0'] = -t['x0']
    t['phi'] = _wrap_axis_angle(-t['phi'])
    t['gamma_phi'] = _wrap_axis_angle(-t['gamma_phi'])
    t['psf_e2'] = -t['psf_e2']
    return t


def transform_truth_flipud(t: dict) -> dict:
    t = dict(t)
    t['y0'] = -t['y0']
    t['phi'] = _wrap_axis_angle(-t['phi'])
    t['gamma_phi'] = _wrap_axis_angle(-t['gamma_phi'])
    t['psf_e2'] = -t['psf_e2']
    return t


def transform_truth_rot90(t: dict) -> dict:
    t = dict(t)
    x0, y0 = t['x0'], t['y0']
    t['x0'], t['y0'] = y0, -x0
    t['phi'] = _wrap_axis_angle(t['phi'] - math.pi / 2.0)
    t['gamma_phi'] = _wrap_axis_angle(t['gamma_phi'] - math.pi / 2.0)
    t['psf_e1'] = -t['psf_e1']
    t['psf_e2'] = -t['psf_e2']
    return t


class ConditionalLensingFitsDataset(torch.utils.data.Dataset):
    def __init__(self, files, augment=False, return_truth: bool = False):
        self.files = sorted(files)
        assert len(self.files) > 0, "No FITS files provided"
        self.augment = augment
        self.return_truth = return_truth

    def __len__(self):
        return len(self.files)

    def _read_pair(self, fn):
        hdul = fits.open(fn, memmap=False)
        gt = None
        lensed = None
        truth_vec = np.zeros(len(TRUTH_ORDER), dtype=np.float32)
        truth_mask = np.zeros(len(TRUTH_ORDER), dtype=np.float32)

        if self.return_truth:
            try:
                truth_vec, truth_mask, _ = pack_truth_from_header(hdul[0].header)
            except Exception:
                pass

        for h in hdul:
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

        return lensed, gt, truth_vec, truth_mask

    def __getitem__(self, idx):
        fn = self.files[idx]
        obs, gt, tvec, tmask = self._read_pair(fn)

        if self.augment:
            tdict = {name: float(tvec[i]) for i, name in enumerate(TRUTH_ORDER)}
            if np.random.rand() < 0.5:
                obs = np.fliplr(obs)
                gt = np.fliplr(gt)
                tdict = transform_truth_fliplr(tdict)
            if np.random.rand() < 0.5:
                obs = np.flipud(obs)
                gt = np.flipud(gt)
                tdict = transform_truth_flipud(tdict)
            k = np.random.choice([0, 1, 2, 3])
            if k != 0:
                obs = np.rot90(obs, k)
                gt = np.rot90(gt, k)
                for _ in range(int(k)):
                    tdict = transform_truth_rot90(tdict)
            tvec = np.array([tdict[name] for name in TRUTH_ORDER], dtype=np.float32)

        obs_t = torch.from_numpy(obs.copy()).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        if self.return_truth:
            return obs_t, gt_t, torch.from_numpy(tvec.copy()), torch.from_numpy(tmask.copy())
        return obs_t, gt_t


CHECKPOINT_META_KEY = 'astrorim_meta'


def extract_state_and_meta(blob, prefer=('forward_state', 'model_state', 'state_dict')):
    """Accept either a bare state_dict or a dict containing one under common
    keys; return (state_dict, meta_dict_or_empty)."""
    meta = {}
    if isinstance(blob, dict):
        if CHECKPOINT_META_KEY in blob and isinstance(blob[CHECKPOINT_META_KEY], dict):
            meta = blob[CHECKPOINT_META_KEY]
        for key in prefer:
            if key in blob and isinstance(blob[key], dict):
                return blob[key], meta
        # Heuristic: a state dict's values are tensors
        if blob and all(isinstance(v, torch.Tensor) for v in blob.values()):
            return blob, meta
    return blob, meta


def load_state_dict_diagnosed(module: nn.Module, state_dict: dict, label: str = 'model',
                              allow_partial: bool = False) -> dict:
    model_keys = set(module.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    if not missing and not unexpected:
        module.load_state_dict(state_dict, strict=True)
        return {'missing': [], 'unexpected': [], 'partial': False}

    print(f"[checkpoint:{label}] key mismatch:")
    if missing:
        print(f"  MISSING from checkpoint ({len(missing)}) -- these stay at their"
              f" current (random/initial) values if partial loading is allowed:")
        for k in missing:
            print(f"    - {k}")
    if unexpected:
        print(f"  UNEXPECTED in checkpoint ({len(unexpected)}) -- ignored:")
        for k in unexpected:
            print(f"    - {k}")

    if not allow_partial:
        raise RuntimeError(
            f"[checkpoint:{label}] refusing to load a mismatched checkpoint "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys). "
            f"This usually means a v1 checkpoint or a different architecture "
            f"config (use_nfw/use_subhalos/param_mode). Pass allow_partial=True "
            f"(--allow-partial-load) only if you understand which parts will "
            f"remain at initialization.")

    module.load_state_dict(state_dict, strict=False)
    print(f"[checkpoint:{label}] partial load complete "
          f"({len(model_keys) - len(missing)}/{len(model_keys)} tensors loaded).")
    return {'missing': missing, 'unexpected': unexpected, 'partial': True}


def load_checkpoint_file(module: nn.Module, path: str, label: str,
                         prefer=('forward_state', 'model_state', 'state_dict'),
                         allow_partial: bool = False, map_location='cpu') -> dict:
    """torch.load + extract + diagnosed strict load. Returns meta dict.
    Warns if the checkpoint meta declares a psf_fwhm_range different from the
    module's (the [FIX C2] bound change makes this an easy silent error)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    blob = torch.load(path, map_location=map_location)
    state, meta = extract_state_and_meta(blob, prefer=prefer)
    load_state_dict_diagnosed(module, state, label=label, allow_partial=allow_partial)

    ck_range = meta.get('psf_fwhm_range', None)
    mod_range = getattr(module, 'psf_fwhm_range', None)
    if ck_range is not None and mod_range is not None:
        if tuple(float(v) for v in ck_range) != tuple(float(v) for v in mod_range):
            print(f"[checkpoint:{label}] WARNING: checkpoint was trained with "
                  f"psf_fwhm_range={tuple(ck_range)} but the module is configured "
                  f"with {tuple(mod_range)}. PSF FWHM predictions will be remapped "
                  f"incorrectly. Re-instantiate with psf_fwhm_range={tuple(ck_range)} "
                  f"(--psf-fwhm-min/--psf-fwhm-max).")
    elif ck_range is None and mod_range is not None and tuple(mod_range) != LEGACY_PSF_FWHM_RANGE:
        print(f"[checkpoint:{label}] NOTE: checkpoint has no meta (pre-v2.1). If it "
              f"was trained with the legacy PSF FWHM bounds {LEGACY_PSF_FWHM_RANGE}, "
              f"pass --psf-fwhm-min 1.0 to reproduce its PSF predictions exactly.")
    return meta



# Supervised parameter loss

def supervised_param_loss(pred: Dict[str, torch.Tensor],
                          truth_vec: torch.Tensor,
                          truth_mask: torch.Tensor,
                          psf_fwhm_range: Tuple[float, float] = DEFAULT_PSF_FWHM_RANGE,
                          weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:

    device = truth_vec.device
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    per_param: Dict[str, float] = {}

    idx = {name: i for i, name in enumerate(TRUTH_ORDER)}
    w = weights or {}

    gamma_true = truth_vec[:, idx['gamma']]
    ks_true = truth_vec[:, idx['kappa_s']]

    for name in TRUTH_ORDER:
        i = idx[name]
        m = truth_mask[:, i]
        if m.sum() <= 0:
            continue
        p = pred[name]
        t = truth_vec[:, i]

        if name in PERIODIC_PI_PARAMS:
            # distance on the period-pi circle, scaled so a 90 deg error ~ 1
            d2 = ((torch.cos(2 * p) - torch.cos(2 * t)) ** 2 +
                  (torch.sin(2 * p) - torch.sin(2 * t)) ** 2) / 4.0
        elif name == 'psf_fwhm':
            # [FIX L1] psf_fwhm is squashed log-uniformly (line 108) but was
            # being penalized with a *linear* residual normalized by the full
            # encoder range (24.7 px). That makes a 150%-relative error at the
            # ~1.5-2.5px scale of real data score ~0.015 -- about 100x weaker
            # than the gradient a correctly-scaled parameter gets, so the
            # supervised signal couldn't out-compete the forward-fidelity
            # term's use of FWHM as a source-size-degeneracy slack variable.
            # Computing the residual in log space (matching the squash)
            # restores a relative-error-sized gradient regardless of where in
            # the range the true value sits.
            lo, hi = psf_fwhm_range
            log_width = math.log(hi) - math.log(lo)
            p_safe = p.clamp(min=lo * 0.5)  # guard log(<=0) during early/unstable training
            t_safe = t.clamp(min=lo * 0.5)
            d2 = ((torch.log(p_safe) - torch.log(t_safe)) / log_width) ** 2
        else:
            width = param_range_width(name, psf_fwhm_range=psf_fwhm_range)
            d2 = ((p - t) / width) ** 2

        eff_w = float(w.get(name, 1.0)) * m
        if name == 'gamma_phi':
            eff_w = eff_w * (gamma_true / 0.2).clamp(0.0, 1.0)
        if name == 'rs':
            eff_w = eff_w * (ks_true / 0.05).clamp(0.0, 1.0)

        contrib = (eff_w * d2).sum()
        weight_sum = eff_w.sum()
        total = total + contrib
        denom = denom + weight_sum
        if weight_sum.item() > 0:
            per_param[name] = float((contrib / weight_sum).detach().cpu())

    if denom.item() <= 0:
        return torch.tensor(0.0, device=device), per_param
    return total / denom, per_param