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

# ----------------------- Dataset (robust to PrimaryHDU GT + ImageHDU LENSED) -----------------------
class LensingFitsDataset(Dataset):
    def __init__(self, files, augment=False, debug=False):
        self.files = sorted(files)
        assert len(self.files) > 0, "No FITS files provided"
        self.augment = augment
        self.debug = debug

    def __len__(self):
        return len(self.files)

    def _read_pair(self, fn):
        hdul = fits.open(fn, memmap=False)
        gt = None
        lensed = None

        # 1) Try to find by EXTNAME / h.name (explicit)
        for i, h in enumerate(hdul):
            extname_hdr = h.header.get('EXTNAME', '')
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
            elif name == 'LENSED' or name == 'OBS' or name == 'OBSERVED':
                lensed = d

        # 2) Fallback: PrimaryHDU -> GT, second -> LENSED
        if gt is None:
            primary = hdul[0]
            if primary.data is not None:
                try:
                    gt = primary.data.astype(np.float32)
                except Exception:
                    gt = None

        if lensed is None:
            if len(hdul) > 1 and hdul[1].data is not None:
                try:
                    lensed = hdul[1].data.astype(np.float32)
                except Exception:
                    lensed = None

        # 3) Further heuristic
        if gt is None or lensed is None:
            for h in hdul:
                hdr = h.header
                if h.data is None:
                    continue
                if ('SRCMAG' in hdr or 'THETA_E' in hdr or 'PSF_FWH_TRUE' in hdr) and gt is None:
                    try:
                        gt = h.data.astype(np.float32)
                    except Exception:
                        pass
                if ('PSF_FWH' in hdr or 'SKYADU' in hdr or 'LENSED' in hdr.values()) and lensed is None and h is not hdul[0]:
                    try:
                        lensed = h.data.astype(np.float32)
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
            raise RuntimeError(f"{fn} missing GT or LENSED HDU after heuristics.\nDetected ext list:\n" + "\n".join(info_lines))

        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        eps = 1e-8
        m = max(gt.max() if gt.size else 0.0, lensed.max() if lensed.size else 0.0, eps)
        gt = gt / m
        lensed = lensed / m
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

# ---------------------------
# Helper: Moffat kernel
# ---------------------------
def make_moffat_kernel(fwhm_pix: float, beta: float, size: int, eps: float = 1e-12):
    assert size % 2 == 1
    alpha = fwhm_pix / (2.0 * math.sqrt(2 ** (1.0 / beta) - 1.0) + 1e-12)
    k = size
    ax = np.arange(-(k // 2), k // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    rr2 = xx ** 2 + yy ** 2
    moff = (1.0 + (rr2 / (alpha ** 2))) ** (-beta)
    moff = moff.astype(np.float32)
    moff = moff / (moff.sum() + eps)
    return torch.from_numpy(moff).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

# -----------------------
# PhysicalForwardAdvanced (new)
# -----------------------
class PhysicalForwardAdvanced(nn.Module):
    def __init__(self,
                 kernel_size: int = 21,
                 device: Optional[torch.device] = None,
                 enforce_nonneg: bool = True,
                 init_fwhm: float = 3.0,
                 init_beta: float = 4.5,
                 init_b: float = 0.08,
                 init_rc: float = 0.01,
                 pixel_scale: float = 0.05,
                 learn_residual_psf: bool = True,
                 residual_scale: float = 1e-2,
                 use_fft_conv: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.device_cached = device
        self.enforce_nonneg = enforce_nonneg
        self.pixel_scale = pixel_scale
        self.use_fft_conv = use_fft_conv

        # lens params
        self.x0 = nn.Parameter(torch.tensor(0.0))
        self.y0 = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(init_b))
        self.raw_rc = nn.Parameter(torch.tensor(init_rc))
        self.raw_q = nn.Parameter(torch.tensor(0.0))
        self.raw_phi = nn.Parameter(torch.tensor(0.0))

        # shear
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))
        self.raw_gamma_phi = nn.Parameter(torch.tensor(0.0))

        self.raw_subpix = nn.Parameter(torch.zeros(2))

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

        self.log_sigma = self.log_sigma_read

        if self.raw_res_psf is not None:
            with torch.no_grad():
                self.raw_res_psf.mul_(1e-2)

        self.psf_tv_weight = 1e-3

    @property
    def b_pos(self):
        return F.softplus(self.raw_b) + 1e-12

    @property
    def rc_pos(self):
        return F.softplus(self.raw_rc) + 1e-12

    @property
    def q_pos(self):
        q = torch.sigmoid(self.raw_q)
        return 0.2 + 0.8 * q

    @property
    def phi(self):
        return self.raw_phi

    @property
    def gamma(self):
        return torch.tanh(self.raw_gamma) * 0.2

    @property
    def gamma_phi(self):
        return self.raw_gamma_phi

    @property
    def subpix(self):
        return 0.5 * torch.tanh(self.raw_subpix)

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
            device = self.device_cached if self.device_cached is not None else torch.device('cpu')
        size = size or self.kernel_size
        fwhm = float(self.fwhm.detach().cpu().item())
        beta = float(self.beta.detach().cpu().item())
        moff = make_moffat_kernel(fwhm_pix=fwhm, beta=beta, size=size).to(device)
        return moff

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
        s = k.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        return k / s

    def _build_mesh(self, B, H, W, device):
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='xy')
        grid = torch.stack((xv, yv), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def _compute_deflection(self, H, W, device):
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        yv, xv = torch.meshgrid(ys, xs, indexing='xy')

        b = self.b_pos
        rc = self.rc_pos
        q = self.q_pos
        phi = self.phi
        x0 = self.x0
        y0 = self.y0

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

        gamma = self.gamma
        gphi = self.gamma_phi
        cos2 = torch.cos(2.0 * gphi)
        sin2 = torch.sin(2.0 * gphi)
        ax_s = gamma * (xv * cos2 + yv * sin2)
        ay_s = gamma * (xv * sin2 - yv * cos2)

        ax = ax + ax_s
        ay = ay + ay_s

        ax = ax.unsqueeze(0).unsqueeze(0)
        ay = ay.unsqueeze(0).unsqueeze(0)
        return ax, ay

    def warp_source(self, src):
        B, C, H, W = src.shape
        device = src.device
        grid = self._build_mesh(B, H, W, device)
        ax, ay = self._compute_deflection(H, W, device)
        ax_b = ax.repeat(B, 1, 1, 1).squeeze(1)
        ay_b = ay.repeat(B, 1, 1, 1).squeeze(1)
        sx = (self.subpix[0] * 2.0) / max(1, W - 1)
        sy = (self.subpix[1] * 2.0) / max(1, H - 1)
        grid_x = grid[..., 0] - ax_b + sx
        grid_y = grid[..., 1] - ay_b + sy
        samp_grid = torch.stack((grid_x, grid_y), dim=-1)
        y_sim = F.grid_sample(src, samp_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return y_sim

    def warp_adjoint(self, residual):
        B, C, H, W = residual.shape
        device = residual.device
        grid = self._build_mesh(B, H, W, device)
        ax, ay = self._compute_deflection(H, W, device)
        ax_b = ax.repeat(B, 1, 1, 1).squeeze(1)
        ay_b = ay.repeat(B, 1, 1, 1).squeeze(1)
        sx = (self.subpix[0] * 2.0) / max(1, W - 1)
        sy = (self.subpix[1] * 2.0) / max(1, H - 1)
        grid_x = grid[..., 0] + ax_b - sx
        grid_y = grid[..., 1] + ay_b - sy
        adj_grid = torch.stack((grid_x, grid_y), dim=-1)
        src_grad = F.grid_sample(residual, adj_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return src_grad

    def _fft_conv2d(self, x, kernel):
        B, C, H, W = x.shape
        kH, kW = kernel.shape[-2], kernel.shape[-1]
        outH = H + kH - 1
        outW = W + kW - 1
        nH = int(2 ** math.ceil(math.log2(outH)))
        nW = int(2 ** math.ceil(math.log2(outW)))
        x_pad = F.pad(x, (0, nW - W, 0, nH - H))
        k_pad = torch.zeros(1, 1, nH, nW, device=x.device, dtype=x.dtype)
        k_pad[..., :kH, :kW] = kernel
        Xf = torch.fft.rfft2(x_pad, dim=(-2, -1))
        Kf = torch.fft.rfft2(k_pad, dim=(-2, -1))
        Yf = Xf * Kf
        y = torch.fft.irfft2(Yf, s=(nH, nW), dim=(-2, -1))
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
        y = y[..., pad_h:pad_h + H, pad_w:pad_w + W]
        return y

    def forward(self, src, add_noise: bool = False, return_noise_map: bool = False):
        B, C, H, W = src.shape
        device = src.device
        y_warp = self.warp_source(src)
        psf = self.get_psf().to(device)
        if self.use_fft_conv and max(H, W) > 64:
            y_conv = self._fft_conv2d(y_warp, psf)
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
            noise_map = (torch.sqrt(torch.clamp(counts, min=0.0)) + sigma_read) / (gain + 1e-12)
            if return_noise_map:
                return y_noisy, noise_map
            return y_noisy
        else:
            if return_noise_map:
                noise_map = (torch.sqrt(torch.clamp(y_conv * self.gain, min=0.0)) + self.sigma_read) / (self.gain + 1e-12)
                return y_conv, noise_map
            return y_conv

    def adjoint(self, residual):
        psf = self.get_psf().to(next(self.parameters()).device)
        psf_flipped = torch.flip(psf, dims=(2, 3))
        if self.use_fft_conv:
            r_conv = self._fft_conv2d(residual, psf_flipped)
        else:
            r_conv = F.conv2d(residual, psf_flipped, padding=self.kernel_size // 2)
        src_grad = self.warp_adjoint(r_conv)
        return src_grad

    def psf_tv(self):
        if self.raw_res_psf is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        r = self.raw_res_psf
        tv = torch.mean(torch.abs(r[:, :, :-1, :] - r[:, :, 1:, :])) + torch.mean(torch.abs(r[:, :, :, :-1] - r[:, :, :, 1:]))
        return tv

    def adjoint_test(self, H=64, W=64, device=None, atol=1e-4):
        device = device or (next(self.parameters()).device)
        B = 1
        x = torch.randn(B, 1, H, W, device=device)
        y = torch.randn(B, 1, H, W, device=device)
        Ax = self.forward(x)
        ATy = self.adjoint(y)
        lhs = (Ax * y).sum().item()
        rhs = (x * ATy).sum().item()
        rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)
        return lhs, rhs, rel

# ----------------------- RIM improved (unchanged) -----------------------
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

class RIMCellImproved(nn.Module):
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
        delta = self.step * self._to_image(h_new)
        x_new = x - delta
        return x_new, h_new

    def _to_image(self, h):
        return self.to_image(h)

class RIMImproved(nn.Module):
    def __init__(self, n_iter=8, hidden_dim=128):
        super().__init__()
        self.n_iter = n_iter
        self.hidden_dim = hidden_dim
        self.cell = RIMCellImproved(hidden_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    def forward(self, y, forward_operator):
        B, C, H, W = y.shape
        device = y.device
        h = torch.zeros(B, self.hidden_dim, H, W, device=device)
        x = forward_operator.adjoint(y)
        for _ in range(self.n_iter):
            y_sim = forward_operator.forward(x)
            residual = y_sim - y
            grad = forward_operator.adjoint(residual)
            x, h = self.cell(x, grad, h)
        x = x + self.refine(h)
        return x

# ----------------------- Utility functions -----------------------
def compute_grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
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
        recon = model(obs, forward_operator)

    B = obs.shape[0]
    n = min(B, num_examples)

    obs_np = obs.cpu().numpy()
    recon_np = recon.cpu().numpy()
    gt_np = gt.cpu().numpy()

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(3 * 3, n * 3))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i in range(n):
        o = np.clip(obs_np[i, 0], 0.0, 1.0)
        r = np.clip(recon_np[i, 0], 0.0, 1.0)
        g = np.clip(gt_np[i, 0], 0.0, 1.0)

        ax = axes[i, 0]
        ax.imshow(o, origin='lower')
        ax.set_title('Observed')
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(r, origin='lower')
        ax.set_title('Reconstruction')
        ax.axis('off')

        ax = axes[i, 2]
        ax.imshow(g, origin='lower')
        ax.set_title('Ground Truth')
        ax.axis('off')

    plt.suptitle(f'Reconstructions - Epoch {epoch}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f'recons_epoch_{epoch}.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved reconstructions to {out_path}")

# -----------------------
# Robust forward-model loader
# -----------------------
def load_forward_checkpoint(forward_operator, ck_path, map_device='cpu'):
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f"Forward checkpoint not found: {ck_path}")
    state = torch.load(ck_path, map_location=map_device)
    if isinstance(state, dict):
        # common wrapped keys
        if 'forward_state' in state and isinstance(state['forward_state'], dict):
            forward_operator.load_state_dict(state['forward_state'], strict=False)
            return
        if 'forward' in state and isinstance(state['forward'], dict):
            forward_operator.load_state_dict(state['forward'], strict=False)
            return
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            forward_operator.load_state_dict(state['state_dict'], strict=False)
            return
        # direct plausible state_dict
        keys = list(state.keys())
        if any(k in keys for k in ['raw_res_psf', 'log_fwhm', 'raw_b', 'raw_rc']):
            forward_operator.load_state_dict(state, strict=False)
            return
        # search nested
        for k, v in state.items():
            if isinstance(v, dict):
                vk = list(v.keys())
                if any(x in vk for x in ['raw_res_psf', 'log_fwhm', 'raw_b', 'raw_rc']):
                    forward_operator.load_state_dict(v, strict=False)
                    return
        # last resort
        forward_operator.load_state_dict(state, strict=False)
    else:
        forward_operator.load_state_dict(state, strict=False)

# ----------------------- Finetune training loop -----------------------
def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().isoformat()}] Using device: {device}")

    # Dataset
    all_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.fits")))
    if len(all_files) == 0:
        raise RuntimeError(f"No .fits files found in {args.dataset_dir}")
    train_files, val_files = train_test_split(all_files, test_size=args.val_split, random_state=42)

    train_dataset = LensingFitsDataset(train_files, augment=args.augment)
    val_dataset = LensingFitsDataset(val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type=='cuda'), num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=(device.type=='cuda'), num_workers=args.num_workers)

    # Instantiate advanced forward operator
    forward_operator = PhysicalForwardAdvanced(kernel_size=args.kernel_size, device=device, enforce_nonneg=True,
                                              init_fwhm=3.0, init_beta=4.5, init_b=0.08, init_rc=0.01,
                                              learn_residual_psf=True, use_fft_conv=True).to(device)
    model = RIMImproved(n_iter=args.n_iter, hidden_dim=args.hidden_dim).to(device)

    start_epoch = 0
    best_val_loss = float('inf')

    # Load checkpoints if provided
    if args.rim_checkpoint and os.path.exists(args.rim_checkpoint):
        print(f"Loading RIM weights from {args.rim_checkpoint}")
        sd = torch.load(args.rim_checkpoint, map_location='cpu')
        try:
            model.load_state_dict(sd)
        except Exception:
            if 'model_state' in sd:
                model.load_state_dict(sd['model_state'], strict=False)
            else:
                model.load_state_dict(sd, strict=False)
        print("Loaded RIM checkpoint.")

    if args.forward_checkpoint and os.path.exists(args.forward_checkpoint):
        print(f"Loading Forward operator weights from {args.forward_checkpoint}")
        try:
            load_forward_checkpoint(forward_operator, args.forward_checkpoint, map_device='cpu')
            print("Loaded forward checkpoint.")
        except Exception as e:
            print(f"Warning: failed to load forward checkpoint with robust loader: {e}")
            sd = torch.load(args.forward_checkpoint, map_location='cpu')
            try:
                forward_operator.load_state_dict(sd, strict=False)
                print("Loaded forward checkpoint via fallback load_state_dict(..., strict=False).")
            except Exception as e2:
                print(f"Failed to load forward checkpoint: {e2}")

    forward_operator.to(device)
    model.to(device)

    # Freeze logic
    if args.finetune_mode == 'rim':
        for p in forward_operator.parameters():
            p.requires_grad = False
        print("Freezing forward operator parameters (finetuning RIM only).")
    elif args.finetune_mode == 'forward':
        for p in model.parameters():
            p.requires_grad = False
        print("Freezing RIM parameters (finetuning forward operator only).")
    else:
        print("Finetuning both RIM and forward operator.")

    # Optional freeze of PSF-only (compatibly handle new forward operator)
    if args.freeze_psf:
        print("Freezing PSF-related parameters in forward operator (log_fwhm/log_beta/raw_res_psf if present).")
        for name, p in forward_operator.named_parameters():
            if any(k in name for k in ['raw_res_psf', 'log_fwhm', 'log_beta', 'raw_psf']):
                p.requires_grad = False

    # Setup optimizer: separate param groups if desired
    params = []
    if args.finetune_mode in ['both', 'rim']:
        params.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_rim})
    if args.finetune_mode in ['both', 'forward']:
        params.append({"params": [p for p in forward_operator.parameters() if p.requires_grad], "lr": args.lr_forward})
    if len(params) == 0:
        raise RuntimeError("No parameters left to optimize (check finetune_mode / freeze flags).")

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.use_amp))

    loss_history = {"train": [], "val": [], "val_ssim": [], "grad_norm": []}
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        ck = torch.load(args.resume_checkpoint, map_location='cpu')
        start_epoch = ck.get('epoch', 0)
        if 'optimizer' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer'])
                print("Loaded optimizer state.")
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")
        if 'best_val_loss' in ck:
            best_val_loss = ck['best_val_loss']
        print(f"Resuming from epoch {start_epoch} (loaded checkpoint file).")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        epoch_idx = epoch + 1
        start_epoch_time = time.time()
        model.train()
        forward_operator.train()
        running_loss = 0.0
        grad_norm_epoch = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch_idx}/{args.num_epochs}")
        for obs, gt in loop:
            obs = obs.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.use_amp)):
                recon = model(obs, forward_operator)
                mse = F.mse_loss(recon, gt, reduction='mean')

                psf = forward_operator.get_psf()
                lap = psf - F.avg_pool2d(psf, 3, 1, padding=1)
                psf_pen = (lap ** 2).mean()

                # mild lens prior to keep parameters reasonable
                lens_prior = ((forward_operator.b_pos - 0.1) ** 2 + (forward_operator.rc_pos - 0.01) ** 2).mean()
                subpix_pen = (forward_operator.subpix ** 2).mean()

                loss = mse + args.lambda_psf * psf_pen + args.lambda_lens_prior * lens_prior + args.lambda_subpix * subpix_pen

            scaler.scale(loss).backward()

            if args.use_amp and device.type == 'cuda':
                scaler.unscale_(optimizer)

            grad_norm = compute_grad_norm(list(model.parameters()) + list(forward_operator.parameters()))
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(forward_operator.parameters()), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * obs.size(0)
            grad_norm_epoch += grad_norm
            loop.set_postfix({"loss": f"{loss.item():.6e}", "g_norm": f"{grad_norm:.3f}"})

        avg_train_loss = running_loss / max(1, len(train_dataset))
        avg_grad_norm = grad_norm_epoch / max(1, len(train_loader))
        loss_history['train'].append(avg_train_loss)
        loss_history['grad_norm'].append(avg_grad_norm)

        # Validation
        model.eval()
        forward_operator.eval()
        val_loss = 0.0
        val_ssim = 0.0
        n_val = 0
        with torch.no_grad():
            for obs, gt in val_loader:
                obs = obs.to(device)
                gt = gt.to(device)
                recon = model(obs, forward_operator)
                l = F.mse_loss(recon, gt).item()
                val_loss += l * obs.size(0)

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
        loss_history['val'].append(avg_val_loss)
        loss_history['val_ssim'].append(avg_val_ssim)

        epoch_time = time.time() - start_epoch_time
        print(f"[Epoch {epoch_idx}] Train: {avg_train_loss:.6e} | Val: {avg_val_loss:.6e} | SSIM: {avg_val_ssim:.4f} | GradNorm: {avg_grad_norm:.3f} | Time: {epoch_time:.1f}s")

        try:
            scheduler.step(avg_val_loss)
        except Exception:
            print("Warning: scheduler.step failed with provided metric. Continuing without scheduler update.")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.6e} - saving best models.")
            rim_best_path = os.path.join(args.output_dir, args.save_rim_best)
            forward_best_path = os.path.join(args.output_dir, args.save_forward_best)
            torch.save(model.state_dict(), rim_best_path)
            torch.save(forward_operator.state_dict(), forward_best_path)
            print(f"Saved best RIM -> {rim_best_path}")
            print(f"Saved best Forward -> {forward_best_path}")

        # Checkpoints + PSF visualization
        if epoch_idx % args.checkpoint_every == 0 or epoch_idx == args.num_epochs:
            ck = {
                'epoch': epoch_idx,
                'model_state': model.state_dict(),
                'forward_state': forward_operator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            ck_path = os.path.join(args.checkpoint_dir, f"finetune_checkpoint_epoch_{epoch_idx}.pt")
            torch.save(ck, ck_path)
            print(f"Saved checkpoint at epoch {epoch_idx} -> {ck_path}")

            with torch.no_grad():
                k = forward_operator.get_psf().cpu().numpy()[0, 0]
                plt.figure(figsize=(4, 4))
                plt.imshow(k)
                plt.title(f'PSF kernel epoch {epoch_idx}')
                plt.axis('off')
                plt.tight_layout()
                psf_path = os.path.join(args.checkpoint_dir, f'psf_epoch_{epoch_idx}.png')
                plt.savefig(psf_path, dpi=300)
                plt.close()
                print(f"Saved PSF image -> {psf_path}")

        # Save reconstructions
        if epoch_idx % args.recon_every == 0 or epoch_idx == args.num_epochs:
            try:
                out_recon_dir = os.path.join(args.recon_out_dir, f'epoch_{epoch_idx}')
                save_reconstructions(epoch_idx, model, forward_operator, val_loader, device,
                                     num_examples=args.recon_num_examples, out_dir=out_recon_dir)
            except Exception as e:
                print(f"Failed to save reconstructions at epoch {epoch_idx}: {e}")

    # final loss curves
    mse_curve_path = os.path.join(args.output_dir, 'mse_curve_finetune.png')
    ssim_curve_path = os.path.join(args.output_dir, 'ssim_curve_finetune.png')

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['train'], label='Train MSE')
    plt.plot(loss_history['val'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(mse_curve_path, dpi=300)
    plt.close()
    print(f"Saved MSE curve -> {mse_curve_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['val_ssim'], label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    plt.savefig(ssim_curve_path, dpi=300)
    plt.close()
    print(f"Saved SSIM curve -> {ssim_curve_path}")

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Finetune RIM + Physics Forward operator")

    p.add_argument('--dataset-dir', type=str,
                   default="/home/jwalsh/astropy/Datasets/test_set2",
                   help="Directory with .fits files")
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--augment', action='store_true', help="Apply augmentation during training")

    p.add_argument('--num-epochs', type=int, default=50)
    p.add_argument('--n-iter', type=int, default=10)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--kernel-size', type=int, default=21)

    p.add_argument('--finetune-mode', choices=['both', 'rim', 'forward'], default='both',
                   help="Which modules to finetune (default both)")
    p.add_argument('--freeze-psf', action='store_true', help="Freeze PSF kernel parameters in forward operator")

    p.add_argument('--rim-checkpoint', type=str, default="/home/jwalsh/astropy/imrpoved_rim_mixed_model.pt")
    p.add_argument('--forward-checkpoint', type=str, default="/home/jwalsh/astropy/improved_op_mixed.pt")
    p.add_argument('--resume-checkpoint', type=str, default='', help="Optional full checkpoint to resume optimizer/epoch")

    p.add_argument('--lr-rim', type=float, default=5e-5, help="Learning rate for RIM parameters (lower than initial train)")
    p.add_argument('--lr-forward', type=float, default=5e-6, help="Learning rate for forward operator")
    p.add_argument('--weight-decay', type=float, default=1e-6)

    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-patience', type=int, default=6)

    p.add_argument('--lambda-psf', type=float, default=1e-3)
    p.add_argument('--lambda-lens-prior', type=float, default=1e-4)
    p.add_argument('--lambda-subpix', type=float, default=1e-4)

    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--use-amp', action='store_true', help="Use mixed precision (FP16) when CUDA available")

    p.add_argument('--checkpoint-every', type=int, default=5)
    p.add_argument('--output-dir', type=str, default="/home/jwalsh/astropy/finetuned_model",
                   help="Base output directory for checkpoints, best models, reconstructions and curves")
    p.add_argument('--checkpoint-dir', type=str, default="/home/jwalsh/astropy/betterfinetuned_model")
    p.add_argument('--save-rim-best', type=str, default='rim_finetune_best5.pt')
    p.add_argument('--save-forward-best', type=str, default='forward_finetune_best5.pt')

    p.add_argument('--recon-every', type=int, dest='recon_every', default=3)
    p.add_argument('--recon-num-examples', type=int, default=7)
    p.add_argument('--recon-out-dir', type=str, default="/home/jwalsh/astropy/finetuned_model")

    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)
    finetune(args)
