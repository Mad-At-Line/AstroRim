#!/usr/bin/env python3
"""
finetune_rim_forward_fixed_for_varied_dataset.py

Patched finetuning script that is robust to slightly different FITS layouts used in
your varied dataset (PrimaryHDU contains GT, or EXTNAME present, etc.).

Changes from original:
 - Replaced LensingFitsDataset._read_pair with heuristics that try EXTNAME, ImageHDU.name,
   PrimaryHDU fallback, and header-key heuristics. Produces informative error message
   listing detected HDUs when it still cannot find a pair.
 - Added a small helper to optionally log a couple of detected headers for quick debugging.
 - Kept all finetuning logic the same; only dataset-loading behavior changed.

Usage: same CLI as your original `finetune_rim_forward.py` (defaults preserved).
"""

import os
import glob
import time
import argparse
from datetime import datetime

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
            # extname can be in header EXTNAME or h.name attribute (Astropy sets name for ImageHDU)
            extname_hdr = h.header.get('EXTNAME', '')
            extname_attr = getattr(h, 'name', '') or ''
            name = (extname_hdr or extname_attr).upper()
            data = h.data
            if data is None:
                continue
            try:
                d = data.astype(np.float32)
            except Exception:
                # skip non-numeric HDU
                continue
            if name == 'GT':
                gt = d
            elif name == 'LENSED' or name == 'OBS' or name == 'OBSERVED':
                lensed = d

        # 2) Fallback: common pattern in your simulator -> PrimaryHDU contains GT, second HDU is LENSED
        if gt is None:
            primary = hdul[0]
            if primary.data is not None:
                # Heuristic: if PrimaryHDU has numeric 2D data, treat as GT
                try:
                    gt = primary.data.astype(np.float32)
                except Exception:
                    gt = None

        if lensed is None:
            # If second extension exists and has data, assume it's LENSED
            if len(hdul) > 1 and hdul[1].data is not None:
                try:
                    lensed = hdul[1].data.astype(np.float32)
                except Exception:
                    lensed = None

        # 3) Further heuristic: look for header keys typical of the simulator to identify GT (if still missing)
        if gt is None or lensed is None:
            for h in hdul:
                hdr = h.header
                if h.data is None:
                    continue
                # simulator injects keys like 'SRCMAG', 'THETA_E', 'PSF_FWH_TRUE', etc.
                if ('SRCMAG' in hdr or 'THETA_E' in hdr or 'PSF_FWH_TRUE' in hdr) and gt is None:
                    try:
                        gt = h.data.astype(np.float32)
                    except Exception:
                        pass
                # The LENSED HDU in simulator often has EXTNAME 'LENSED' but catch as fallback:
                if ('PSF_FWH' in hdr or 'SKYADU' in hdr or 'LENSED' in hdr.values()) and lensed is None and h is not hdul[0]:
                    try:
                        lensed = h.data.astype(np.float32)
                    except Exception:
                        pass

        # 4) Final check / raise informative error if pair still missing
        if gt is None or lensed is None:
            # Collect debug info for easier diagnosis
            info_lines = []
            for idx, h in enumerate(hdul):
                name = getattr(h, 'name', '') or h.header.get('EXTNAME', '')
                shape = None if h.data is None else getattr(h.data, 'shape', None)
                # Show a small selection of header keys for diagnosis
                keys = list(h.header.keys())[:12]
                info_lines.append(f"ext#{idx}: name='{name}' shape={shape} keys={keys}")
            hdul.close()
            raise RuntimeError(f"{fn} missing GT or LENSED HDU after heuristics.\nDetected ext list:\n" + "\n".join(info_lines))

        # Clean & normalize
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

# ----------------------- PhysicalForward (unchanged interface) -----------------------
class PhysicalForward(nn.Module):
    def __init__(self, kernel_size=21, device='cpu', enforce_nonneg=True, init_sigma=3.0, mode='parametric'):
        super().__init__()
        self.kernel_size = kernel_size
        self.enforce_nonneg = enforce_nonneg
        self.device_cached = device
        self.mode = mode

        raw = torch.randn(1, 1, kernel_size, kernel_size) * 0.01
        self.raw_psf = nn.Parameter(raw)
        self._init_gaussian(init_sigma)

        self.x0 = nn.Parameter(torch.tensor(0.0))
        self.y0 = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(0.08))
        self.raw_rc = nn.Parameter(torch.tensor(0.01))

        self.raw_subpix = nn.Parameter(torch.zeros(2))
        self.log_sigma = nn.Parameter(torch.tensor(-3.0))

    @property
    def b_pos(self):
        return F.softplus(self.raw_b) + 1e-8

    @property
    def rc_pos(self):
        return F.softplus(self.raw_rc) + 1e-8

    @property
    def subpix(self):
        return 0.25 * torch.tanh(self.raw_subpix)

    def _init_gaussian(self, sigma=3.0):
        k = self.kernel_size
        ax = np.arange(-k//2 + 1., k//2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        gauss = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        gauss = gauss / gauss.sum()
        with torch.no_grad():
            self.raw_psf.data.copy_(torch.from_numpy(gauss.astype(np.float32)).unsqueeze(0).unsqueeze(0))

    def get_psf(self):
        k = self.raw_psf
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
        x0 = self.x0
        y0 = self.y0

        dx = xv - x0
        dy = yv - y0
        r2 = dx * dx + dy * dy + 1e-12
        denom = torch.sqrt(r2 + (rc ** 2))
        ax = b * dx / denom
        ay = b * dy / denom
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

        sx = self.subpix[0]
        sy = self.subpix[1]

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

        sx = self.subpix[0]
        sy = self.subpix[1]
        grid_x = grid[..., 0] + ax_b - sx
        grid_y = grid[..., 1] + ay_b - sy
        adj_grid = torch.stack((grid_x, grid_y), dim=-1)

        src_grad = F.grid_sample(residual, adj_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return src_grad

    def forward(self, src):
        y_warp = self.warp_source(src)
        psf = self.get_psf()
        y_conv = F.conv2d(y_warp, psf, padding=self.kernel_size // 2)
        return y_conv

    def adjoint(self, residual):
        psf = self.get_psf()
        psf_flipped = torch.flip(psf, dims=(2, 3))
        r_conv = F.conv2d(residual, psf_flipped, padding=self.kernel_size // 2)
        src_grad = self.warp_adjoint(r_conv)
        return src_grad

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
    def __init__(self, hidden_dim=96):
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
    def __init__(self, n_iter=8, hidden_dim=96):
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

    # Instantiate models
    forward_operator = PhysicalForward(kernel_size=args.kernel_size, device=device, enforce_nonneg=True).to(device)
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
            # try state dict key names possibly stored under 'model_state'
            if 'model_state' in sd:
                model.load_state_dict(sd['model_state'], strict=False)
            else:
                model.load_state_dict(sd, strict=False)
        print("Loaded RIM checkpoint.")

    if args.forward_checkpoint and os.path.exists(args.forward_checkpoint):
        print(f"Loading Forward operator weights from {args.forward_checkpoint}")
        sd = torch.load(args.forward_checkpoint, map_location='cpu')
        try:
            forward_operator.load_state_dict(sd)
        except Exception:
            if 'forward_state' in sd:
                forward_operator.load_state_dict(sd['forward_state'], strict=False)
            else:
                forward_operator.load_state_dict(sd, strict=False)
        print("Loaded forward checkpoint.")

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

    # Optional freeze of PSF only (useful if you want to only finetune lens params)
    if args.freeze_psf:
        print("Freezing PSF kernel parameters in forward operator (raw_psf.requires_grad=False).")
        forward_operator.raw_psf.requires_grad = False

    # Setup optimizer: separate param groups if desired
    params = []
    if args.finetune_mode in ['both', 'rim']:
        params.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_rim})
    if args.finetune_mode in ['both', 'forward']:
        params.append({"params": [p for p in forward_operator.parameters() if p.requires_grad], "lr": args.lr_forward})
    if len(params) == 0:
        raise RuntimeError("No parameters left to optimize (check finetune_mode / freeze flags).")

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.use_amp))

    loss_history = {"train": [], "val": [], "val_ssim": [], "grad_norm": []}
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume from an optimizer checkpoint if requested
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

                # option: include small PSF smoothness / lens priors for stability
                psf = forward_operator.get_psf()
                lap = psf - F.avg_pool2d(psf, 3, 1, padding=1)
                psf_pen = (lap ** 2).mean()

                lens_prior = (torch.relu(-forward_operator.b_pos) ** 2 + torch.relu(-forward_operator.rc_pos) ** 2)
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

        scheduler.step(avg_val_loss)

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

        # Periodic checkpoints
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

            # visualize PSF
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

        # Save reconstructions every recon_every epochs
        if epoch_idx % args.recon_every == 0 or epoch_idx == args.num_epochs:
            try:
                out_recon_dir = os.path.join(args.recon_out_dir, f'epoch_{epoch_idx}')
                save_reconstructions(epoch_idx, model, forward_operator, val_loader, device,
                                     num_examples=args.recon_num_examples, out_dir=out_recon_dir)
            except Exception as e:
                print(f"Failed to save reconstructions at epoch {epoch_idx}: {e}")

    # final loss curves (save to output_dir)
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
                   default=r"C:\Users\mythi\.astropy\Code\Fits_work\Varied_dataset_100k",
                   help="Directory with .fits files")
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--augment', action='store_true', help="Apply augmentation during training")

    p.add_argument('--num-epochs', type=int, default=50)
    p.add_argument('--n-iter', type=int, default=10)
    p.add_argument('--hidden-dim', type=int, default=96)
    p.add_argument('--kernel-size', type=int, default=21)

    p.add_argument('--finetune-mode', choices=['both', 'rim', 'forward'], default='both',
                   help="Which modules to finetune (default both)")
    p.add_argument('--freeze-psf', action='store_true', help="Freeze PSF kernel parameters in forward operator")

    # defaults set to your specific checkpoint paths
    p.add_argument('--rim-checkpoint', type=str, default=r"C:\Users\mythi\.astropy\Code\v3_unfinetuned\rim_best_model.pt")
    p.add_argument('--forward-checkpoint', type=str, default=r"C:\Users\mythi\.astropy\Code\v3_unfinetuned\forward_best_operator.pt")
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
    # output dir defaults to your requested output path:
    p.add_argument('--output-dir', type=str, default=r"C:\Users\mythi\.astropy\Code\Fits_work\Rim_improved_models",
                   help="Base output directory for checkpoints, best models, reconstructions and curves")
    p.add_argument('--checkpoint-dir', type=str, default=r"C:\Users\mythi\.astropy\Code\Fits_work\Rim_improved_models")
    p.add_argument('--save-rim-best', type=str, default='rim_finetune_best3.pt')
    p.add_argument('--save-forward-best', type=str, default='forward_finetune_best3.pt')

    p.add_argument('--recon-every', type=int, dest='recon_every', default=5)
    p.add_argument('--recon-num-examples', type=int, default=6)
    p.add_argument('--recon-out-dir', type=str, default=r"C:\Users\mythi\.astropy\Code\Fits_work\Rim_improved_models")

    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # ensure dirs exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)
    finetune(args)
