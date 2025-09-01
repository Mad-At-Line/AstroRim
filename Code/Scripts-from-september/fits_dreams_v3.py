#Works with new sims I think
import os
import glob
import time
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

        # 1) Try to find by EXTNAME / h.name (explicit)
        for i, h in enumerate(hdul):
            # extname can be in header EXTNAME or h.name attribute (Astropy sets name for ImageHDU)
            extname_hdr = h.header.get('EXTNAME', '')
            extname_attr = getattr(h, 'name', '') or ''
            name = (extname_hdr or extname_attr).upper()
            data = h.data
            if data is None:
                continue
            if name == 'GT':
                gt = data.astype(np.float32)
            elif name == 'LENSED':
                lensed = data.astype(np.float32)

        # 2) Fallback: common pattern in your simulator -> PrimaryHDU contains GT, second HDU is LENSED
        if gt is None:
            primary = hdul[0]
            if primary.data is not None:
                # Heuristic: if PrimaryHDU has numeric 2D data, treat as GT
                gt = primary.data.astype(np.float32)

        if lensed is None:
            # If second extension exists and has data, assume it's LENSED
            if len(hdul) > 1 and hdul[1].data is not None:
                lensed = hdul[1].data.astype(np.float32)

        # 3) Further heuristic: look for header keys typical of the simulator to identify GT (if still missing)
        if gt is None or lensed is None:
            for h in hdul:
                hdr = h.header
                # simulator injects keys like 'SRCMAG', 'THETA_E', 'PSF_FWH_TRUE', etc.
                if ('SRCMAG' in hdr or 'THETA_E' in hdr or 'PSF_FWH_TRUE' in hdr) and h.data is not None:
                    # prefer to set GT from primary-like headers
                    if gt is None:
                        gt = h.data.astype(np.float32)
                # The LENSED HDU in simulator often has EXTNAME 'LENSED' but catch as fallback:
                if ('PSF_FWH' in hdr or 'SKYADU' in hdr) and h.data is not None:
                    if lensed is None and h is not hdul[0]:
                        lensed = h.data.astype(np.float32)

        # 4) Final check / raise informative error if pair still missing
        if gt is None or lensed is None:
            # Collect debug info for easier diagnosis
            info_lines = []
            for idx, h in enumerate(hdul):
                name = getattr(h, 'name', '') or h.header.get('EXTNAME', '')
                info_lines.append(f"ext#{idx}: name='{name}' shape={None if h.data is None else h.data.shape} keys={list(h.header.keys())[:8]}")
            hdul.close()
            raise RuntimeError(f"{fn} missing GT or LENSED HDU after heuristics.\nDetected ext list:\n" + "\n".join(info_lines))

        # Clean & normalize
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        eps = 1e-8
        m = max(gt.max(), lensed.max(), eps)
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


# ----------------------- Physics-informed forward operator -----------------------
class PhysicalForward(nn.Module):
    """Composite forward operator:
    y = D( PSF( Warp(source; lens_params) ) )
    ...
    (unchanged - same as your original)
    """

    def __init__(self, kernel_size=21, device='cpu', enforce_nonneg=True, init_sigma=3.0,
                 mode='parametric'):
        super().__init__()
        self.kernel_size = kernel_size
        self.enforce_nonneg = enforce_nonneg
        self.device_cached = device
        self.mode = mode

        # PSF parameter (raw) - we'll normalise in get_psf()
        raw = torch.randn(1, 1, kernel_size, kernel_size) * 0.01
        self.raw_psf = nn.Parameter(raw)
        self._init_gaussian(init_sigma)

        # Parametric lens parameters (we store raw params and expose positive versions)
        self.x0 = nn.Parameter(torch.tensor(0.0))
        self.y0 = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(0.08))   # raw -> positive via softplus
        self.raw_rc = nn.Parameter(torch.tensor(0.01))  # raw -> positive

        # small learnable sub-pixel shift per image (raw), constrained via tanh
        self.raw_subpix = nn.Parameter(torch.zeros(2))  # dx, dy in normalized coords (raw)

        # Learnable log-sigma for gaussian noise (stabilises training)
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
        grid = torch.stack((xv, yv), dim=-1)  # (H,W,2)
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
        try:
            y_sim = F.grid_sample(src, samp_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        except Exception as e:
            raise RuntimeError("grid_sample failed. Possible cudnn incompatibility. "
                               "Try setting num_workers=0 and running on CPU to debug. Original error: " + str(e))
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
        try:
            src_grad = F.grid_sample(residual, adj_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        except Exception as e:
            raise RuntimeError("grid_sample failed in adjoint. See notes. Error: " + str(e))
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


# ----------------------- RIM improvements -----------------------
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


# ----------------------- Training loop -----------------------
def train_rim():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- Config ----------
    batch_size = 12
    num_epochs = 200
    learning_rate = 4e-4
    forward_lr = 5e-5
    weight_decay = 1e-6
    kernel_size = 21
    n_iter = 10
    patience = 20
    augment = True
    checkpoint_every = 5
    recon_every = 5
    recon_num_examples = 6
    recon_out_dir = 'recons'
    pin_memory = True if device.type == 'cuda' else False
    num_workers = 0
    grad_clip = 5.0
    use_amp = True if device.type == 'cuda' else False
    lambda_psf = 1e-3
    lambda_lens_prior = 1e-3
    lambda_subpix = 1e-3
    # ----------------------------

    dataset_dir = r"C:\Users\mythi\.astropy\Code\Fits_work\Varied_dataset_100k"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.fits")))
    if len(all_files) == 0:
        raise RuntimeError(f"No .fits files found in {dataset_dir}")
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = LensingFitsDataset(train_files, augment=augment)
    val_dataset = LensingFitsDataset(val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    forward_operator = PhysicalForward(kernel_size=kernel_size, device=device, enforce_nonneg=True).to(device)
    model = RIMImproved(n_iter=n_iter, hidden_dim=96).to(device)

    opt_params = [
        {"params": model.parameters(), "lr": learning_rate},
        {"params": forward_operator.parameters(), "lr": forward_lr},
    ]
    optimizer = torch.optim.AdamW(opt_params, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    loss_history = {"train": [], "val": [], "val_ssim": [], "grad_norm": []}

    os.makedirs(recon_out_dir, exist_ok=True)

    for epoch in range(num_epochs):
        start_epoch_time = time.time()
        model.train()
        forward_operator.train()
        running_loss = 0.0
        grad_norm_epoch = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for obs, gt in loop:
            obs = obs.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                recon = model(obs, forward_operator)
                sigma = torch.exp(forward_operator.log_sigma)
                mse = F.mse_loss(recon, gt, reduction='mean')
                psf = forward_operator.get_psf()
                lap = psf - F.avg_pool2d(psf, 3, 1, padding=1)
                psf_pen = (lap ** 2).mean()
                lens_prior = (torch.relu(-forward_operator.b_pos) ** 2 + torch.relu(-forward_operator.rc_pos) ** 2)
                subpix_pen = (forward_operator.subpix ** 2).mean()
                loss = mse + lambda_psf * psf_pen + lambda_lens_prior * lens_prior + lambda_subpix * subpix_pen

            scaler.scale(loss).backward()

            if use_amp:
                scaler.unscale_(optimizer)

            grad_norm = compute_grad_norm(list(model.parameters()) + list(forward_operator.parameters()))
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(forward_operator.parameters()), max_norm=grad_clip)
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
        print(f"[Epoch {epoch+1}] Train: {avg_train_loss:.6e} | Val: {avg_val_loss:.6e} | SSIM: {avg_val_ssim:.4f} | GradNorm: {avg_grad_norm:.3f} | Time: {epoch_time:.1f}s")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "rim_best_model.pt")
            torch.save(forward_operator.state_dict(), "forward_best_operator.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # Checkpoints + PSF visualization
        if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'forward_state': forward_operator.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"checkpoint_epoch_{epoch+1}.pt")

            with torch.no_grad():
                k = forward_operator.get_psf().cpu().numpy()[0, 0]
                plt.figure(figsize=(4, 4))
                plt.imshow(k, cmap='viridis')
                plt.title(f'PSF kernel epoch {epoch+1}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'psf_epoch_{epoch+1}.png', dpi=300)
                plt.close()

        # Save reconstructions every recon_every epochs
        if (epoch + 1) % recon_every == 0 or epoch == num_epochs - 1:
            try:
                save_reconstructions(epoch + 1, model, forward_operator, val_loader, device,
                                     num_examples=recon_num_examples, out_dir=recon_out_dir)
            except Exception as e:
                print(f"Failed to save reconstructions at epoch {epoch+1}: {e}")

    # final loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['train'], label='Train MSE')
    plt.plot(loss_history['val'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('mse_curve.png', dpi=300)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['val_ssim'], label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    plt.savefig('ssim_curve.png', dpi=300)


if __name__ == '__main__':
    start = time.time()
    train_rim()
    end = time.time()
    elapsed = (end - start) / 3600.0
    print(f"Training finished in {elapsed:.2f} hours")
