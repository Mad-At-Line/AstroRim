#!/usr/bin/env python3
"""
rim_train_for_two_hdu_fits.py

Training script adapted to the TWO-HDU FITS simulator that writes:
  - Primary HDU (metadata)
  - IMAGE HDU named "GT"     -> unlensed, detector-sampled (ADU)
  - IMAGE HDU named "LENSED" -> lensed + lens light + PSF + noise (ADU)

Features:
 - Dataset reads GT/LENSED and per-file headers (EXPTIME, ZP, PIXSCALE, PSF_FWH, OVERSAMP)
 - Several normalization modes (per-image, adu_per_sec, global)
 - RIM model and training loop based on your original script
"""

import os
import glob
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from astropy.io import fits
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

# -----------------------
# User settings / knobs
# -----------------------
DATASET_DIR = r"C:\Users\mythi\.astropy\Code\Fits_work\fits_dataset_5.0"  # where simulator writes files
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 4e-4
WEIGHT_DECAY = 1e-4
MODEL_SAVE_PATH = "rim_best_model.pt"
NORMALIZE_MODE = "per_image"   # "per_image", "adu_per_sec", or "global"
GLOBAL_MAX = 1.0               # used only if NORMALIZE_MODE == "global"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_EVERY = 5                 # save recon plots every PLOT_EVERY epochs
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------
# Dataset
# -----------------------
class LensingFITSDataset(Dataset):
    """
    Reads FITS files produced by the simulator.
    Expects TWO image HDUs named 'GT' and 'LENSED'. Reads important header keywords.
    Returns:
      obs_tensor: (1, H, W) float32 (normalized)
      gt_tensor:  (1, H, W) float32 (normalized)
      meta: dict with header metadata (optional use)
    """
    def __init__(self, file_list, normalize_mode="per_image", global_max=1.0):
        assert len(file_list) > 0, "No FITS files provided to dataset."
        self.files = sorted(file_list)
        self.normalize_mode = normalize_mode
        self.global_max = float(global_max)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _safe_read_hdu(hdul, name_or_index):
        """
        Return HDU object given name or index; raise informative error if missing.
        """
        try:
            return hdul[name_or_index]
        except Exception as e:
            raise RuntimeError(f"Failed to read HDU '{name_or_index}': {e}")

    def _normalize_pair(self, lensed, gt, header):
        # Ensure float32 and no NaNs
        lensed = np.nan_to_num(lensed).astype(np.float32)
        gt = np.nan_to_num(gt).astype(np.float32)

        if self.normalize_mode == "per_image":
            # Normalize each image independently to its max (keeps structure)
            gt_max = float(np.max(gt)) if np.max(gt) > 0 else 1.0
            lensed_max = float(np.max(lensed)) if np.max(lensed) > 0 else 1.0
            gt_norm = gt / gt_max
            lensed_norm = lensed / lensed_max
            meta_scale = {"gt_scale": gt_max, "lensed_scale": lensed_max}
            return lensed_norm, gt_norm, meta_scale

        elif self.normalize_mode == "adu_per_sec":
            # Convert ADU -> ADU / sec by dividing by EXPTIME if available
            exptime = header.get("EXPTIME", None)
            if exptime is None or exptime <= 0:
                # fallback to per-image normalization
                return self._normalize_pair(lensed, gt, header={})
            gt_norm = gt.astype(np.float32) / float(exptime)
            lensed_norm = lensed.astype(np.float32) / float(exptime)
            # scale both to their joint max to keep values in a reasonable range
            joint_max = max(gt_norm.max(), lensed_norm.max(), 1.0)
            return (lensed_norm / joint_max).astype(np.float32), (gt_norm / joint_max).astype(np.float32), {"scale_factor": joint_max}

        elif self.normalize_mode == "global":
            # divide by a user-provided global max
            gm = max(self.global_max, 1e-12)
            return (lensed / gm).astype(np.float32), (gt / gm).astype(np.float32), {"global_max": gm}

        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with fits.open(file_path, memmap=False) as hdul:
            # Expect GT and LENSED image HDUs
            # Primary may be 0, then GT=1, LENSED=2; or explicit names
            # Prefer names 'GT' and 'LENSED' if present
            names = [hdu.name.upper() for hdu in hdul if hasattr(hdu, "name")]
            if "GT" in names and "LENSED" in names:
                gt_hdu = hdul[names.index("GT") + (0 if hdul[0].name == '' else 0)]
                # safer: use hdul['GT']
                gt_hdu = hdul['GT']
                lensed_hdu = hdul['LENSED']
            else:
                # fallback to first two image HDUs (skip primary)
                image_hdus = [h for h in hdul if getattr(h, "data", None) is not None]
                if len(image_hdus) >= 2:
                    gt_hdu = image_hdus[0]
                    lensed_hdu = image_hdus[1]
                else:
                    raise RuntimeError(f"File {file_path} does not contain two image HDUs.")

            gt_data = np.array(gt_hdu.data, dtype=np.float32)
            lensed_data = np.array(lensed_hdu.data, dtype=np.float32)
            header = hdul[0].header.copy()
            # allow also header in the image HDU to override:
            header.update(gt_hdu.header)

        # squeeze any singleton dims (should be 2D images)
        gt_data = np.squeeze(gt_data)
        lensed_data = np.squeeze(lensed_data)

        # normalize according to selected mode
        lensed_norm, gt_norm, meta = self._normalize_pair(lensed_data, gt_data, header)

        # convert to torch tensors shape (1, H, W)
        obs_t = torch.from_numpy(lensed_norm).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_norm).unsqueeze(0).float()

        meta_out = {
            "file": file_path,
            "header": {k: header.get(k) for k in ("EXPTIME", "ZP", "PIXSCALE", "PSF_FWH", "OVERSAMP")}
        }
        meta_out.update(meta)
        return obs_t, gt_t, meta_out


# -----------------------
# Small differentiable forward operator (learnable surrogate)
# -----------------------
class DifferentiableLensing(nn.Module):
    """
    A small CNN that acts as a differentiable forward operator (learnable surrogate).
    Optionally can be initialized to a mild Gaussian blur kernel (helps when the
    simulator mainly differs by PSF + additive lens light).
    """
    def __init__(self, init_blur=False, kernel_size=9):
        super().__init__()
        # A few conv layers; keep capacity moderate so RIM can learn physics
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(48, 48, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(48, 1, 3, padding=1)
        )
        if init_blur:
            # Try to bias the last conv to a small blur kernel (approx PSF)
            with torch.no_grad():
                # create gaussian kernel
                k = kernel_size
                ax = np.arange(-k//2 + 1., k//2 + 1.)
                xx, yy = np.meshgrid(ax, ax)
                sigma = kernel_size / 6.0
                kern = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
                kern = kern / kern.sum()
                # Set last conv weights to implement blur (shape out_ch,in_ch,k,k)
                w = self.net[-1].weight  # shape (1,48,k,k) if sizes match; fallback to small scale
                if w.shape[2] == k:
                    for i in range(w.shape[1]):
                        w[0, i] = torch.from_numpy(kern.astype(np.float32))
                # small bias
                self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)


# -----------------------
# RIM cell / model (unchanged structure but tidied)
# -----------------------
class RIMCell(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # input channels: x (1) + grad (1) + h (hidden_dim)
        self.conv_gate = nn.Sequential(
            nn.Conv2d(1 + 1 + hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, 3, padding=1)
        )
        self.conv_candidate = nn.Sequential(
            nn.Conv2d(1 + 1 + hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )

    def forward(self, x, grad, h):
        gates = torch.sigmoid(self.conv_gate(torch.cat([x, grad, h], dim=1)))
        update_gate, reset_gate = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.conv_candidate(torch.cat([x, grad, reset_gate * h], dim=1)))
        return (1 - update_gate) * h + update_gate * candidate


class RIM(nn.Module):
    def __init__(self, n_iter=12, hidden_dim=96):
        super().__init__()
        self.n_iter = n_iter
        self.hidden_dim = hidden_dim
        self.cell = RIMCell(hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, y, forward_operator):
        B, C, H, W = y.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=y.device)
        x = y.clone()
        for _ in range(self.n_iter):
            # ensure x is detached from previous graph and requires grad for implicit gradient
            x = x.detach().clone().requires_grad_(True)
            y_sim = forward_operator(x)
            loss = F.mse_loss(y_sim, y)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            h = self.cell(x, grad, h)
            x = x + self.final_conv(h)
        return x


# -----------------------
# Training loop
# -----------------------
def train_rim(dataset_dir=DATASET_DIR):
    print(f"Using device: {DEVICE}")

    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.fits")))
    if len(all_files) == 0:
        raise RuntimeError(f"No .fits files found in {dataset_dir}")

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=SEED)
    train_ds = LensingFITSDataset(train_files, normalize_mode=NORMALIZE_MODE, global_max=GLOBAL_MAX)
    val_ds = LensingFITSDataset(val_files, normalize_mode=NORMALIZE_MODE, global_max=GLOBAL_MAX)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=_collate_fn)

    forward_operator = DifferentiableLensing(init_blur=True).to(DEVICE)
    model = RIM(n_iter=12, hidden_dim=96).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(forward_operator.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    loss_history = {"train": [], "val": [], "val_ssim": [], "grad_norm": []}
    patience = 15
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        forward_operator.train()
        train_loss = 0.0
        grad_norm_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for obs_batch, gt_batch, metas in pbar:
            obs = obs_batch.to(DEVICE)
            gt = gt_batch.to(DEVICE)

            optimizer.zero_grad()
            recon = model(obs, forward_operator)
            loss = F.mse_loss(recon, gt)
            loss.backward()

            # gradient norm logging
            grad_norm_total += sum(p.grad.norm(2).item() for p in (list(model.parameters()) + list(forward_operator.parameters())) if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(forward_operator.parameters()), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = grad_norm_total / len(train_loader)
        loss_history["train"].append(avg_train_loss)
        loss_history["grad_norm"].append(avg_grad_norm)

        # Validation
        model.eval()
        forward_operator.eval()
        val_loss = 0.0
        val_image_count = 0
        val_ssim_total = 0.0
        with torch.no_grad():
            for obs_batch, gt_batch, metas in val_loader:
                obs = obs_batch.to(DEVICE)
                gt = gt_batch.to(DEVICE)
                recon = model(obs, forward_operator).detach()
                # accumulate
                batch_size = obs.size(0)
                val_loss += F.mse_loss(recon, gt, reduction='mean').item() * batch_size
                # compute SSIM per image with correct data_range
                for i in range(batch_size):
                    gt_np = gt[i, 0].cpu().numpy()
                    recon_np = recon[i, 0].cpu().numpy()
                    data_range = float(gt_np.max() - gt_np.min()) if (gt_np.max() - gt_np.min()) > 0 else 1.0
                    val_ssim_total += ssim(gt_np, recon_np, data_range=data_range)
                val_image_count += batch_size

        avg_val_loss = val_loss / val_image_count
        avg_val_ssim = val_ssim_total / val_image_count
        loss_history["val"].append(avg_val_loss)
        loss_history["val_ssim"].append(avg_val_ssim)

        print(f"[Epoch {epoch+1}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | SSIM: {avg_val_ssim:.4f}")

        # checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                "rim_state": model.state_dict(),
                "forward_state": forward_operator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "loss_history": loss_history
            }, MODEL_SAVE_PATH)
            print(f"Saved best model -> {MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

        # occasional recon plots (save some examples from validation dataset)
        if (epoch + 1) % PLOT_EVERY == 0:
            _save_recon_examples(model, forward_operator, val_ds, n_examples=4, epoch=epoch+1)

    # final plotting of metrics
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history["train"], label="Train Loss")
    plt.plot(loss_history["val"], label="Val Loss")
    plt.plot(loss_history["val_ssim"], label="Val SSIM")
    plt.plot(loss_history["grad_norm"], label="Grad Norm")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics.png", dpi=300)
    plt.show()


# -----------------------
# Helpers
# -----------------------
def _collate_fn(batch):
    """
    Collate because our dataset __getitem__ returns (obs, gt, meta)
    """
    obs = torch.stack([b[0] for b in batch], dim=0)
    gt = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return obs, gt, metas


def _save_recon_examples(rim_model, forward_operator, val_dataset, n_examples=4, epoch=0):
    rim_model.eval()
    forward_operator.eval()
    n_examples = min(n_examples, len(val_dataset))
    plt.figure(figsize=(12, 3 * n_examples))
    for i in range(n_examples):
        obs, gt, meta = val_dataset[i]
        with torch.no_grad():
            recon = rim_model(obs.unsqueeze(0).to(DEVICE), forward_operator).detach().cpu().squeeze().numpy()
        obs_np = obs.squeeze().numpy()
        gt_np = gt.squeeze().numpy()
        # plotting rows
        plt.subplot(n_examples, 3, i*3 + 1)
        plt.imshow(obs_np, origin='lower', cmap='gray')
        plt.title(f"Obs {i}")
        plt.axis('off')

        plt.subplot(n_examples, 3, i*3 + 2)
        plt.imshow(gt_np, origin='lower', cmap='gray')
        plt.title("GT")
        plt.axis('off')

        plt.subplot(n_examples, 3, i*3 + 3)
        plt.imshow(recon, origin='lower', cmap='gray')
        plt.title("Recon")
        plt.axis('off')

    plt.tight_layout()
    fname = f"recon_epoch_{epoch}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved recon examples to {fname}")


# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    start = time.time()
    train_rim(DATASET_DIR)
    print(f"Elapsed: {(time.time() - start)/3600:.2f}h")
