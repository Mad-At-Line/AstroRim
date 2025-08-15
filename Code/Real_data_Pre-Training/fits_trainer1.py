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


class LensingImageFITS(Dataset):
    """
    Reads FITS files and returns (lensed, ground truth) image pairs.
    Uses HDU names 'GT' and 'LENSED' if available, otherwise falls back to first two image HDUs.
    Detection is done once using the first file for efficiency.
    """
    def __init__(self, file_list, gt_name="GT", lensed_name="LENSED"):
        self.files = file_list
        assert len(self.files) > 0, "No FITS files provided."

        # Detect HDU lookup strategy from the first file
        self.use_names = False
        self.gt_hdu = None
        self.lensed_hdu = None

        sample_file = self.files[0]
        with fits.open(sample_file, memmap=False) as hdul:
            names = {hdu.name.upper(): i for i, hdu in enumerate(hdul) if hasattr(hdu, "name")}
            image_hdus = [i for i, hdu in enumerate(hdul) if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) >= 2]

            if gt_name.upper() in names and lensed_name.upper() in names:
                self.use_names = True
                self.gt_hdu = gt_name.upper()
                self.lensed_hdu = lensed_name.upper()
            elif len(image_hdus) >= 2:
                self.use_names = False
                self.gt_hdu, self.lensed_hdu = image_hdus[:2]
            else:
                raise RuntimeError("Could not find required GT/LENSED data in FITS file.")

    def __len__(self):
        return len(self.files)

    def _read_pair_from_file(self, file_path):
        with fits.open(file_path, memmap=False) as hdul:
            gt_hdu = hdul[self.gt_hdu]
            lensed_hdu = hdul[self.lensed_hdu]

            gt_data = np.array(gt_hdu.data, dtype=np.float32)
            lensed_data = np.array(lensed_hdu.data, dtype=np.float32)

        # Clean NaNs and normalize
        gt_data = np.nan_to_num(gt_data)
        lensed_data = np.nan_to_num(lensed_data)

        gt_data = np.squeeze(gt_data)
        lensed_data = np.squeeze(lensed_data)

        gt_max = np.max(gt_data) or 1.0
        lensed_max = np.max(lensed_data) or 1.0
        gt_data /= gt_max
        lensed_data /= lensed_max

        return lensed_data.astype(np.float32), gt_data.astype(np.float32)

    def __getitem__(self, idx):
        lensed, gt = self._read_pair_from_file(self.files[idx])
        return torch.tensor(lensed).unsqueeze(0), torch.tensor(gt).unsqueeze(0)


class DifferentiableLensing(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class RIMCell(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
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
    def __init__(self, n_iter=12, hidden_dim=96):  # n_iter set to 12
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
            # Ensure x is fresh and tracked
            x = x.detach().clone().requires_grad_(True)
            y_sim = forward_operator(x)
            loss = F.mse_loss(y_sim, y)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        
            h = self.cell(x, grad, h)
            x = x + self.final_conv(h)
        return x


def train_rim():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_dir = r"C:\Users\mythi\.astropy\Code\Fits_work\4.5_fits_dataset1"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.fits")))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # Detect HDU strategy from training set, reuse for validation
    train_dataset = LensingImageFITS(train_files)
    val_dataset = LensingImageFITS(val_files, gt_name=train_dataset.gt_hdu if train_dataset.use_names else "GT")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    forward_operator = DifferentiableLensing().to(device)
    model = RIM(n_iter=12).to(device)  # changed here to 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)

    best_val_loss = float('inf')
    loss_history = {"train": [], "val": [], "grad_norm": [], "val_ssim": []}
    patience = 15
    epochs_no_improve = 0

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        grad_norm_total = 0.0
        for obs, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs, gt = obs.to(device), gt.to(device)
            optimizer.zero_grad()
            recon = model(obs, forward_operator)
            loss = F.mse_loss(recon, gt)
            loss.backward()

            grad_norm_total += sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tightened clipping to 1.0
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = grad_norm_total / len(train_loader)
        loss_history["train"].append(avg_train_loss)
        loss_history["grad_norm"].append(avg_grad_norm)

        # Validation
        model.eval()
        val_loss = 0.0
        val_ssim_total = 0
        val_image_count = 0
        for obs, gt in val_loader:
            obs, gt = obs.to(device), gt.to(device)
            recon = model(obs, forward_operator).detach()
            val_loss += F.mse_loss(recon, gt).item() * obs.size(0)
            for i in range(recon.size(0)):
                val_ssim_total += ssim(gt[i, 0].cpu().numpy(), recon[i, 0].cpu().numpy(), data_range=1.0)
            val_image_count += recon.size(0)

        avg_val_loss = val_loss / val_image_count
        avg_val_ssim = val_ssim_total / val_image_count
        loss_history["val"].append(avg_val_loss)
        loss_history["val_ssim"].append(avg_val_ssim)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | SSIM: {avg_val_ssim:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "rim_best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

        if (epoch + 1) % 5 == 0:
            model.eval()
            plt.figure(figsize=(15, 12))
            for i in range(min(4, len(val_dataset))):
                obs, gt = val_dataset[i]
                obs_b = obs.unsqueeze(0).to(device)
                recon = model(obs_b, forward_operator).detach().cpu().squeeze()
                obs_img = obs.cpu().squeeze()
                gt_img = gt.squeeze()

                plt.subplot(4, 3, i*3+1)
                plt.imshow(obs_img, cmap='gray', origin='lower')
                plt.title(f"Lensed {i+1}")
                plt.axis('off')

                plt.subplot(4, 3, i*3+2)
                plt.imshow(gt_img, cmap='gray', origin='lower')
                plt.title(f"True {i+1}")
                plt.axis('off')

                plt.subplot(4, 3, i*3+3)
                plt.imshow(recon, cmap='gray', origin='lower')
                plt.title(f"Recon {i+1}")
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"recon_epoch_{epoch+1}.png", dpi=300)
            plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history["train"], label='Train Loss')
    plt.plot(loss_history["val"], label='Val Loss')
    plt.plot(loss_history["val_ssim"], label='SSIM')
    plt.plot(loss_history["grad_norm"], label='Grad Norm')
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    train_rim()
    print(f"Elapsed: {(time.time() - start)/3600:.2f}h")
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


class LensingImageFITS(Dataset):
    """
    Reads FITS files and returns (lensed, ground truth) image pairs.
    Uses HDU names 'GT' and 'LENSED' if available, otherwise falls back to first two image HDUs.
    Detection is done once using the first file for efficiency.
    """
    def __init__(self, file_list, gt_name="GT", lensed_name="LENSED"):
        self.files = file_list
        assert len(self.files) > 0, "No FITS files provided."

        # Detect HDU lookup strategy from the first file
        self.use_names = False
        self.gt_hdu = None
        self.lensed_hdu = None

        sample_file = self.files[0]
        with fits.open(sample_file, memmap=False) as hdul:
            names = {hdu.name.upper(): i for i, hdu in enumerate(hdul) if hasattr(hdu, "name")}
            image_hdus = [i for i, hdu in enumerate(hdul) if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) >= 2]

            if gt_name.upper() in names and lensed_name.upper() in names:
                self.use_names = True
                self.gt_hdu = gt_name.upper()
                self.lensed_hdu = lensed_name.upper()
            elif len(image_hdus) >= 2:
                self.use_names = False
                self.gt_hdu, self.lensed_hdu = image_hdus[:2]
            else:
                raise RuntimeError("Could not find required GT/LENSED data in FITS file.")

    def __len__(self):
        return len(self.files)

    def _read_pair_from_file(self, file_path):
        with fits.open(file_path, memmap=False) as hdul:
            gt_hdu = hdul[self.gt_hdu]
            lensed_hdu = hdul[self.lensed_hdu]

            gt_data = np.array(gt_hdu.data, dtype=np.float32)
            lensed_data = np.array(lensed_hdu.data, dtype=np.float32)

        # Clean NaNs and normalize
        gt_data = np.nan_to_num(gt_data)
        lensed_data = np.nan_to_num(lensed_data)

        gt_data = np.squeeze(gt_data)
        lensed_data = np.squeeze(lensed_data)

        gt_max = np.max(gt_data) or 1.0
        lensed_max = np.max(lensed_data) or 1.0
        gt_data /= gt_max
        lensed_data /= lensed_max

        return lensed_data.astype(np.float32), gt_data.astype(np.float32)

    def __getitem__(self, idx):
        lensed, gt = self._read_pair_from_file(self.files[idx])
        return torch.tensor(lensed).unsqueeze(0), torch.tensor(gt).unsqueeze(0)


class DifferentiableLensing(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class RIMCell(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
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
    def __init__(self, n_iter=12, hidden_dim=96):  # n_iter set to 12
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
            # Ensure x is fresh and tracked
            x = x.detach().clone().requires_grad_(True)
            y_sim = forward_operator(x)
            loss = F.mse_loss(y_sim, y)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        
            h = self.cell(x, grad, h)
            x = x + self.final_conv(h)
        return x


def train_rim():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_dir = r"C:\Users\mythi\.astropy\Code\Fits_work\4.5_fits_dataset1"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.fits")))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # Detect HDU strategy from training set, reuse for validation
    train_dataset = LensingImageFITS(train_files)
    val_dataset = LensingImageFITS(val_files, gt_name=train_dataset.gt_hdu if train_dataset.use_names else "GT")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    forward_operator = DifferentiableLensing().to(device)
    model = RIM(n_iter=12).to(device)  # changed here to 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_val_loss = float('inf')
    loss_history = {"train": [], "val": [], "grad_norm": [], "val_ssim": []}
    patience = 15
    epochs_no_improve = 0

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        grad_norm_total = 0.0
        for obs, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs, gt = obs.to(device), gt.to(device)
            optimizer.zero_grad()
            recon = model(obs, forward_operator)
            loss = F.mse_loss(recon, gt)
            loss.backward()

            grad_norm_total += sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tightened clipping to 1.0
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = grad_norm_total / len(train_loader)
        loss_history["train"].append(avg_train_loss)
        loss_history["grad_norm"].append(avg_grad_norm)

        # Validation
        model.eval()
        val_loss = 0.0
        val_ssim_total = 0
        val_image_count = 0
        for obs, gt in val_loader:
            obs, gt = obs.to(device), gt.to(device)
            recon = model(obs, forward_operator).detach()
            val_loss += F.mse_loss(recon, gt).item() * obs.size(0)
            for i in range(recon.size(0)):
                val_ssim_total += ssim(gt[i, 0].cpu().numpy(), recon[i, 0].cpu().numpy(), data_range=1.0)
            val_image_count += recon.size(0)

        avg_val_loss = val_loss / val_image_count
        avg_val_ssim = val_ssim_total / val_image_count
        loss_history["val"].append(avg_val_loss)
        loss_history["val_ssim"].append(avg_val_ssim)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | SSIM: {avg_val_ssim:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "rim_best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

        if (epoch + 1) % 5 == 0:
            model.eval()
            plt.figure(figsize=(15, 12))
            for i in range(min(4, len(val_dataset))):
                obs, gt = val_dataset[i]
                obs_b = obs.unsqueeze(0).to(device)
                recon = model(obs_b, forward_operator).detach().cpu().squeeze()
                obs_img = obs.cpu().squeeze()
                gt_img = gt.squeeze()

                plt.subplot(4, 3, i*3+1)
                plt.imshow(obs_img, cmap='gray', origin='lower')
                plt.title(f"Lensed {i+1}")
                plt.axis('off')

                plt.subplot(4, 3, i*3+2)
                plt.imshow(gt_img, cmap='gray', origin='lower')
                plt.title(f"True {i+1}")
                plt.axis('off')

                plt.subplot(4, 3, i*3+3)
                plt.imshow(recon, cmap='gray', origin='lower')
                plt.title(f"Recon {i+1}")
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"recon_epoch_{epoch+1}.png", dpi=300)
            plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history["train"], label='Train Loss')
    plt.plot(loss_history["val"], label='Val Loss')
    plt.plot(loss_history["val_ssim"], label='SSIM')
    plt.plot(loss_history["grad_norm"], label='Grad Norm')
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    train_rim()
    print(f"Elapsed: {(time.time() - start)/3600:.2f}h")
