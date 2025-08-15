import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from astropy.io import fits
from skimage.metrics import structural_similarity as ssim

# =====================
# Dataset loader (FITS)
# =====================
class LensingImageFITS(Dataset):
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
            image_hdus = [
                i for i, hdu in enumerate(hdul)
                if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) >= 2
            ]
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


# =====================
# Model definitions
# =====================
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
    def __init__(self, hidden_dim=96):
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
        candidate = torch.tanh(
            self.conv_candidate(torch.cat([x, grad, reset_gate * h], dim=1))
        )
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
            x = x.detach().clone().requires_grad_(True)
            y_sim = forward_operator(x)
            loss = F.mse_loss(y_sim, y)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            h = self.cell(x, grad, h)
            x = x + self.final_conv(h)

        return x


# =====================
# Inference function
# =====================
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config paths
    model_path = r"C:\Users\mythi\.astropy\Code\Fits_work\fits4.5_modelholder1\rim_best_model.pt"
    dataset_dir = r"C:\Users\mythi\.astropy\Code\Fits_work\fits_unseen4.5"
    output_dir = r"C:\Users\mythi\.astropy\Code\Fits_work\fits_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    fits_files = sorted(glob.glob(os.path.join(dataset_dir, "*.fits")))
    dataset = LensingImageFITS(fits_files)
    print(f"Loaded {len(dataset)} FITS samples.")

    # Init models
    forward_operator = DifferentiableLensing().to(device)
    model = RIM(n_iter=12, hidden_dim=96).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # IMPORTANT: Set forward_operator to training mode to enable gradients
    forward_operator.train()
    
    print(f"Loaded model from {model_path}")

    # Metrics
    ssim_values = []
    mse_values = []
    num_samples_to_visualize = 8

    # Remove torch.no_grad() context since we need gradients for the RIM model
    for idx, (lensed, gt) in enumerate(DataLoader(dataset, batch_size=1)):
        lensed, gt = lensed.to(device), gt.to(device)
        recon = model(lensed, forward_operator)

        # Convert to numpy for metrics (detach from computation graph)
        lensed_np = lensed.detach().squeeze().cpu().numpy()
        gt_np = gt.detach().squeeze().cpu().numpy()
        recon_np = recon.detach().squeeze().cpu().numpy()

        # Metrics
        cur_ssim = ssim(gt_np, recon_np, data_range=1.0)
        cur_mse = np.mean((gt_np - recon_np) ** 2)
        ssim_values.append(cur_ssim)
        mse_values.append(cur_mse)

        # Save images
        if idx < num_samples_to_visualize:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(lensed_np, cmap='gray', origin='lower')
            axes[0].set_title("Lensed")
            axes[1].imshow(gt_np, cmap='gray', origin='lower')
            axes[1].set_title("Ground Truth")
            axes[2].imshow(recon_np, cmap='gray', origin='lower')
            axes[2].set_title(f"Reconstructed\nSSIM: {cur_ssim:.4f}\nMSE: {cur_mse:.6f}")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{idx+1:03d}.png"), dpi=150)
            plt.close()

    # Summary
    avg_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    avg_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)

    print("\n=== Summary ===")
    print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")

    # Save metrics
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
        f.write(f"Dataset: {dataset_dir}\nModel: {model_path}\n")
        f.write(f"Samples: {len(dataset)}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}\n")

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    run_inference()