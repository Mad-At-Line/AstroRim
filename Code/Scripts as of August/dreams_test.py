import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import glob
from skimage.metrics import structural_similarity as ssim

# Define model architecture
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
    def __init__(self, hidden_dim=96):  # Updated hidden_dim to 96
        super().__init__()
        self.hidden_dim = hidden_dim
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
        combined = torch.cat([x, grad, h], dim=1)
        gates = torch.sigmoid(self.conv_gate(combined))
        update_gate, reset_gate = gates.chunk(2, dim=1)
        combined_reset = torch.cat([x, grad, reset_gate * h], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        h_new = (1 - update_gate) * h + update_gate * candidate
        return h_new

class RIM(nn.Module):
    def __init__(self, n_iter=15, hidden_dim=96):  # Updated hidden_dim to 96
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
            if torch.is_grad_enabled():
                x = x.detach().clone().requires_grad_(True)
                y_sim = forward_operator(x)
                loss = F.mse_loss(y_sim, y)
                grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            else:
                with torch.no_grad():
                    y_sim = forward_operator(x)
                    grad = torch.zeros_like(x)
            h = self.cell(x, grad, h)
            x = x + self.final_conv(h)
        return x

class LensingImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        assert len(self.files) > 0, f"No .npz files found in directory: {directory}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        lensed = data['lensed'].astype(np.float32)
        gt = data['gt'].astype(np.float32)
        lensed /= np.max(lensed) if np.max(lensed) > 0 else 1.0
        gt /= np.max(gt) if np.max(gt) > 0 else 1.0
        return torch.tensor(lensed).unsqueeze(0), torch.tensor(gt).unsqueeze(0)

def run_inference():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"C:\Users\mythi\.astropy\Code\Rim4.5_dreams\rim_dreams4.5.pt"  # Updated to match the saved model name from training script
    dataset_dir = r"C:\Users\mythi\.astropy\Code\unseen_lv4.5"
    output_dir = r"C:\Users\mythi\.astropy\Code\live_demo"
    num_samples_to_visualize = 8  # Number of samples to save as images
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    forward_operator = DifferentiableLensing().to(device)
    model = RIM(n_iter=10, hidden_dim=96).to(device)  # Updated hidden_dim to 96
    
    # Load trained model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Create dataset
    try:
        dataset = LensingImageDataset(dataset_dir)
        print(f"Loaded dataset with {len(dataset)} samples from {dataset_dir}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize metrics
    ssim_values = []
    mse_values = []
    
    # Process dataset
    with torch.no_grad():
        for idx, (lensed, gt) in enumerate(DataLoader(dataset, batch_size=1)):
            lensed = lensed.to(device)
            recon = model(lensed, forward_operator)
            
            # Convert to numpy
            lensed_np = lensed.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()
            recon_np = recon.squeeze().cpu().numpy()
            
            # Calculate metrics
            current_ssim = ssim(gt_np, recon_np, data_range=1.0)
            current_mse = np.mean((gt_np - recon_np) ** 2)
            ssim_values.append(current_ssim)
            mse_values.append(current_mse)
            
            # Save visualizations for first N samples
            if idx < num_samples_to_visualize:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(lensed_np, cmap='gray')
                axes[0].set_title("Lensed Image")
                axes[0].axis('off')
                
                axes[1].imshow(gt_np, cmap='gray')
                axes[1].set_title("Ground Truth")
                axes[1].axis('off')
                
                axes[2].imshow(recon_np, cmap='gray')
                axes[2].set_title(f"Reconstructed\nSSIM: {current_ssim:.4f}\nMSE: {current_mse:.6f}")
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'sample_{idx+1:03d}.png'), dpi=150, bbox_inches='tight')
                plt.close()
    
    # Calculate and print summary statistics
    avg_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    avg_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    
    print("\n=== Summary Statistics ===")
    print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Average MSE:  {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"Median SSIM:  {np.median(ssim_values):.4f}")
    print(f"Median MSE:   {np.median(mse_values):.6f}")
    
    # Save metrics distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(ssim_values, bins=20, edgecolor='black')
    plt.title(f'SSIM Distribution (μ={avg_ssim:.3f})')
    plt.xlabel('SSIM')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(mse_values, bins=20, edgecolor='black')
    plt.title(f'MSE Distribution (μ={avg_mse:.5f})')
    plt.xlabel('MSE')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=150)
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Model: {model_path}\n\n")
        f.write(f"Number of samples: {len(dataset)}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"Average MSE:  {avg_mse:.6f} ± {std_mse:.6f}\n")
        f.write(f"Median SSIM:  {np.median(ssim_values):.4f}\n")
        f.write(f"Median MSE:   {np.median(mse_values):.6f}\n")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    run_inference()