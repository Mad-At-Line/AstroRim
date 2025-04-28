import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import glob
import time
import random
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        # Calculate SSIM for each image in batch and average
        batch_size = x.size(0)
        ssim_val = 0.0
        for i in range(batch_size):
            ssim_val += ssim(x[i, 0].detach().cpu().numpy(), 
                              y[i, 0].detach().cpu().numpy(), 
                              data_range=1.0, 
                              win_size=self.window_size)
        return 1 - (ssim_val / batch_size)  # Convert to loss (1-SSIM)

class LensingImageDataset(Dataset):
    def __init__(self, directory, files=None, augment=False):
        self.directory = directory
        if files is not None:
            self.files = files
        else:
            self.files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        assert len(self.files) > 0, "No .npz files found in dataset directory"
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        lensed = data['lensed'].astype(np.float32)
        gt = data['gt'].astype(np.float32)
        lensed /= np.max(lensed) if np.max(lensed) > 0 else 1.0
        gt /= np.max(gt) if np.max(gt) > 0 else 1.0
        
        # Convert to torch tensors
        lensed = torch.tensor(lensed).unsqueeze(0)
        gt = torch.tensor(gt).unsqueeze(0)
        
        # Apply augmentations if enabled
        if self.augment:
            if random.random() > 0.5:
                # Random horizontal flip
                lensed = torch.flip(lensed, [2])
                gt = torch.flip(gt, [2])
            
            if random.random() > 0.5:
                # Random vertical flip
                lensed = torch.flip(lensed, [1])
                gt = torch.flip(gt, [1])
            
            # Random rotation (90, 180, 270 degrees)
            k = random.randint(0, 3)
            if k > 0:
                lensed = torch.rot90(lensed, k, [1, 2])
                gt = torch.rot90(gt, k, [1, 2])
                
        return lensed, gt

# Include the model definitions from your original code
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
    def __init__(self, n_iter=15, hidden_dim=96):
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

def finetune_rim():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Improved configuration
    batch_size = 16
    num_epochs = 50
    learning_rate = 2e-4  # Increased initial learning rate
    patience = 15  # Increased patience for early stopping
    pretrained_model_path = r"C:\Users\mythi\.astropy\Code\Working_Model_Dreams\rim_Dreams.pt"
    finetuned_model_path = "rim_finetuned_dreams.pt"
    best_ssim_model_path = "rim_best_ssim_dreams.pt"  # New path for SSIM-optimized model
    
    # Dataset setup
    dataset_dir = r"C:\Users\mythi\.astropy\Code\Hopes_and_dreams\Level5_dataset"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = LensingImageDataset(directory=dataset_dir, files=train_files, augment=True)  # Enable augmentation for training
    val_dataset = LensingImageDataset(directory=dataset_dir, files=val_files, augment=False)

    # Use fewer workers if experiencing memory issues
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # Initialize models
    forward_operator = DifferentiableLensing().to(device)
    model = RIM(n_iter=10, hidden_dim=96).to(device)
    
    # Load pretrained weights
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}. Starting from scratch.")
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Improved learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    # Initialize loss functions
    ssim_loss_fn = SSIMLoss()
    
    # Tracking metrics
    best_val_loss = float('inf')
    best_val_ssim = 0.0
    loss_history = {
        "train": [], 
        "val": [], 
        "train_ssim": [],
        "val_ssim": [], 
        "val_psnr": [],
        "grad_norm": []
    }
    epochs_no_improve = 0
    
    # Curriculum learning - gradually increase iterations
    def get_curriculum_iters(epoch):
        if epoch < 10:
            return 8
        elif epoch < 20:
            return 10
        else:
            return 12

    for epoch in range(num_epochs):
        # Set curriculum iterations
        model.n_iter = get_curriculum_iters(epoch)
        print(f"Epoch {epoch+1}: Using {model.n_iter} iterations in RIM")
        
        # Training phase
        model.train()
        train_loss = 0
        train_ssim_total = 0
        grad_norm_total = 0
        progress = tqdm(train_loader, desc=f"Finetuning Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (obs, gt) in enumerate(progress):
            obs, gt = obs.to(device), gt.to(device)
            
            optimizer.zero_grad()
            
            recon = model(obs, forward_operator)
            
            # Combined loss: MSE + SSIM
            mse_loss = F.mse_loss(recon, gt)
            ssim_loss_val = ssim_loss_fn(recon, gt)
            
            # Dynamic loss weighting that changes over time
            alpha = max(0.8 - epoch * 0.01, 0.5)  # Gradually increase SSIM weight
            loss = alpha * mse_loss + (1-alpha) * ssim_loss_val
            
            loss.backward()
            
            # Calculate gradient statistics
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_norm_total += grad_norm
            
            # Step optimizer
            optimizer.step()
            
            # Calculate training metrics
            train_loss += loss.item()
            with torch.no_grad():
                for i in range(recon.size(0)):
                    train_ssim_total += ssim(gt[i, 0].cpu().numpy(), recon[i, 0].detach().cpu().numpy(), data_range=1.0)
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{loss.item():.6f}", 
                'mse': f"{mse_loss.item():.6f}",
                'ssim_loss': f"{ssim_loss_val.item():.4f}",
                'grad': f"{grad_norm:.2f}"
            })

        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ssim = train_ssim_total / len(train_dataset)
        avg_grad_norm = grad_norm_total / len(train_loader)
        
        # Store training metrics
        loss_history["train"].append(avg_train_loss)
        loss_history["train_ssim"].append(avg_train_ssim)
        loss_history["grad_norm"].append(avg_grad_norm)

        # Validation phase
        model.eval()
        val_loss = 0
        val_ssim = 0
        val_psnr = 0
        with torch.no_grad():
            for obs, gt in val_loader:
                obs, gt = obs.to(device), gt.to(device)
                recon = model(obs, forward_operator)
                val_loss += F.mse_loss(recon, gt).item()
                
                # Calculate SSIM and PSNR for each image in batch
                for i in range(recon.size(0)):
                    gt_np = gt[i, 0].cpu().numpy()
                    recon_np = recon[i, 0].cpu().numpy()
                    val_ssim += ssim(gt_np, recon_np, data_range=1.0)
                    val_psnr += psnr(gt_np, recon_np, data_range=1.0)

        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ssim = val_ssim / len(val_dataset)
        avg_val_psnr = val_psnr / len(val_dataset)
        
        # Store validation metrics
        loss_history["val"].append(avg_val_loss)
        loss_history["val_ssim"].append(avg_val_ssim)
        loss_history["val_psnr"].append(avg_val_psnr)

        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Train SSIM: {avg_train_ssim:.4f} | Val SSIM: {avg_val_ssim:.4f} | "
              f"Val PSNR: {avg_val_psnr:.2f} | Grad Norm: {avg_grad_norm:.2f} | LR: {current_lr:.6f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), finetuned_model_path)
            print(f"Model improved (loss) - saving to {finetuned_model_path}")
        else:
            epochs_no_improve += 1
        
        # Save model if SSIM improves (separately)
        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            torch.save(model.state_dict(), best_ssim_model_path)
            print(f"Model improved (SSIM) - saving to {best_ssim_model_path}")

        # Early stopping check
        if epochs_no_improve >= patience:
            print("Early stopping triggered after no improvement for", patience, "epochs.")
            break

        # Generate visualizations every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            plt.figure(figsize=(15, 12))
            with torch.no_grad():
                for i in range(4):
                    obs, gt = val_dataset[i]
                    obs = obs.unsqueeze(0).to(device)
                    recon = model(obs, forward_operator).detach().cpu().squeeze()
                    
                    # Calculate sample metrics
                    sample_ssim = ssim(gt.squeeze().numpy(), recon.numpy(), data_range=1.0)
                    sample_psnr = psnr(gt.squeeze().numpy(), recon.numpy(), data_range=1.0)
                    
                    plt.subplot(4, 3, i*3+1)
                    plt.imshow(obs.cpu().squeeze(), cmap='gray')
                    plt.title(f"Lensed {i+1}")
                    plt.axis('off')
                    
                    plt.subplot(4, 3, i*3+2)
                    plt.imshow(gt.squeeze(), cmap='gray')
                    plt.title(f"Ground Truth")
                    plt.axis('off')
                    
                    plt.subplot(4, 3, i*3+3)
                    plt.imshow(recon, cmap='gray')
                    plt.title(f"Recon (SSIM: {sample_ssim:.4f}, PSNR: {sample_psnr:.2f})")
                    plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"finetune_epoch_{epoch+1}.png", dpi=300)
            plt.close()

    # Plot comprehensive training history
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(loss_history["train"], label='Train Loss')
    plt.plot(loss_history["val"], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot SSIM
    plt.subplot(2, 2, 2)
    plt.plot(loss_history["train_ssim"], label='Train SSIM')
    plt.plot(loss_history["val_ssim"], label='Val SSIM')
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM Metrics")
    plt.legend()
    plt.grid(True)
    
    # Plot PSNR
    plt.subplot(2, 2, 3)
    plt.plot(loss_history["val_psnr"], label='Val PSNR')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Metrics")
    plt.legend()
    plt.grid(True)
    
    # Plot gradient norm
    plt.subplot(2, 2, 4)
    plt.plot(loss_history["grad_norm"], label='Gradient Norm')
    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    plt.title("Gradient Norm")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("finetuning_metrics.png", dpi=300)
    plt.close()
    
    # Final comparison of models
    if os.path.exists(pretrained_model_path) and os.path.exists(finetuned_model_path) and os.path.exists(best_ssim_model_path):
        print("\nComparing original, finetuned, and SSIM-optimized models...")
        
        # Load all models
        original_model = RIM(n_iter=10, hidden_dim=96).to(device)
        original_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        original_model.eval()
        
        finetuned_model = RIM(n_iter=10, hidden_dim=96).to(device)
        finetuned_model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
        finetuned_model.eval()
        
        ssim_model = RIM(n_iter=10, hidden_dim=96).to(device)
        ssim_model.load_state_dict(torch.load(best_ssim_model_path, map_location=device))
        ssim_model.eval()
        
        # Create comparison figure
        plt.figure(figsize=(18, 20))
        with torch.no_grad():
            for i in range(5):
                obs, gt = val_dataset[i]
                obs = obs.unsqueeze(0).to(device)
                
                # Get reconstructions from all models
                original_recon = original_model(obs, forward_operator).detach().cpu().squeeze()
                finetuned_recon = finetuned_model(obs, forward_operator).detach().cpu().squeeze()
                ssim_recon = ssim_model(obs, forward_operator).detach().cpu().squeeze()
                
                # Calculate metrics
                original_ssim_val = ssim(gt.squeeze().numpy(), original_recon.numpy(), data_range=1.0)
                finetuned_ssim_val = ssim(gt.squeeze().numpy(), finetuned_recon.numpy(), data_range=1.0)
                ssim_model_ssim_val = ssim(gt.squeeze().numpy(), ssim_recon.numpy(), data_range=1.0)
                
                original_psnr_val = psnr(gt.squeeze().numpy(), original_recon.numpy(), data_range=1.0)
                finetuned_psnr_val = psnr(gt.squeeze().numpy(), finetuned_recon.numpy(), data_range=1.0)
                ssim_model_psnr_val = psnr(gt.squeeze().numpy(), ssim_recon.numpy(), data_range=1.0)
                
                # Plot comparisons
                plt.subplot(5, 5, i*5+1)
                plt.imshow(obs.cpu().squeeze(), cmap='gray')
                plt.title(f"Lensed {i+1}")
                plt.axis('off')
                
                plt.subplot(5, 5, i*5+2)
                plt.imshow(gt.squeeze(), cmap='gray')
                plt.title(f"Ground Truth")
                plt.axis('off')
                
                plt.subplot(5, 5, i*5+3)
                plt.imshow(original_recon, cmap='gray')
                plt.title(f"Original Model\nSSIM: {original_ssim_val:.4f}\nPSNR: {original_psnr_val:.2f}")
                plt.axis('off')
                
                plt.subplot(5, 5, i*5+4)
                plt.imshow(finetuned_recon, cmap='gray')
                plt.title(f"Finetuned (MSE)\nSSIM: {finetuned_ssim_val:.4f}\nPSNR: {finetuned_psnr_val:.2f}")
                plt.axis('off')
                
                plt.subplot(5, 5, i*5+5)
                plt.imshow(ssim_recon, cmap='gray')
                plt.title(f"SSIM-optimized\nSSIM: {ssim_model_ssim_val:.4f}\nPSNR: {ssim_model_psnr_val:.2f}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("model_comparison_final.png", dpi=300)
        plt.close()

if __name__ == '__main__':
    start = time.time()
    finetune_rim()
    end = time.time()
    elapsed_hours = (end - start) / 3600
    print(f"Finetuning complete in {elapsed_hours:.2f} hours")