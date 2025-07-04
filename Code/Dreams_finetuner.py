#working I think
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
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

class LensingImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        assert len(self.files) > 0, "No .npz files found in dataset directory"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        lensed = data['lensed'].astype(np.float32)
        gt = data['gt'].astype(np.float32)
        lensed /= np.max(lensed) if np.max(lensed) > 0 else 1.0
        gt /= np.max(gt) if np.max(gt) > 0 else 1.0
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

    # Finetuning configuration
    batch_size = 16
    num_epochs = 75  # Reduced for finetuning
    learning_rate = 2e-4  # Reduced learning rate for finetuning
    patience = 20
    pretrained_model_path = r"C:\Users\mythi\.astropy\Code\Dreams_finetuned\rim_finetuned_dreams.pt"  # Path to the pretrained model
    finetuned_model_path = "rim_finetuned_dreams.pt"  # Path to save the finetuned model
    
    # Finetuning dataset - change this to your new dataset path
    dataset_dir = r"C:\Users\mythi\.astropy\Code\Hopes_and_dreams\lv4.5_dataset"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = LensingImageDataset(directory=dataset_dir)
    train_dataset.files = train_files
    val_dataset = LensingImageDataset(directory=dataset_dir)
    val_dataset.files = val_files

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize models
    forward_operator = DifferentiableLensing().to(device)
    model = RIM(n_iter=10, hidden_dim=96).to(device)
    
    # Load pretrained weights
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}. Starting from scratch.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Optional: use a learning rate scheduler for finetuning
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    loss_history = {"train": [], "val": [], "grad_norm": [], "val_ssim": []}
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        grad_norm_total = 0
        progress = tqdm(train_loader, desc=f"Finetuning Epoch {epoch+1}/{num_epochs}")

        for obs, gt in progress:
            obs, gt = obs.to(device), gt.to(device)
            optimizer.zero_grad()
            recon = model(obs, forward_operator)
            loss = F.mse_loss(recon, gt)
            loss.backward()

            grad_norm = sum(p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None)
            grad_norm_total += grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item()
            progress.set_postfix({'loss': f"{loss.item():.6f}", 'grad_norm': f"{grad_norm:.2f}"})

        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = grad_norm_total / len(train_loader)
        loss_history["train"].append(avg_train_loss)
        loss_history["grad_norm"].append(avg_grad_norm)

        model.eval()
        val_loss = 0
        val_ssim = 0
        with torch.no_grad():
            for obs, gt in val_loader:
                obs, gt = obs.to(device), gt.to(device)
                recon = model(obs, forward_operator).detach()
                val_loss += F.mse_loss(recon, gt).item()
                for i in range(recon.size(0)):
                    val_ssim += ssim(gt[i, 0].cpu().numpy(), recon[i, 0].cpu().numpy(), data_range=1.0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_ssim = val_ssim / len(val_dataset)
        loss_history["val"].append(avg_val_loss)
        loss_history["val_ssim"].append(avg_val_ssim)

        # Update learning rate scheduler
        lr_scheduler.step(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Val SSIM: {avg_val_ssim:.4f} | Grad Norm: {avg_grad_norm:.2f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), finetuned_model_path)
            print(f"Model improved - saving to {finetuned_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered after no improvement for", patience, "epochs.")
                break

        if (epoch + 1) % 5 == 0:  # Increased frequency of visualizations during finetuning
            model.eval()
            plt.figure(figsize=(15, 12))
            for i in range(4):
                obs, gt = val_dataset[i]
                obs = obs.unsqueeze(0).to(device)
                recon = model(obs, forward_operator).detach().cpu().squeeze()
                plt.subplot(4, 3, i*3+1)
                plt.imshow(obs.cpu().squeeze(), cmap='gray')
                plt.title(f"Lensed {i+1}")
                plt.axis('off')
                plt.subplot(4, 3, i*3+2)
                plt.imshow(gt.squeeze(), cmap='gray')
                plt.title(f"True {i+1}")
                plt.axis('off')
                plt.subplot(4, 3, i*3+3)
                plt.imshow(recon, cmap='gray')
                plt.title(f"Recon {i+1}")
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"finetune_epoch_{epoch+1}.png", dpi=300)
            plt.close()

    # Comparison of original and finetuned models
    if os.path.exists(pretrained_model_path):
        print("\nComparing original and finetuned models...")
        
        # Load original model for comparison
        original_model = RIM(n_iter=10, hidden_dim=96).to(device)
        original_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        original_model.eval()
        
        # Load finetuned model
        finetuned_model = RIM(n_iter=10, hidden_dim=96).to(device)
        finetuned_model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
        finetuned_model.eval()
        
        # Visualize comparison on validation samples
        plt.figure(figsize=(15, 20))
        with torch.no_grad():
            for i in range(5):
                obs, gt = val_dataset[i]
                obs = obs.unsqueeze(0).to(device)
                
                # Get reconstructions from both models
                original_recon = original_model(obs, forward_operator).detach().cpu().squeeze()
                finetuned_recon = finetuned_model(obs, forward_operator).detach().cpu().squeeze()
                
                # Calculate metrics
                original_ssim = ssim(gt.squeeze().numpy(), original_recon.numpy(), data_range=1.0)
                finetuned_ssim = ssim(gt.squeeze().numpy(), finetuned_recon.numpy(), data_range=1.0)
                
                # Plot comparisons
                plt.subplot(5, 4, i*4+1)
                plt.imshow(obs.cpu().squeeze(), cmap='gray')
                plt.title(f"Lensed {i+1}")
                plt.axis('off')
                
                plt.subplot(5, 4, i*4+2)
                plt.imshow(gt.squeeze(), cmap='gray')
                plt.title(f"Ground Truth")
                plt.axis('off')
                
                plt.subplot(5, 4, i*4+3)
                plt.imshow(original_recon, cmap='gray')
                plt.title(f"Original Model\nSSIM: {original_ssim:.4f}")
                plt.axis('off')
                
                plt.subplot(5, 4, i*4+4)
                plt.imshow(finetuned_recon, cmap='gray')
                plt.title(f"Finetuned Model\nSSIM: {finetuned_ssim:.4f}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=300)
        plt.close()

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["train"], label='Train Loss')
    plt.plot(loss_history["val"], label='Val Loss')
    plt.plot(loss_history["val_ssim"], label='Val SSIM')
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("Finetuning Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("finetuning_curves.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    start = time.time()
    finetune_rim()
    end = time.time()
    elapsed_hours = (end - start) / 3600
    print(f"Finetuning complete in {elapsed_hours:.2f} hours")