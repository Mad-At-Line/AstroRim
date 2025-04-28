import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import os, glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

# ---------------------- Model Architecture ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.conv(x), 0.2)

class DifferentiableLensing(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            ResidualBlock(96),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---------------------- Data Loading ----------------------
class LensingImageDataset(Dataset):
    def __init__(self, files, global_max, augment=False):
        self.files = files
        self.global_max = global_max
        self.augment = augment
        self.transforms = T.RandomChoice([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.RandomRotation(90)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        source = data['gt'].astype(np.float32) / self.global_max
        lensed = data['lensed'].astype(np.float32) / self.global_max
        source = torch.tensor(source).unsqueeze(0)
        lensed = torch.tensor(lensed).unsqueeze(0)
        if self.augment:
            source = self.transforms(source)
        return source, lensed

def compute_global_max(directory):
    files = glob.glob(os.path.join(directory, "*.npz"))
    return max(np.load(f)['lensed'].max() for f in tqdm(files, desc="Computing max"))

# ---------------------- Metrics & Visualization ----------------------
def calculate_ssim(pred, target):
    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    return ssim(target_np, pred_np, data_range=1.0)

def save_visualization(epoch, samples, model, global_max, device, phase="forward_op"):
    model.eval()
    with torch.no_grad():
        fig, axs = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))

        for i, (src, tgt) in enumerate(samples):
            pred = model(src.unsqueeze(0).to(device)).squeeze().cpu().numpy()

            src_img = src.squeeze().numpy() * global_max
            tgt_img = tgt.squeeze().numpy() * global_max
            pred_img = pred * global_max

            display_max = max(tgt_img.max(), pred_img.max()) * 1.1

            axs[i,0].imshow(src_img, cmap='gray', vmin=0, vmax=display_max)
            axs[i,0].set_title(f"Source (GT)\nMax: {src_img.max():.2f}")
            axs[i,1].imshow(pred_img, cmap='gray', vmin=0, vmax=display_max)
            axs[i,1].set_title(f"Predicted Lens\nMax: {pred_img.max():.2f}")
            axs[i,2].imshow(tgt_img, cmap='gray', vmin=0, vmax=display_max)
            axs[i,2].set_title(f"True Lens\nMax: {tgt_img.max():.2f}")

            for j in range(3):
                axs[i,j].axis('off')
                plt.colorbar(axs[i,j].images[0], ax=axs[i,j], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"{phase}_epoch_{epoch:03d}.png", dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

# ---------------------- Training Loop ----------------------
def train_forward_operator():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = r"C:\\Users\\mythi\\.astropy\\Code\\Hopes_and_dreams\\data_15k"
    global_max = compute_global_max(data_dir)
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_ds = LensingImageDataset(train_files, global_max, augment=True)
    val_ds = LensingImageDataset(val_files, global_max)
    viz_samples = [val_ds[i] for i in range(min(4, len(val_ds)))]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, pin_memory=True)
    model = DifferentiableLensing().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_ssim': []}

    for epoch in range(1, 101):
        model.train()
        train_loss = 0.0
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(src)
                mse = F.mse_loss(pred, tgt)
                ssim_val = calculate_ssim(pred, tgt)
                loss = mse + 0.8 * (1 - ssim_val)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                pred = model(src)
                mse = F.mse_loss(pred, tgt)
                ssim_val = calculate_ssim(pred, tgt)
                loss = mse + 0.8 * (1 - ssim_val)
                val_loss += loss.item()
                val_ssim += ssim_val * src.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_ssim = val_ssim / len(val_ds)
        history['val_loss'].append(avg_val_loss)
        history['val_ssim'].append(avg_val_ssim)

        scheduler.step()

        print(f"\nEpoch {epoch} Metrics:")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        print(f"Val SSIM: {avg_val_ssim:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_forward_op.pt")
            print(f"🎉 New best model saved with val loss: {best_val_loss:.6f}, SSIM: {avg_val_ssim:.4f}")

        if epoch % 10 == 0 or epoch == 1:
            save_visualization(epoch, viz_samples, model, global_max, device)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Loss Curves')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history['val_ssim'], label='Val SSIM', color='green')
            plt.title('Validation SSIM')
            plt.legend()

            plt.tight_layout()
            plt.savefig('training_metrics.png', dpi=150)
            plt.close()

if __name__ == "__main__":
    train_forward_operator()