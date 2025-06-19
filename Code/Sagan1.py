import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import glob, os, time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------- Dataset ----------
class LensingImageDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        assert self.files, "No .npz files found"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        lensed = data['lensed'].astype(np.float32)
        gt     = data['gt'].astype(np.float32)
        # Normalize each channel independently
        for c in range(3):
            max_l = max(np.max(lensed[c]), 1e-6)
            max_g = max(np.max(gt[c]),     1e-6)
            lensed[c] /= max_l
            gt[c]     /= max_g
        return torch.tensor(lensed), torch.tensor(gt)

# ---------- Forward Operator ----------
class DifferentiableLensing(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4), nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        # freeze physics model
        for p in self.net.parameters(): p.requires_grad = False
    def forward(self, x):
        return self.net(x)

# ---------- RIM Cell ----------
class RIMCell(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv2d(3+3+hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, 3, padding=1)
        )
        self.conv_candidate = nn.Sequential(
            nn.Conv2d(3+3+hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
    def forward(self, x, grad, h):
        combined = torch.cat([x, grad, h], dim=1)
        ug, rg = torch.sigmoid(self.conv_gate(combined)).chunk(2, dim=1)
        combined_reset = torch.cat([x, grad, rg*h], dim=1)
        cand = torch.tanh(self.conv_candidate(combined_reset))
        return (1-ug)*h + ug*cand

# ---------- RIM Model ----------
class RIM(nn.Module):
    def __init__(self, n_iter=10, hidden_dim=96):
        super().__init__()
        self.n_iter = n_iter
        self.cell = RIMCell(hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, 3, 3, padding=1)
    def forward(self, y, forward_op):
        B,C,H,W = y.shape
        h = torch.zeros(B, self.cell.conv_gate[0].out_channels, H, W, device=y.device)
        x = y.clone()
        for _ in range(self.n_iter):
            x = x.detach().clone().requires_grad_(True)
            y_sim = forward_op(x)
            mse_loss = F.mse_loss(y_sim, y)
            grad = torch.autograd.grad(mse_loss, x, create_graph=True)[0]
            h = self.cell(x, grad, h)
            x = x + self.final_conv(h)
        return x

# ---------- Utility ----------
def normalize_for_display(t):
    a = t.detach().cpu().numpy()
    a = np.transpose(a, (1,2,0))
    return np.clip(a/np.max(a), 0,1)

# ---------- Training ----------
def train_rim_rgb():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = r"C:\Users\mythi\.astropy\Code\COLOR\Colordata1"
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    train_f, val_f = train_test_split(files, test_size=0.2, random_state=42)

    train_ds = LensingImageDataset(data_dir); train_ds.files = train_f
    val_ds   = LensingImageDataset(data_dir); val_ds.files   = val_f

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, pin_memory=True)

    model = RIM(n_iter=10, hidden_dim=96).to(device)
    forward_op = DifferentiableLensing().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val, no_imp = float('inf'), 0
    for epoch in range(1,51):
        model.train()
        train_loss, train_ssim = 0,0
        for obs, gt in tqdm(train_loader, desc=f"Train {epoch}"):
            obs,gt = obs.to(device), gt.to(device)
            optimizer.zero_grad()
            recon = model(obs, forward_op)
            mse = F.mse_loss(recon, gt)
            # SSIM loss
            ssim_batch = []
            for i in range(obs.size(0)):
                gt_img   = gt[i].permute(1,2,0).cpu().numpy()
                rec_img  = recon[i].permute(1,2,0).cpu().detach().numpy()
                ssim_batch.append(ssim(gt_img, rec_img, channel_axis=2, data_range=1.0))
            ssim_val = torch.tensor(ssim_batch, device=device).mean()
            loss = mse + 0.2*(1-ssim_val)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            train_loss += loss.item(); train_ssim += ssim_val.item()
        train_loss /= len(train_loader); train_ssim /= len(train_loader)

        # validation
        model.eval(); val_loss,val_ssim=0,0
        with torch.no_grad():
            for obs, gt in val_loader:
                obs,gt = obs.to(device), gt.to(device)
                recon = model(obs, forward_op)
                val_loss += F.mse_loss(recon, gt).item()
                for i in range(obs.size(0)):
                    gt_img  = gt[i].permute(1,2,0).cpu().numpy()
                    rec_img = recon[i].permute(1,2,0).cpu().numpy()
                    val_ssim += ssim(gt_img, rec_img, channel_axis=2, data_range=1.0)
        val_loss /= len(val_loader); val_ssim /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Train SSIM: {train_ssim:.4f} | Val Loss: {val_loss:.4f} | Val SSIM: {val_ssim:.4f}")

        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
            torch.save(model.state_dict(), 'rim_rgb_best.pth')
        else:
            no_imp +=1
            if no_imp>=10: break

        # quick viz
        if epoch%5==0:
            obs,gt = next(iter(val_loader)); obs,gt=obs.to(device),gt.to(device)
            with torch.no_grad(): recon=model(obs,forward_op)
            fig,axs=plt.subplots(3,3,figsize=(10,10))
            for i in range(3): axs[i,2].imshow(normalize_for_display(recon[i])); axs[i,1].imshow(normalize_for_display(gt[i])); axs[i,0].imshow(normalize_for_display(obs[i]));
            plt.savefig(f'viz_{epoch}.png'); plt.close()

if __name__=='__main__':
    t0=time.time(); train_rim_rgb(); print(f"Elapsed: {(time.time()-t0)/3600:.2f}h")
