
import os
import glob
import argparse
import csv
import time

import numpy as np
from astropy.io import fits

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.transform import resize as sk_resize
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


DEFAULT_INPUT_FILE = r"C:\Users\mythi\.astropy\Code\Lensing_option\CFRS0310787.fits"
DEFAULT_INPUT_DIR = None
DEFAULT_OUT_DIR = r"C:\Users\mythi\.astropy\Code\Real_results_v5"
DEFAULT_MODEL = r"C:\Users\mythi\.astropy\Code\WORKING_WITH_SIM_MODELS\Finetuning_Round2_model\rim_finetune_best.pt"
DEFAULT_FORWARD = r"C:\Users\mythi\.astropy\Code\WORKING_WITH_SIM_MODELS\Finetuning_Round2_model\forward_finetune_best.pt"
DEFAULT_KERNEL = 21
DEFAULT_NITER = 8
DEFAULT_RESIZE = None 

class PhysicalForward(nn.Module):
    def __init__(self, kernel_size=21, device='cpu', enforce_nonneg=True, init_sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.enforce_nonneg = enforce_nonneg
        self.device_cached = device

        raw = torch.randn(1, 1, kernel_size, kernel_size) * 0.01
        self.raw_psf = nn.Parameter(raw)
        self._init_gaussian(init_sigma)

        # lens params
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
        x = forward_operator.adjoint(y)  # initial guess by adjoint
        for _ in range(self.n_iter):
            y_sim = forward_operator.forward(x)
            residual = y_sim - y
            grad = forward_operator.adjoint(residual)
            x, h = self.cell(x, grad, h)
        x = x + self.refine(h)
        return x


def read_fits_lensed(fn):
    hdul = fits.open(fn, memmap=False)
    gt = None
    lensed = None
    hdrs = {}
    for h in hdul:
        name = getattr(h, 'name', '').upper()
        if name == 'GT':
            gt = h.data.astype(np.float32)
            hdrs['GT'] = h.header
        elif name == 'LENSED':
            lensed = h.data.astype(np.float32)
            hdrs['LENSED'] = h.header
    if lensed is None and hdul[0].data is not None:
        lensed = hdul[0].data.astype(np.float32)
        hdrs['LENSED'] = hdul[0].header
    hdul.close()
    if lensed is None:
        raise RuntimeError(f"No image found in {fn}")
    lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if gt is not None:
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return lensed, gt, hdrs


def save_twohdu_fits(outpath, lensed, recon, hdrs=None):
    prih = fits.PrimaryHDU()
    hdul = fits.HDUList([prih])
    hdul.append(fits.ImageHDU(data=lensed.astype(np.float32), name='LENSED',
                              header=(hdrs.get('LENSED') if hdrs else None)))
    hdul.append(fits.ImageHDU(data=recon.astype(np.float32), name='RECON'))
    hdul.writeto(outpath, overwrite=True)


def save_preview_png_two(img_path, lensed, recon, vmax=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(lensed, origin='lower')
    axes[0].set_title('LENSED')
    axes[0].axis('off')
    axes[1].imshow(recon, origin='lower', vmax=vmax)
    axes[1].set_title('RECON')
    axes[1].axis('off')
    plt.tight_layout()
    fig.savefig(img_path, dpi=150)
    plt.close(fig)


def preprocess_and_maybe_resize(img, target_size=None):
    arr = img.astype(np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[..., 0]
    if target_size is not None:
        arr_resized = sk_resize(arr, (target_size, target_size), preserve_range=True, anti_aliasing=True)
        return arr_resized.astype(np.float32)
    return arr


def infer_file(fn, model, forward_operator, device, out_dir, rescale=True, save_png=True, resize_to=None):
    lensed, gt, hdrs = read_fits_lensed(fn)
    if resize_to is not None:
        lensed_proc = preprocess_and_maybe_resize(lensed, target_size=resize_to)
        gt_proc = preprocess_and_maybe_resize(gt, target_size=resize_to) if gt is not None else None
    else:
        lensed_proc = preprocess_and_maybe_resize(lensed, target_size=None)
        gt_proc = preprocess_and_maybe_resize(gt, target_size=None) if gt is not None else None

    eps = 1e-8
    m = max(float(np.nanmax(lensed_proc)), eps)
    obs = lensed_proc / m
    if obs.ndim != 2:
        raise RuntimeError("Expected 2D images for LENSED HDU after preprocessing")
    obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    forward_operator.eval()
    with torch.no_grad():
        recon_t = model(obs_t, forward_operator)
    recon_np = recon_t.cpu().numpy()[0, 0]
    recon_np = np.clip(recon_np, 0.0, 1.0)

    if rescale:
        recon_out = (recon_np * m).astype(np.float32)
        lensed_out = lensed_proc.astype(np.float32)
        gt_out = gt_proc.astype(np.float32) if gt_proc is not None else None
    else:
        recon_out = recon_np.astype(np.float32)
        lensed_out = (lensed_proc / m).astype(np.float32)
        gt_out = (gt_proc / m).astype(np.float32) if gt_proc is not None else None

    metrics = {}
    if gt_out is not None:
        try:
            g = np.clip(gt_out, 0.0, np.max(gt_out) if np.max(gt_out) > 0 else 1.0)
            r = np.clip(recon_out, 0.0, np.max(recon_out) if np.max(recon_out) > 0 else 1.0)
            metrics['mse'] = float(np.mean((g - r) ** 2))
            rng = max(g.max(), r.max(), 1e-8)
            metrics['ssim'] = float(ssim((g / rng).astype(np.float32), (r / rng).astype(np.float32), data_range=1.0))
        except Exception:
            metrics['mse'] = None
            metrics['ssim'] = None

    base = os.path.splitext(os.path.basename(fn))[0]
    out_fits = os.path.join(out_dir, f"{base}_recon.fits")
    save_twohdu_fits(out_fits, lensed_out, recon_out, hdrs=hdrs)

    png_path = None
    if save_png:
        png_path = os.path.join(out_dir, f"{base}_recon_preview.png")
        try:
            vmax = np.percentile(recon_out, 99.0)
        except Exception:
            vmax = None
        save_preview_png_two(png_path, lensed_out, recon_out, vmax=vmax)

    return out_fits, png_path, metrics


def load_forward_weights(forward_operator, forward_path, device):
    if not os.path.isfile(forward_path):
        raise FileNotFoundError(f"Forward operator file not found: {forward_path}")
    state = torch.load(forward_path, map_location=device)
    # state might be a state_dict, or a dict containing forward_state
    if isinstance(state, dict):
        # common possibilities
        if 'forward_state' in state:
            forward_operator.load_state_dict(state['forward_state'])
        elif 'raw_psf' in state or 'raw_psf' in next(iter([state]).__iter__(), {}):
            # either state is direct state_dict for forward_operator
            forward_operator.load_state_dict(state)
        else:
            # attempt to find plausible keys subset
            try:
                forward_operator.load_state_dict(state)
            except Exception:
                # try to extract nested forward key
                found = False
                for k, v in state.items():
                    if isinstance(v, dict) and ('raw_psf' in v or 'x0' in v):
                        forward_operator.load_state_dict(v)
                        found = True
                        break
                if not found:
                    raise RuntimeError("Could not interpret forward operator checkpoint format.")
    else:
        forward_operator.load_state_dict(state)
    return forward_operator


def load_model_weights(model, model_path, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if 'model_state' in state:
            model.load_state_dict(state['model_state'])
        elif any(k.startswith('cell') or k.startswith('refine') or k.startswith('to_image') for k in state.keys()):
            model.load_state_dict(state)
        else:
            # try nested search
            found = False
            for k, v in state.items():
                if isinstance(v, dict) and any(s.startswith('cell') or s.startswith('refine') for s in v.keys()):
                    model.load_state_dict(v)
                    found = True
                    break
            if not found:
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    raise RuntimeError(f"Unrecognised model checkpoint format: {e}")
    else:
        model.load_state_dict(state)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description="RIM inference compatible with your training script (PhysicalForward + RIMImproved).")
    parser.add_argument('--input-file', default=DEFAULT_INPUT_FILE, help='Single .fits file to process')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR, help='Directory with .fits files to process (optional)')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Directory to write outputs')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Path to RIM model weights')
    parser.add_argument('--forward', default=DEFAULT_FORWARD, help='Path to forward operator weights')
    parser.add_argument('--kernel', type=int, default=DEFAULT_KERNEL, help='Kernel size for PSF conv')
    parser.add_argument('--niter', type=int, default=DEFAULT_NITER, help='RIM iterations (should match training)')
    parser.add_argument('--no-rescale', dest='rescale', action='store_false', help='Do NOT rescale output to observation max')
    parser.add_argument('--no-png', dest='save_png', action='store_false', help='Do NOT save PNG previews')
    parser.add_argument('--device', default=None, help='cuda or cpu (auto if not given)')
    parser.add_argument('--resize', type=int, default=DEFAULT_RESIZE, help='Resize input to this spatial size (e.g. 96); if omitted, keep native size')
    args = parser.parse_args(argv)

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # instantiate
    forward_operator = PhysicalForward(kernel_size=args.kernel, device=str(device), enforce_nonneg=True).to(device)
    model = RIMImproved(n_iter=args.niter, hidden_dim=96).to(device)

    # load forward weights
    try:
        _ = load_forward_weights(forward_operator, args.forward, device)
        print(f"Loaded forward operator weights from {args.forward}")
    except Exception as e:
        raise RuntimeError(f"Failed to load forward operator: {e}")

    # load model weights
    try:
        _ = load_model_weights(model, args.model, device)
        print(f"Loaded RIM model weights from {args.model}")
    except Exception as e:
        raise RuntimeError(f"Failed to load RIM model: {e}")

    # prepare file list
    files = []
    if args.input_file:
        files.append(args.input_file)
    if args.input_dir:
        files += sorted(glob.glob(os.path.join(args.input_dir, "*.fits")))
    files = [f for f in dict.fromkeys(files) if f]  # unique and drop empties
    if len(files) == 0:
        print("No files found. Check input-file / input-dir.")
        return

    t0 = time.time()
    results = {}
    for fn in files:
        print(f"Processing {fn} ...")
        try:
            out_fits, png_path, metrics = infer_file(fn, model, forward_operator, device,
                                                     args.out_dir, rescale=args.rescale, save_png=args.save_png,
                                                     resize_to=args.resize)
            print(f"  -> wrote {out_fits}")
            if png_path:
                print(f"  -> wrote {png_path}")
            if metrics:
                print(f"     Metrics (if GT present): MSE: {metrics.get('mse', 'N/A')}, SSIM: {metrics.get('ssim', 'N/A')}")
            results[fn] = metrics or {}
        except Exception as e:
            print(f"  Error processing {fn}: {e}")
            results[fn] = {'error': str(e)}

    dt = time.time() - t0
    print(f"Done. Processed {len(files)} files in {dt:.1f}s")

    # write CSV summary
    try:
        csvp = os.path.join(args.out_dir, 'recon_summary.csv')
        with open(csvp, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'mse', 'ssim', 'error'])
            for f, m in results.items():
                if not m:
                    w.writerow([f, '', '', ''])
                else:
                    w.writerow([f, m.get('mse', ''), m.get('ssim', ''), m.get('error', '')])
        print(f"Summary CSV written to {csvp}")
    except Exception:
        pass


if __name__ == '__main__':
    main()


