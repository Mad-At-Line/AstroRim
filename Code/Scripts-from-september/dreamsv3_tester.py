

import os
import glob
import time
import numpy as np
from astropy.io import fits
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


INPUT_FILE = None
INPUT_DIR = r"C:\Users\mythi\.astropy\Code\Fits_work\unseen_lv5"

OUT_DIR = r"C:\Users\mythi\.astropy\Code\sim_results_lv5"
MODEL_PATH = r"C:\Users\mythi\.astropy\Code\Fits_work\Rim_improved_models\rim_finetune_best3.pt"
FORWARD_PATH = r"C:\Users\mythi\.astropy\Code\Fits_work\Rim_improved_models\forward_finetune_best3.pt"


N_ITER = 8           
RESCALE_OUTPUT = True
SAVE_PNG = True
DEVICE = None      

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

        # PSF parameter
        raw = torch.randn(1, 1, kernel_size, kernel_size) * 0.01
        self.raw_psf = nn.Parameter(raw)
        self._init_gaussian(init_sigma)

        # Parametric lens parameters 
        self.x0 = nn.Parameter(torch.tensor(0.0))
        self.y0 = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(0.08))   
        self.raw_rc = nn.Parameter(torch.tensor(0.01))  

        # small learnable sub-pixel shift per image 
        self.raw_subpix = nn.Parameter(torch.zeros(2))  # dx, dy in normalized coords (raw)
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

def read_fits_pair(fn):
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
        raise RuntimeError(f"No LENSED HDU found in {fn}")
    lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if gt is not None:
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return lensed, gt, hdrs

def save_recon_fits(outpath, lensed, recon, gt=None, hdrs=None):
    prih = fits.PrimaryHDU()
    hdul = fits.HDUList([prih])
    hdul.append(fits.ImageHDU(data=lensed.astype(np.float32), name='LENSED', header=(hdrs.get('LENSED') if hdrs else None)))
    if gt is not None:
        hdul.append(fits.ImageHDU(data=gt.astype(np.float32), name='GT', header=(hdrs.get('GT') if hdrs else None)))
    hdul.append(fits.ImageHDU(data=recon.astype(np.float32), name='RECON'))
    hdul.writeto(outpath, overwrite=True)

def save_preview_png(png_path, lensed, recon, gt=None, vmin=None, vmax=None):
    """
    Robust preview saver. Accepts vmin/vmax; if None compute safe defaults while guarding NaNs/Infs.
    """
    # ensure numeric finite arrays
    lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if gt is not None:
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # determine display vmin/vmax if not provided
    try:
        if vmin is None:
            vmin = float(np.nanpercentile(recon, 1.0)) if recon.size else 0.0
        if vmax is None:
            vmax = float(np.nanpercentile(recon, 99.0)) if recon.size else 1.0
    except Exception:
        # fallback to data min/max
        try:
            vmin = float(np.nanmin(recon))
            vmax = float(np.nanmax(recon))
        except Exception:
            vmin, vmax = 0.0, 1.0

    # guard against NaN/Inf
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0

    # if vmin > vmax, swap, if equal, expand a tiny bit so Normalize is happy :)
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if vmin == vmax:
        vmax = vmin + 1e-6

    ncols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    axes[0].imshow(lensed, origin='lower')
    axes[0].set_title('LENSED')
    axes[0].axis('off')
    axes[1].imshow(recon, origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title('RECON')
    axes[1].axis('off')
    if gt is not None:
        axes[2].imshow(gt, origin='lower', vmin=vmin, vmax=vmax)
        axes[2].set_title('GT')
        axes[2].axis('off')
    plt.tight_layout()
    try:
        fig.savefig(png_path, dpi=150)
    finally:
        plt.close(fig)

def infer_file(fn, model, forward_operator, device, out_dir, rescale=True, save_png=True):
    lensed, gt, hdrs = read_fits_pair(fn)
    eps = 1e-8
    m = max(lensed.max(), eps)
    obs = lensed / m
    if obs.ndim != 2:
        raise RuntimeError("Expected 2D images")
    obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    forward_operator.eval()
    with torch.no_grad():
        # make sure RIM uses the requested iteration count 
        if hasattr(model, 'n_iter') and model.n_iter != N_ITER:
            model.n_iter = N_ITER
        recon_t = model(obs_t, forward_operator)
    recon_np = recon_t.cpu().numpy()[0, 0]
    recon_np = np.clip(recon_np, 0.0, 1.0)

    if rescale:
        recon_out = (recon_np * m).astype(np.float32)
        lensed_out = lensed.astype(np.float32)
        gt_out = gt.astype(np.float32) if gt is not None else None
    else:
        recon_out = recon_np.astype(np.float32)
        lensed_out = (lensed / m).astype(np.float32)
        gt_out = (gt / m).astype(np.float32) if gt is not None else None

    metrics = {}
    if gt is not None:
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
    save_recon_fits(out_fits, lensed_out, recon_out, gt_out, hdrs=hdrs)
    if save_png:
        png_path = os.path.join(out_dir, f"{base}_recon_preview.png")

        # compute vmin/vmax and pass to the saver
        try:
            vmin = float(np.nanpercentile(recon_out, 1.0)) if recon_out.size else 0.0
            vmax = float(np.nanpercentile(recon_out, 99.0)) if recon_out.size else 1.0
        except Exception:
            # fallback to min/max
            try:
                vmin = float(np.nanmin(recon_out)) if recon_out.size else 0.0
                vmax = float(np.nanmax(recon_out)) if recon_out.size else 1.0
            except Exception:
                vmin, vmax = 0.0, 1.0

        # fix invalid cases TODO
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmax = vmin + 1e-6

        # save the preview
        try:
            save_preview_png(png_path, lensed_out, recon_out, gt_out, vmin=vmin, vmax=vmax)
        except Exception as e:
            # don't crash the whole inference for plotting problems; print diagnostics
            print(f"  Warning: failed to save preview PNG for {base}: {e}")
            try:
                p1 = np.nanpercentile(recon_out, 1.0)
                p99 = np.nanpercentile(recon_out, 99.0)
            except Exception:
                p1, p99 = np.nan, np.nan
            try:
                rmin = np.nanmin(recon_out)
                rmax = np.nanmax(recon_out)
            except Exception:
                rmin, rmax = np.nan, np.nan
            print(f"    recon stats: min={rmin}, max={rmax}, p1={p1}, p99={p99}")

    return out_fits, metrics

def extract_forward_state(path):
    """Load forward operator checkpoint (cpu) and return the state_dict and detected kernel size."""
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and ('forward_state' in state or 'forward_operator' in state):    
        if 'forward_state' in state:
            fd = state['forward_state']
        else:
            fd = state['forward_operator']
    elif isinstance(state, dict) and ('raw_psf' in state or 'raw_kernel' in state or 'raw_psf.data' in state):
        fd = state
    else:        
        fd = state

    # detect kernel size from keys raw_psf 
    kernel_size = None
    if isinstance(fd, dict):
        if 'raw_psf' in fd and hasattr(fd['raw_psf'], 'shape'):
            kernel_size = int(fd['raw_psf'].shape[-1])
        elif 'raw_kernel' in fd and hasattr(fd['raw_kernel'], 'shape'):
            kernel_size = int(fd['raw_kernel'].shape[-1])
        else:
            # fallback
            for k, v in fd.items():
                if hasattr(v, 'ndim') and v.ndim == 4:
                    if v.shape[-1] <= 101:  
                        kernel_size = int(v.shape[-1])
                        break
    return fd, kernel_size

def extract_model_state(path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model_state' in state:
        return state['model_state']
    elif isinstance(state, dict) and any(k.startswith('cell') or k.startswith('delta_head') or k.startswith('conv') for k in state.keys()):
        return state
    else:
        return state

def robust_load_state_dict(module, state_dict):
    if not isinstance(state_dict, dict):
        raise RuntimeError("Provided state_dict is not a mapping/dict")

    stripped = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        stripped[new_k] = v

    model_keys = set(module.state_dict().keys())
    matched = {k: v for k, v in stripped.items() if k in model_keys}

    if len(matched) == 0:
        try:
            module.load_state_dict(state_dict, strict=False)
            print("Warning: loaded state_dict with strict=False (no matched keys found in pre-filtering).")
            return
        except Exception as e:
            raise RuntimeError("No matching parameter names found when trying to load checkpoint for module.") from e

    load_info = module.load_state_dict(matched, strict=False)
    missing = load_info.missing_keys if hasattr(load_info, 'missing_keys') else []
    unexpected = load_info.unexpected_keys if hasattr(load_info, 'unexpected_keys') else []

    print(f"robust_load_state_dict: loaded {len(matched)} params for module '{module.__class__.__name__}'")
    if len(missing) > 0:
        print(f"  missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if len(unexpected) > 0:
        print(f"  unexpected keys ignored ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")


def main():
    if DEVICE:
        device = torch.device(DEVICE)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)

    if INPUT_FILE:
        files = [INPUT_FILE]
    else:
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fits")))
    if len(files) == 0:
        print("No files found. Check INPUT_FILE/INPUT_DIR in the USER CONFIG block.")
        return

    sample_lensed, sample_gt, _ = read_fits_pair(files[0])
    H = int(sample_lensed.shape[0])
    W = int(sample_lensed.shape[1])
    print(f"Inferred image size HxW = {H} x {W} from {files[0]}")

    # load forward operator checkpoint to detect kernel size 
    if not os.path.isfile(FORWARD_PATH):
        raise FileNotFoundError(f"Forward operator file not found: {FORWARD_PATH}")
    forward_state_raw, detected_kernel = extract_forward_state(FORWARD_PATH)
    if detected_kernel is None:
        # default to 21 if cannot detect
        detected_kernel = 21
        print("Warning: could not detect kernel size from checkpoint; defaulting kernel_size=21")
    else:
        print(f"Detected forward op kernel size = {detected_kernel}")

    # instantiate models with the right sizes
    forward_operator = PhysicalForward(kernel_size=detected_kernel, device=str(device), enforce_nonneg=True).to(device)
    model = RIMImproved(n_iter=N_ITER).to(device)

    # load forward operator weights
    try:
        fd = forward_state_raw
        if isinstance(fd, dict):
            robust_load_state_dict(forward_operator, fd)
        else:
            # fallback
            ck = torch.load(FORWARD_PATH, map_location='cpu')
            if isinstance(ck, dict) and 'forward_state' in ck:
                robust_load_state_dict(forward_operator, ck['forward_state'])
            elif isinstance(ck, dict) and 'state_dict' in ck:
                robust_load_state_dict(forward_operator, ck['state_dict'])
            else:
                robust_load_state_dict(forward_operator, ck)
        print(f"Loaded forward operator weights from {FORWARD_PATH} (robust mode)")
    except Exception as e:
        try:
            ck = torch.load(FORWARD_PATH, map_location='cpu')
            if isinstance(ck, dict) and 'forward_state' in ck:
                forward_operator.load_state_dict(ck['forward_state'], strict=False)
                print("Loaded forward operator from checkpoint['forward_state'] with strict=False")
            else:
                forward_operator.load_state_dict(ck, strict=False)
                print("Loaded forward operator from checkpoint with strict=False")
        except Exception as e2:
            raise RuntimeError(f"Failed to load forward operator: {e2}") from e

    # load model weights (RIM)
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        mstate = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(mstate, dict) and 'model_state' in mstate:
            ms = mstate['model_state']
        elif isinstance(mstate, dict) and 'model' in mstate and isinstance(mstate['model'], dict):
            ms = mstate['model']
        elif isinstance(mstate, dict) and any(k.startswith('cell') or k.startswith('delta_head') or k.startswith('conv') for k in mstate.keys()):
            ms = mstate
        else:
            # fallback
            ms = mstate

        if isinstance(ms, dict):
            try:
                robust_load_state_dict(model, ms)
                print(f"Loaded RIM model weights from {MODEL_PATH} (robust mode)")
            except Exception:
                model.load_state_dict(ms, strict=False)
                print(f"Loaded RIM model (fallback strict=False) from {MODEL_PATH}")
        else:
            model.load_state_dict(ms)
            print(f"Loaded RIM model weights from {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load RIM model: {e}")

    # Move models to device
    forward_operator.to(device)
    model.to(device)

    # Run inference
    t0 = time.time()
    results = {}
    for fn in files:
        print(f"Processing {fn} ...")
        try:
            out_fits, metrics = infer_file(fn, model, forward_operator, device, OUT_DIR,
                                           rescale=RESCALE_OUTPUT, save_png=SAVE_PNG)
            print(f"  -> wrote {out_fits}")
            if metrics:
                print(f"     MSE: {metrics.get('mse', 'N/A')}, SSIM: {metrics.get('ssim', 'N/A')}")
            results[fn] = metrics
        except Exception as e:
            print(f"  Error processing {fn}: {e}")
            results[fn] = {'error': str(e)}

    dt = time.time() - t0
    print(f"Done. Processed {len(files)} files in {dt:.1f}s")

    # write summary CSV
    try:
        import csv
        csvp = os.path.join(OUT_DIR, 'recon_summary.csv')
        with open(csvp, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'mse', 'ssim', 'error'])
            for f, m in results.items():
                if m is None:
                    w.writerow([f, '', '', ''])
                else:
                    w.writerow([f, m.get('mse', ''), m.get('ssim', ''), m.get('error', '')])
        print(f"Summary CSV written to {csvp}")
    except Exception:
        pass

if __name__ == '__main__':

    main()
