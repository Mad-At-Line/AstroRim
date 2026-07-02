import os
import glob
import json
import time
import argparse
import math
from typing import Optional, Dict

import numpy as np
from astropy.io import fits
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# [FIX R1] single source of truth for the model and physics
from astrorim_core import (
    TRUTH_ORDER, TRUTH_FITS_KEYS, PERIODIC_PI_PARAMS,
    ConditionalPhysicalForward, ConditionalRIM,
    load_checkpoint_file, extract_state_and_meta,
    pack_truth_from_header, squash_param, unsquash_param, param_range_width,
    DEFAULT_PSF_FWHM_RANGE, LEGACY_PSF_FWHM_RANGE,
)

# CONFIG 

INPUT_FILE = None
INPUT_DIR = r"C:\Users\mythi\AstroRIM\AstroRIM_2.1\Sims"
OUT_DIR = r"C:\Users\mythi\AstroRIM\AstroRIM_2.1\sims_recon"
MODEL_PATH = r"C:\Users\mythi\AstroRIM\AstroRIM_2.1\model_holder\cond_rim_2.1_best_update.pt"
FORWARD_PATH = r"C:\Users\mythi\AstroRIM\AstroRIM_2.1\model_holder\cond_forward_2.1_best_update.pt"
DEVICE = None  # 'cpu' or 'cuda' or None auto-detect
KERNEL_SIZE = None
N_ITER = None
RESCALE_OUTPUT = True   # [FIX R3] now actually applied (CLI overrides)
SAVE_PNG = True         # [FIX R3] now actually applied (CLI overrides)
DEBUG_FITS_STRUCTURE = False
USE_NFW = True          # fallback when the checkpoint has no meta
USE_SUBHALOS = False
N_SUBHALOS = 2
USE_PER_OBS_PSF = True
USE_LENS_LIGHT = True


# -----------------------------------------------------------------------------
# FITS I/O (logic preserved from v2.0; primary header now kept for truth)
# -----------------------------------------------------------------------------

def read_fits_pair(fn):
    """Robustly read FITS files from different simulator versions."""
    hdul = fits.open(fn, memmap=False)
    gt = None
    lensed = None
    hdrs = {}
    hdrs['PRIMARY'] = hdul[0].header.copy()

    if DEBUG_FITS_STRUCTURE:
        print(f"  Debug: {os.path.basename(fn)} has {len(hdul)} HDUs:")
        for idx, h in enumerate(hdul):
            name = getattr(h, 'name', '') or h.header.get('EXTNAME', '')
            has_data = h.data is not None
            shape = h.data.shape if has_data else None
            print(f"    HDU[{idx}]: type={type(h).__name__}, name='{name}', "
                  f"has_data={has_data}, shape={shape}")

    # First pass: explicitly named HDUs
    for idx, h in enumerate(hdul):
        name = getattr(h, 'name', '') or h.header.get('EXTNAME', '')
        name = str(name).upper().strip()
        data = h.data
        if name == 'GT' and data is not None:
            gt = data.astype(np.float32)
            hdrs['GT'] = h.header.copy()
        elif name in ('LENSED', 'OBS', 'OBSERVED') and data is not None:
            lensed = data.astype(np.float32)
            hdrs['LENSED'] = h.header.copy()

    # Second pass: v3/v4 format (GT in primary)
    if gt is None and len(hdul) >= 2:
        primary_data = hdul[0].data
        if primary_data is not None:
            if primary_data.ndim > 2:
                primary_data = np.squeeze(primary_data)
            if primary_data.ndim == 2:
                gt = primary_data.astype(np.float32)
                hdrs['GT'] = hdul[0].header.copy()
                print("  [v3/v4 format] GT from Primary[0]")
                if lensed is None:
                    for idx in range(1, len(hdul)):
                        h = hdul[idx]
                        if h.data is not None:
                            second = h.data
                            if second.ndim > 2:
                                second = np.squeeze(second)
                            if second.ndim == 2:
                                lensed = second.astype(np.float32)
                                hdrs['LENSED'] = h.header.copy()
                                print(f"  [v3/v4 format] LENSED from HDU[{idx}]")
                                break

    # Third pass: fallback (real-data single image)
    if lensed is None:
        if hdul[0].data is not None:
            primary_data = hdul[0].data
            if primary_data.ndim > 2:
                primary_data = np.squeeze(primary_data)
            if primary_data.ndim == 2:
                lensed = primary_data.astype(np.float32)
                hdrs['LENSED'] = hdul[0].header.copy()
                print("  [Fallback] Using Primary[0] as LENSED (no GT available)")

    hdul.close()

    if lensed is None:
        raise RuntimeError(f"No valid LENSED data found in {fn}")

    lensed = np.nan_to_num(lensed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if gt is not None:
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if gt.shape != lensed.shape:
            print(f"  Warning: GT shape {gt.shape} != LENSED shape {lensed.shape}, discarding GT")
            gt = None
    if gt is None:
        print("  Note: No GT found - metrics will not be computed")
    return lensed, gt, hdrs


def save_recon_fits(outpath, lensed, recon, gt=None, hdrs=None, pred_params=None):
    primary_hdr = hdrs.get('PRIMARY') if hdrs else None
    prih = fits.PrimaryHDU(header=primary_hdr)
    # [NEW R5] predicted parameters into the primary header (PD_* block)
    if pred_params is not None:
        for name in TRUTH_ORDER:
            key = 'PD' + TRUTH_FITS_KEYS[name][2:]
            try:
                prih.header[key] = (float(pred_params[name]),
                                    f'predicted {name} (model convention)')
            except Exception:
                pass
    hdul = fits.HDUList([prih])
    hdul.append(fits.ImageHDU(data=lensed.astype(np.float32), name='LENSED',
                              header=(hdrs.get('LENSED') if hdrs else None)))
    if gt is not None:
        hdul.append(fits.ImageHDU(data=gt.astype(np.float32), name='GT',
                                  header=(hdrs.get('GT') if hdrs else None)))
    hdul.append(fits.ImageHDU(data=recon.astype(np.float32), name='RECON'))
    hdul.writeto(outpath, overwrite=True)


def save_preview_png(png_path, lensed, recon, gt=None, vmax=None):
    ncols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(lensed, origin='lower')
    axes[0].set_title('LENSED')
    axes[0].axis('off')
    axes[1].imshow(recon, origin='lower', vmax=vmax)
    axes[1].set_title('RECON')
    axes[1].axis('off')
    if gt is not None:
        axes[2].imshow(gt, origin='lower', vmax=vmax)
        axes[2].set_title('GT')
        axes[2].axis('off')
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def create_top8_visualization(results_data, out_dir):
    """Create top-8 visualization (unchanged from v2.0)."""
    valid_results = [(fn, data) for fn, data in results_data.items()
                     if data.get('ssim') is not None and data.get('gt_norm') is not None]
    if len(valid_results) == 0:
        print("  Warning: No valid results with GT to create top-8 visualization")
        return
    valid_results.sort(key=lambda x: x[1]['ssim'], reverse=True)
    top8 = valid_results[:min(8, len(valid_results))]
    n_show = len(top8)
    if n_show == 0:
        return
    avg_ssim = np.mean([data['ssim'] for _, data in top8])
    avg_mse = np.mean([data['mse_norm'] for _, data in top8])
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Top {n_show} Reconstructions by SSIM\n'
                 f'Average SSIM: {avg_ssim:.4f} | Average MSE: {avg_mse:.6f}',
                 fontsize=16, y=0.995)
    for idx, (fn, data) in enumerate(top8):
        basename = os.path.basename(fn)
        lensed_norm = data['lensed_norm']
        gt_norm = data['gt_norm']
        recon_norm = data['recon_norm']
        vmax = np.percentile(np.concatenate([lensed_norm.flatten(),
                                             gt_norm.flatten(),
                                             recon_norm.flatten()]), 99.5)
        axes[idx, 0].imshow(lensed_norm, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        axes[idx, 0].set_title('LENSED', fontsize=10)
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(gt_norm, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        axes[idx, 1].set_title('GT', fontsize=10)
        axes[idx, 1].axis('off')
        axes[idx, 2].imshow(recon_norm, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        axes[idx, 2].set_title('RECON', fontsize=10)
        axes[idx, 2].axis('off')
        text_y = 0.98 - (idx / n_show)
        fig.text(0.5, text_y, f'#{idx + 1}: {basename}', ha='center', va='top',
                 fontsize=9, weight='bold', transform=fig.transFigure)
        fig.text(0.5, text_y - 0.01,
                 f"SSIM: {data['ssim']:.4f} | MSE: {data['mse_norm']:.6f}",
                 ha='center', va='top', fontsize=8, transform=fig.transFigure)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, 'top8_reconstructions.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved top-{n_show} visualization to: {out_path}")


def identify_sim_script(filename):
    """[FIX R4] Map output filenames to the simulator that wrote them, using
    the prefixes the simulators ACTUALLY write (the v2.0 map checked
    rim_simv2_/rim_simv3_/rim_simv5-6_, none of which exist)."""
    basename = os.path.basename(filename)
    prefix_map = [
        ('sim_v1_', 'simgenv1'),
        ('rim_sim2_', 'simgenv2'),
        ('rim_sim3_', 'simgenv3'),
        ('rim_simv4_', 'simgenv4'),
        ('rim_simv5_', 'simgenv5'),
        ('rim_simv6_', 'simgenv6'),
    ]
    for pref, name in prefix_map:
        if basename.startswith(pref):
            return name
    return 'unknown'


# -----------------------------------------------------------------------------
# [NEW R5] Parameter extraction / truth join
# -----------------------------------------------------------------------------

def params_to_floats(lens_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(lens_params[k].detach().cpu().reshape(-1)[0])
            for k in TRUTH_ORDER if k in lens_params}


def derive_physical(pred: Dict[str, float], header) -> Dict[str, float]:
    """Convert normalized predictions to arcsec where a pixel scale is known.
    Priority: TR_PS (sim truth block) > PP_PS_N (normalizer) > PIXSCALE."""
    out = {}
    ps = None
    for key in ('TR_PS', 'PP_PS_N', 'PIXSCALE'):
        if header is not None and key in header:
            try:
                v = float(header[key])
                if np.isfinite(v) and v > 0:
                    ps = v
                    out['pixscale_source'] = key
                    break
            except Exception:
                pass
    if ps is not None:
        half_fov = 48.0 * ps  # 96x96 convention
        out['pixscale_arcsec'] = ps
        out['theta_E_arcsec'] = pred['b'] * half_fov
        out['rs_arcsec'] = pred['rs'] * half_fov
        out['x0_arcsec'] = pred['x0'] * half_fov
        out['y0_arcsec'] = pred['y0'] * half_fov
        out['psf_fwhm_arcsec'] = pred['psf_fwhm'] * ps
    return out


def angle_diff_pi(a, b):
    """Signed difference of two axis-type (period-pi) angles, in (-pi/2, pi/2]."""
    d = (a - b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return d


def write_param_outputs(out_dir, rows):
    """Combined predicted_params.csv (+ predicted_vs_true.csv / scatter grid
    when any truth rows exist)."""
    import csv
    if not rows:
        return
    csvp = os.path.join(out_dir, 'predicted_params.csv')
    pred_cols = [f'pred_{k}' for k in TRUTH_ORDER]
    extra = ['theta_E_arcsec', 'psf_fwhm_arcsec', 'pixscale_arcsec', 'pixscale_source']
    with open(csvp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'sim_script', 'ssim', 'mse_norm', 'fwd_fid',
                    'fwd_fid_refined'] + pred_cols + extra)
        for r in rows:
            w.writerow([r['file'], r['sim_script'], r.get('ssim', ''),
                        r.get('mse_norm', ''), r.get('fwd_fid', ''),
                        r.get('fwd_fid_refined', '')] +
                       [r['pred'].get(k, '') for k in TRUTH_ORDER] +
                       [r.get('phys', {}).get(c, '') for c in extra])
    print(f"Predicted-parameter CSV written to {csvp}")

    truth_rows = [r for r in rows if r.get('truth') is not None]
    if not truth_rows:
        return
    csvt = os.path.join(out_dir, 'predicted_vs_true.csv')
    with open(csvt, 'w', newline='') as fh:
        w = csv.writer(fh)
        head = ['file', 'sim_script', 'clean']
        for k in TRUTH_ORDER:
            head += [f'true_{k}', f'pred_{k}', f'mask_{k}']
        w.writerow(head)
        for r in truth_rows:
            row = [r['file'], r['sim_script'], r['truth_info'].get('clean', '')]
            for i, k in enumerate(TRUTH_ORDER):
                row += [r['truth'][i], r['pred'].get(k, ''), int(r['truth_mask'][i])]
            w.writerow(row)
    print(f"Predicted-vs-true CSV written to {csvt}  ({len(truth_rows)} files with truth)")

    # Scatter grid: one panel per parameter, identity line, mask filtering.
    ncols = 4
    nrows = int(math.ceil(len(TRUTH_ORDER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes)
    for i, k in enumerate(TRUTH_ORDER):
        ax = axes[i // ncols, i % ncols]
        xs, ys = [], []
        for r in truth_rows:
            if r['truth_mask'][i] > 0 and k in r['pred']:
                xs.append(float(r['truth'][i]))
                ys.append(float(r['pred'][k]))
        if xs:
            xs = np.array(xs)
            ys = np.array(ys)
            if k in PERIODIC_PI_PARAMS:
                # plot true vs (true + wrapped residual): visualizes the
                # period-pi error without 0/pi wrap artifacts
                res = np.array([angle_diff_pi(y, x) for x, y in zip(xs, ys)])
                ax.scatter(xs, xs + res, s=14, alpha=0.6)
                med = float(np.median(np.abs(res)))
                ax.set_title(f"{k}  (med |d| = {med:.3f} rad)", fontsize=10)
            else:
                ax.scatter(xs, ys, s=14, alpha=0.6)
                err = ys - xs
                ax.set_title(f"{k}  (bias {np.median(err):+.3g}, "
                             f"MAD {np.median(np.abs(err - np.median(err))):.3g})",
                             fontsize=10)
            lo = min(xs.min(), ys.min())
            hi = max(xs.max(), ys.max())
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1, alpha=0.6)
            ax.set_xlim(lo - pad, hi + pad)
        else:
            ax.set_title(f"{k}  (no supervised truth)", fontsize=10)
        ax.set_xlabel('true')
        ax.set_ylabel('predicted')
        ax.grid(alpha=0.25)
    for j in range(len(TRUTH_ORDER), nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
    fig.suptitle('Predicted vs true parameters (mask-filtered)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, 'predicted_vs_true.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Predicted-vs-true scatter grid written to {out_path}")


# -----------------------------------------------------------------------------
# [NEW R6] Test-time MAP refinement
# -----------------------------------------------------------------------------

def map_refine(model, forward_operator, obs_t, lens_params0, subhalo_params,
               steps: int, rounds: int, lr: float = 0.05,
               prior_weight: float = 1e-3, positivity: bool = False,
               verbose: bool = True):
    """Per-image MAP refinement of the conditioning parameters.

    Alternates: (1) RIM reconstruction with current parameters (fixed), then
    (2) `steps` Adam updates of the RAW (unbounded) parameter vector on
        || F_theta(x_hat; params) - obs ||^2
        + prior_weight * ||raw - raw_encoder_init||^2,
    re-squashing through the SAME bounds the encoder uses, so refined values
    can never leave the physical ranges. Returns (refined lens_params dict,
    final recon tensor, fidelity_before, fidelity_after).
    """
    device = obs_t.device
    psf_range = forward_operator.psf_fwhm_range

    def positivity_fn(x):
        return F.softplus(x) if positivity else x

    # encoder init -> raw space
    raw0 = {k: unsquash_param(k, lens_params0[k].detach(),
                              psf_fwhm_range=psf_range) for k in TRUTH_ORDER}
    raws = {k: v.clone().requires_grad_(True) for k, v in raw0.items()}

    def squash_all():
        return {k: squash_param(k, raws[k], psf_fwhm_range=psf_range)
                for k in TRUTH_ORDER}

    with torch.no_grad():
        recon = positivity_fn(model(obs_t, obs_t, forward_operator,
                                    lens_params=lens_params0,
                                    subhalo_params=subhalo_params))
        fid_before = float(F.mse_loss(
            forward_operator(recon, obs_t, lens_params=lens_params0,
                             subhalo_params=subhalo_params), obs_t))

    fid_after = fid_before
    current = lens_params0
    for rnd in range(rounds):
        with torch.no_grad():
            recon = positivity_fn(model(obs_t, obs_t, forward_operator,
                                        lens_params=current,
                                        subhalo_params=subhalo_params))
        x_fixed = recon.detach()
        opt = torch.optim.Adam(list(raws.values()), lr=lr)
        for s in range(steps):
            opt.zero_grad()
            params = squash_all()
            pred = forward_operator(x_fixed, obs_t, lens_params=params,
                                    subhalo_params=subhalo_params)
            loss = F.mse_loss(pred, obs_t)
            prior = sum(((raws[k] - raw0[k]) ** 2).mean() for k in TRUTH_ORDER)
            (loss + prior_weight * prior).backward()
            opt.step()
        current = {k: v.detach() for k, v in squash_all().items()}
        with torch.no_grad():
            recon = positivity_fn(model(obs_t, obs_t, forward_operator,
                                        lens_params=current,
                                        subhalo_params=subhalo_params))
            fid_after = float(F.mse_loss(
                forward_operator(recon, obs_t, lens_params=current,
                                 subhalo_params=subhalo_params), obs_t))
        if verbose:
            print(f"    MAP round {rnd + 1}/{rounds}: forward fidelity "
                  f"{fid_before:.4e} -> {fid_after:.4e}")
    return current, recon, fid_before, fid_after


# -----------------------------------------------------------------------------
# Inference per file
# -----------------------------------------------------------------------------

def infer_file(fn, model, forward_operator, device, out_dir, rescale=True,
               save_png=True, positivity=False, map_steps=0, map_rounds=1):
    lensed, gt, hdrs = read_fits_pair(fn)
    eps = 1e-8

    # Normalize (same convention as training: max over GT + LENSED)
    if gt is not None:
        normalization_factor = max(lensed.max(), gt.max(), eps)
    else:
        normalization_factor = max(lensed.max(), eps)
    lensed_norm = lensed / normalization_factor
    gt_norm = gt / normalization_factor if gt is not None else None
    if lensed_norm.ndim != 2:
        raise RuntimeError("Expected 2D images")

    obs_t = torch.from_numpy(lensed_norm.astype(np.float32)) \
        .unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    forward_operator.eval()

    fwd_fid = fwd_fid_refined = None
    with torch.no_grad():
        lens_params, subhalo_params = forward_operator.encode_lens_parameters(obs_t)

    if map_steps > 0:                                            # [NEW R6]
        lens_params, recon_t, fwd_fid, fwd_fid_refined = map_refine(
            model, forward_operator, obs_t, lens_params, subhalo_params,
            steps=map_steps, rounds=map_rounds, positivity=positivity)
    else:
        with torch.no_grad():
            recon_t = model(obs_t, obs_t, forward_operator,
                            lens_params=lens_params, subhalo_params=subhalo_params)
            if positivity:                                       # [NEW R7]
                recon_t = F.softplus(recon_t)
            fwd_fid = float(F.mse_loss(
                forward_operator(recon_t, obs_t, lens_params=lens_params,
                                 subhalo_params=subhalo_params), obs_t))

    recon_norm = recon_t.detach().cpu().numpy()[0, 0]
    recon_norm = np.clip(recon_norm, 0.0, 1.0)

    pred = params_to_floats(lens_params)
    phys = derive_physical(pred, hdrs.get('PRIMARY'))

    # truth (if the file carries the v2.1 TR_* block)            [NEW R5]
    truth = truth_mask = None
    truth_info = {}
    try:
        tvec, tmask, tinfo = pack_truth_from_header(hdrs['PRIMARY'])
        if tmask.sum() > 0:
            truth, truth_mask, truth_info = tvec, tmask, tinfo
    except Exception:
        pass

    metrics = {}
    if gt_norm is not None:
        try:
            metrics['mse_norm'] = float(np.mean((gt_norm - recon_norm) ** 2))
            metrics['ssim'] = float(ssim(gt_norm, recon_norm, data_range=1.0))
            gt_orig = gt_norm * normalization_factor
            recon_orig = recon_norm * normalization_factor
            metrics['mse_orig'] = float(np.mean((gt_orig - recon_orig) ** 2))
        except Exception as e:
            print(f"Warning: Error computing metrics for {fn}: {e}")
            metrics['mse_norm'] = metrics['mse_orig'] = metrics['ssim'] = None

    if rescale:
        lensed_out = lensed.astype(np.float32)
        recon_out = (recon_norm * normalization_factor).astype(np.float32)
        gt_out = gt.astype(np.float32) if gt is not None else None
    else:
        lensed_out = lensed_norm.astype(np.float32)
        recon_out = recon_norm.astype(np.float32)
        gt_out = gt_norm.astype(np.float32) if gt_norm is not None else None

    base = os.path.splitext(os.path.basename(fn))[0]
    out_fits = os.path.join(out_dir, f"{base}_recon.fits")
    save_recon_fits(out_fits, lensed_out, recon_out, gt_out, hdrs=hdrs,
                    pred_params=pred)

    # [NEW R5] per-file JSON with predictions (+ truth join when available)
    pjson = {'file': fn, 'sim_script': identify_sim_script(fn),
             'normalization_factor': float(normalization_factor),
             'predicted': pred, 'derived': phys,
             'forward_fidelity': fwd_fid,
             'forward_fidelity_refined': fwd_fid_refined,
             'metrics': metrics}
    if truth is not None:
        pjson['truth'] = {k: float(truth[i]) for i, k in enumerate(TRUTH_ORDER)}
        pjson['truth_mask'] = {k: int(truth_mask[i]) for i, k in enumerate(TRUTH_ORDER)}
        pjson['truth_info'] = truth_info
    with open(os.path.join(out_dir, f"{base}_params.json"), 'w') as fh:
        json.dump(pjson, fh, indent=2)

    if save_png:
        png_path = os.path.join(out_dir, f"{base}_recon_preview.png")
        vmax = np.percentile(recon_out, 99.0)
        save_preview_png(png_path, lensed_out, recon_out, gt_out, vmax=vmax)

    result_data = metrics.copy()
    result_data['lensed_norm'] = lensed_norm
    result_data['gt_norm'] = gt_norm
    result_data['recon_norm'] = recon_norm
    result_data['pred'] = pred
    result_data['phys'] = phys
    result_data['fwd_fid'] = fwd_fid
    result_data['fwd_fid_refined'] = fwd_fid_refined
    result_data['truth'] = truth
    result_data['truth_mask'] = truth_mask
    result_data['truth_info'] = truth_info
    result_data['sim_script'] = identify_sim_script(fn)
    result_data['file'] = fn
    return out_fits, result_data


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference for conditional RIM + physical forward operator (v2.1)")
    p.add_argument('--input-file', type=str, default=None)
    p.add_argument('--input-dir', type=str, default=None)
    p.add_argument('--out-dir', type=str, default=None)
    p.add_argument('--model-path', type=str, default=None)
    p.add_argument('--forward-path', type=str, default=None)
    p.add_argument('--kernel-size', type=int, default=None,
                   help='default: checkpoint meta, else 21')
    p.add_argument('--n-iter', type=int, default=None,
                   help='default: checkpoint meta, else 10')
    p.add_argument('--hidden-dim', type=int, default=None,
                   help='default: checkpoint meta, else 128')
    p.add_argument('--device', type=str, default=None, help='cpu or cuda')

    # [FIX R3] proper boolean handling; None = fall back to USER CONFIG/meta
    p.add_argument('--rescale', action=argparse.BooleanOptionalAction, default=None,
                   help='Write outputs in original (un-normalized) units '
                        f'(USER CONFIG default: {RESCALE_OUTPUT})')
    p.add_argument('--save-png', action=argparse.BooleanOptionalAction, default=None,
                   help='Write per-file preview PNGs '
                        f'(USER CONFIG default: {SAVE_PNG}). The v2.0 --no-png '
                        'flag was inverted and PNGs were never saved.')
    p.add_argument('--use-nfw', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--use-subhalos', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--n-subhalos', type=int, default=None)
    p.add_argument('--use-per-obs-psf', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--use-lens-light', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--param-mode', type=str, default=None, choices=['encoder', 'global'])
    p.add_argument('--source-positivity', action=argparse.BooleanOptionalAction,
                   default=None, help='default: checkpoint meta [NEW R7]')

    # [FIX R2]
    p.add_argument('--psf-fwhm-min', type=float, default=None,
                   help='default: checkpoint meta, else 0.3. Pass 1.0 for '
                        'legacy v2.0 checkpoints (their encoder FWHM floor).')
    p.add_argument('--psf-fwhm-max', type=float, default=None)
    p.add_argument('--allow-partial-load', action='store_true', default=False)

    # [NEW R6]
    p.add_argument('--map-refine', type=int, default=0, metavar='STEPS',
                   help='Test-time MAP refinement: Adam steps per round on the '
                        'forward-fidelity objective (0 = off)')
    p.add_argument('--map-rounds', type=int, default=1)
    return p.parse_args()


def _resolve(cli_value, meta_value, config_value, hard_default):
    """CLI (if given) > checkpoint meta > USER CONFIG constant > default."""
    if cli_value is not None:
        return cli_value
    if meta_value is not None:
        return meta_value
    if config_value is not None:
        return config_value
    return hard_default


def main():
    args = parse_args()

    # Path config: CLI > USER CONFIG
    input_file = args.input_file if args.input_file is not None else INPUT_FILE
    input_dir = args.input_dir if args.input_dir is not None else INPUT_DIR
    out_dir = args.out_dir if args.out_dir is not None else OUT_DIR
    model_path = args.model_path if args.model_path is not None else MODEL_PATH
    forward_path = args.forward_path if args.forward_path is not None else FORWARD_PATH

    if args.device:
        device = torch.device(args.device)
    elif DEVICE:
        device = torch.device(DEVICE)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # [FIX R2] read checkpoint meta BEFORE building the model so it can drive
    # the configuration.
    meta = {}
    if forward_path and os.path.isfile(forward_path):
        try:
            blob = torch.load(forward_path, map_location='cpu')
            _, meta = extract_state_and_meta(blob)
        except Exception as e:
            print(f"Note: could not pre-read checkpoint meta: {e}")
    if meta:
        print(f"Checkpoint meta found (v2.1 trainer): {json.dumps(meta, default=str)}")
    else:
        print("No checkpoint meta (pre-v2.1 checkpoint): falling back to USER "
              "CONFIG constants. If this checkpoint was trained with the legacy "
              "PSF bounds, pass --psf-fwhm-min 1.0.")

    use_nfw = _resolve(args.use_nfw, meta.get('use_nfw'), USE_NFW, True)
    use_subhalos = _resolve(args.use_subhalos, meta.get('use_subhalos'), USE_SUBHALOS, False)
    n_subhalos = _resolve(args.n_subhalos, meta.get('n_subhalos'), N_SUBHALOS, 2)
    use_per_obs_psf = _resolve(args.use_per_obs_psf, meta.get('use_per_obs_psf'),
                               USE_PER_OBS_PSF, True)
    use_lens_light = _resolve(args.use_lens_light, meta.get('use_lens_light'),
                              USE_LENS_LIGHT, True)
    param_mode = _resolve(args.param_mode, meta.get('param_mode'), None, 'encoder')
    kernel_size = _resolve(args.kernel_size, meta.get('kernel_size'), KERNEL_SIZE, 21)
    n_iter = _resolve(args.n_iter, meta.get('n_iter'), N_ITER, 10)
    hidden_dim = _resolve(args.hidden_dim, meta.get('hidden_dim'), None, 128)
    use_fft_conv = bool(meta.get('use_fft_conv', True))
    positivity = bool(_resolve(args.source_positivity,
                               meta.get('source_positivity'), None, False))
    meta_range = meta.get('psf_fwhm_range')
    psf_min = _resolve(args.psf_fwhm_min,
                       (meta_range[0] if meta_range else None), None,
                       DEFAULT_PSF_FWHM_RANGE[0])
    psf_max = _resolve(args.psf_fwhm_max,
                       (meta_range[1] if meta_range else None), None,
                       DEFAULT_PSF_FWHM_RANGE[1])
    rescale = bool(_resolve(args.rescale, None, RESCALE_OUTPUT, True))
    save_png = bool(_resolve(args.save_png, None, SAVE_PNG, True))

    print(f"Using device: {device}")
    print(f"NFW: {use_nfw} | Subhalos: {use_subhalos} | Per-obs PSF: {use_per_obs_psf} "
          f"| Lens light: {use_lens_light} | Param mode: {param_mode}")
    print(f"PSF FWHM range: ({psf_min}, {psf_max}) px | Source positivity: {positivity}")
    print(f"Rescale output: {rescale} | Save PNG: {save_png}"
          + (f" | MAP refine: {args.map_refine} steps x {args.map_rounds} rounds"
             if args.map_refine > 0 else ""))

    os.makedirs(out_dir, exist_ok=True)

    forward_operator = ConditionalPhysicalForward(
        kernel_size=kernel_size,
        enforce_nonneg=True,
        pixel_scale=0.05,
        learn_residual_psf=True,
        residual_scale=1e-2,
        use_fft_conv=use_fft_conv,
        use_nfw=use_nfw,
        use_subhalos=use_subhalos,
        n_subhalos=n_subhalos,
        use_per_obs_psf=use_per_obs_psf,
        use_lens_light=use_lens_light,
        psf_fwhm_range=(psf_min, psf_max),
        param_mode=param_mode,
    ).to(device)
    model = ConditionalRIM(n_iter=n_iter, hidden_dim=hidden_dim).to(device)

    # [FIX R2] strict loading with diagnostics
    if not (forward_path and os.path.isfile(forward_path)):
        raise FileNotFoundError(f"Forward operator file not found: {forward_path}")
    load_checkpoint_file(forward_operator, forward_path, label='forward',
                         prefer=('forward_state', 'state_dict'),
                         allow_partial=args.allow_partial_load, map_location=device)
    print(f"Loaded forward operator weights from {forward_path}")

    if not (model_path and os.path.isfile(model_path)):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    load_checkpoint_file(model, model_path, label='RIM',
                         prefer=('model_state', 'state_dict'),
                         allow_partial=args.allow_partial_load, map_location=device)
    print(f"Loaded RIM model weights from {model_path}")

    if input_file:
        files = [input_file]
    else:
        files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
        files = [f for f in files if not f.endswith('_recon.fits')]
    if len(files) == 0:
        print("No files found. Check input-file/input-dir.")
        return

    print(f"\nProcessing {len(files)} files...")
    t0 = time.time()
    results = {}
    param_rows = []
    for fn in files:
        print(f"Processing {fn} ...")
        try:
            out_fits, rd = infer_file(fn, model, forward_operator, device, out_dir,
                                      rescale=rescale, save_png=save_png,
                                      positivity=positivity,
                                      map_steps=args.map_refine,
                                      map_rounds=args.map_rounds)
            print(f"  -> wrote {out_fits}")
            if rd.get('mse_norm') is not None:
                print(f"     MSE (norm): {rd['mse_norm']:.6f}, "
                      f"SSIM: {rd.get('ssim', float('nan')):.4f}, "
                      f"fwd fid: {rd.get('fwd_fid', float('nan')):.4e}")
            results[fn] = rd
            param_rows.append(rd)
        except Exception as e:
            print(f"  Error processing {fn}: {e}")
            results[fn] = {'error': str(e)}

    dt = time.time() - t0
    print(f"\nDone. Processed {len(files)} files in {dt:.1f}s")

    try:
        create_top8_visualization(results, out_dir)
    except Exception as e:
        print(f"Warning: Could not create top-8 visualization: {e}")

    try:
        write_param_outputs(out_dir, [r for r in param_rows if 'pred' in r])
    except Exception as e:
        print(f"Warning: Could not write parameter outputs: {e}")

    try:
        import csv
        csvp = os.path.join(out_dir, 'recon_summary.csv')
        with open(csvp, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'sim_script', 'mse_norm', 'mse_orig', 'ssim',
                        'fwd_fid', 'fwd_fid_refined', 'error'])
            for f, m in results.items():
                if m is None or 'error' in m:
                    w.writerow([f, identify_sim_script(f), '', '', '', '', '',
                                m.get('error', 'Unknown error') if m else 'Unknown'])
                else:
                    w.writerow([f, m.get('sim_script', ''), m.get('mse_norm', ''),
                                m.get('mse_orig', ''), m.get('ssim', ''),
                                m.get('fwd_fid', ''), m.get('fwd_fid_refined', ''), ''])
        print(f"Summary CSV written to {csvp}")
    except Exception as e:
        print(f"Warning: Could not write CSV: {e}")


if __name__ == '__main__':
    main()
