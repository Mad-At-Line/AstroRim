import os
import glob
import json
import time
import argparse
import math
from datetime import datetime
from typing import Optional, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# [FIX T1] single source of truth for the model and physics
from astrorim_core import (
    CORE_VERSION, TRUTH_ORDER,
    ConditionalPhysicalForward, ConditionalRIM, ShadowEMA,
    ConditionalLensingFitsDataset,
    load_checkpoint_file, CHECKPOINT_META_KEY,
    supervised_param_loss, DEFAULT_PSF_FWHM_RANGE,
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def compute_grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            try:
                total += float(p.grad.data.norm(2)) ** 2
            except Exception:
                pass
    return float(total ** 0.5)


def total_variation(x):
    """Isotropic TV (mean |grad|) of a (B,1,H,W) image. [NEW T9]"""
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def apply_positivity(x, enabled: bool):
    """[NEW T9] softplus positivity on the source estimate."""
    return F.softplus(x) if enabled else x


def multires_fidelity(pred, obs):
    """[NEW T10] forward fidelity at 1x, 2x, 4x average-pooled resolution."""
    loss = F.mse_loss(pred, obs)
    for k in (2, 4):
        loss = loss + F.mse_loss(F.avg_pool2d(pred, k), F.avg_pool2d(obs, k))
    return loss / 3.0


def recon_pixel_loss(recon, gt, mode='balanced', fg_thresh=0.05, fg_boost=3.0):
    """[FIX T11] Reconstruction pixel loss.

    The sources are ~1% of pixels (see frac>0.1 ~ 0.01-0.02 in the diagnostic),
    so plain MSE is dominated by the empty background: the trivial minimum is a
    blank source, and under softplus that minimum sits at raw ~ -34 where the
    activation gradient is dead -> permanent collapse. 'balanced' averages the
    error over foreground and background pixels SEPARATELY and then sums them,
    so the sparse source carries weight comparable to the background regardless
    of how few pixels it occupies (fg_boost adds extra source emphasis). This
    moves the constant-field equilibrium to a small negative value where the
    network can still place flux. 'mse' restores the legacy behaviour.
    """
    if mode == 'mse':
        return F.mse_loss(recon, gt)
    err2 = (recon - gt) ** 2
    fg = (gt > fg_thresh).float()
    bg = 1.0 - fg
    n_fg = fg.sum().clamp_min(1.0)
    n_bg = bg.sum().clamp_min(1.0)
    l_fg = (err2 * fg).sum() / n_fg
    l_bg = (err2 * bg).sum() / n_bg
    return l_bg + fg_boost * l_fg


def nonneg_penalty(recon):
    """[FIX T11] Soft non-negativity on a LINEAR reconstruction. Replaces the
    saturating softplus positivity: relu(-x) has a constant gradient on the
    violating side, so it discourages negative surface brightness without the
    dead-gradient collapse mode softplus introduces."""
    return F.relu(-recon).mean()


def save_reconstructions(epoch, model, forward_operator, val_loader, device,
                         positivity, num_examples=6, out_dir='recons'):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    forward_operator.eval()
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        print("Validation loader empty; skipping reconstructions.")
        return
    obs, gt = batch[0], batch[1]
    obs = obs.to(device)
    gt = gt.to(device)
    with torch.no_grad():
        recon = apply_positivity(model(obs, obs, forward_operator), positivity)
    n = min(obs.shape[0], num_examples)
    obs_np, recon_np, gt_np = obs.cpu().numpy(), recon.cpu().numpy(), gt.cpu().numpy()
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for i in range(n):
        for j, (img, title) in enumerate([(obs_np[i, 0], 'Observed'),
                                          (recon_np[i, 0], 'Reconstruction'),
                                          (gt_np[i, 0], 'Ground Truth')]):
            axes[i, j].imshow(np.clip(img, 0.0, 1.0), origin='lower', cmap='viridis')
            axes[i, j].set_title(title)
            axes[i, j].axis('off')
    plt.suptitle(f'Conditional Reconstructions - Epoch {epoch}', fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'cond_recons_epoch_{epoch}.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved reconstructions to {out_path}")


def lr_scale_warmup(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def lambda_fwd_schedule(epoch: int, ramp_epochs: int, target: float) -> float:
    if ramp_epochs <= 0:
        return target
    if epoch < ramp_epochs:
        return target * (epoch + 1) / ramp_epochs
    return target


def make_grad_scaler(device, enabled):
    """Modern AMP API with fallback for older torch. [KEPT behavior]"""
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(device, enabled):
    try:
        return torch.amp.autocast(device_type='cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def build_checkpoint_meta(args, psf_range):
    """[FIX T5/NEW T9] config recorded inside every checkpoint so inference
    scripts can reproduce the exact run configuration (loaders warn on
    psf_fwhm_range mismatch; recon_gen applies source_positivity from meta)."""
    return {
        'core_version': CORE_VERSION,
        'psf_fwhm_range': list(psf_range),
        'param_mode': args.param_mode,
        'use_nfw': args.use_nfw,
        'use_subhalos': args.use_subhalos,
        'n_subhalos': args.n_subhalos,
        'use_per_obs_psf': args.use_per_obs_psf,
        'use_lens_light': args.use_lens_light,
        'use_fft_conv': args.use_fft_conv,
        'kernel_size': args.kernel_size,
        'n_iter': args.n_iter,
        'hidden_dim': args.hidden_dim,
        'source_positivity': args.source_positivity,
        'saved_utc': datetime.utcnow().isoformat(),
    }


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def conditional_finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    psf_range = (args.psf_fwhm_min, args.psf_fwhm_max)
    use_param_sup = args.lambda_param_sup > 0.0

    print(f"\n{'=' * 80}")
    print("CONDITIONAL TRAINING/FINETUNING WITH PER-LENS PARAMETERS (v2.1)")
    print(f"{'=' * 80}\n")
    print(f"Device: {device}")
    print(f"Param mode: {args.param_mode}"
          + ("  [ABLATION: one shared learned parameter set]" if args.param_mode == 'global' else ""))
    print(f"Forward fidelity target weight: lambda = {args.lambda_forward_fidelity:.2e}"
          + ("  (multi-resolution)" if args.multires_fwd else ""))
    print(f"Param supervision weight: {args.lambda_param_sup:.2e}"
          + ("  [TR_* truth headers required in FITS]" if use_param_sup else "  (off)"))
    print(f"Deep supervision: {args.deep_supervision} | Source positivity: {args.source_positivity}"
          f" | lambda_TV: {args.lambda_tv:.2e}")
    print(f"Recon loss: {args.recon_loss}"
          + (f" (fg_thresh={args.fg_thresh}, fg_boost={args.fg_boost})"
             if args.recon_loss == 'balanced' else "")
          + f" | lambda_nonneg: {args.lambda_nonneg:.2e}"
          + f" | lambda_l1: {args.lambda_l1:.2e}"
          + ("  [LINEAR output -- no softplus]" if not args.source_positivity
             else "  [softplus positivity]"))
    print(f"PSF FWHM range: {psf_range} px"
          + ("  [NOTE: legacy v2.0 checkpoints used (1.0, 25.0)]"
             if tuple(psf_range) != (1.0, 25.0) else ""))
    print(f"Warmup epochs: {args.warmup_epochs} | EMA: {args.use_ema} (decay {args.ema_decay})")
    print(f"{'=' * 80}\n")

    # Dataset --------------------------------------------------------------
    all_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.fits")))
    if len(all_files) == 0:
        raise RuntimeError(f"No .fits files found in {args.dataset_dir}")

    train_files, val_files = train_test_split(all_files, test_size=args.val_split,
                                              random_state=42)

    train_dataset = ConditionalLensingFitsDataset(train_files, augment=args.augment,
                                                  return_truth=use_param_sup)
    val_dataset = ConditionalLensingFitsDataset(val_files, augment=False,
                                                return_truth=use_param_sup)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type == 'cuda'),
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=(device.type == 'cuda'),
                            num_workers=args.num_workers)

    # Models ----------------------------------------------------------------
    forward_operator = ConditionalPhysicalForward(
        kernel_size=args.kernel_size,
        enforce_nonneg=True,
        pixel_scale=0.05,
        learn_residual_psf=True,
        residual_scale=1e-2,
        use_fft_conv=args.use_fft_conv,        # [FIX T3]
        use_nfw=args.use_nfw,
        use_subhalos=args.use_subhalos,
        n_subhalos=args.n_subhalos,
        use_per_obs_psf=args.use_per_obs_psf,
        use_lens_light=args.use_lens_light,
        psf_fwhm_range=psf_range,              # [FIX T5]
        param_mode=args.param_mode,            # [NEW T7]
    ).to(device)

    model = ConditionalRIM(n_iter=args.n_iter, hidden_dim=args.hidden_dim).to(device)

    start_epoch = 0
    best_val_loss = float('inf')

    # [FIX T2] strict checkpoint loading with diagnostics
    if args.rim_checkpoint and os.path.exists(args.rim_checkpoint):
        print(f"Loading RIM weights from {args.rim_checkpoint}")
        load_checkpoint_file(model, args.rim_checkpoint, label='RIM',
                             prefer=('model_state', 'state_dict'),
                             allow_partial=args.allow_partial_load)
    if args.forward_checkpoint and os.path.exists(args.forward_checkpoint):
        print(f"Loading Forward operator weights from {args.forward_checkpoint}")
        load_checkpoint_file(forward_operator, args.forward_checkpoint, label='forward',
                             prefer=('forward_state', 'state_dict'),
                             allow_partial=args.allow_partial_load)

    # Freeze logic -----------------------------------------------------------
    if args.finetune_mode == 'rim':
        for p in forward_operator.parameters():
            p.requires_grad = False
        print("Freezing forward operator (finetuning RIM only)")
    elif args.finetune_mode == 'forward':
        for p in model.parameters():
            p.requires_grad = False
        print("Freezing RIM (finetuning forward operator only)")
    else:
        print("Finetuning both RIM and forward operator")

    if args.freeze_psf:
        print("Freezing PSF parameters in forward operator")
        for name, p in forward_operator.named_parameters():
            if 'raw_res_psf' in name or 'psf_' in name:
                p.requires_grad = False

    # Optimizer ---------------------------------------------------------------
    params = []
    if args.finetune_mode in ['both', 'rim']:
        rim_params = [p for p in model.parameters() if p.requires_grad]
        if rim_params:
            params.append({"params": rim_params, "lr": args.lr_rim,
                           "initial_lr": args.lr_rim})
    if args.finetune_mode in ['both', 'forward']:
        forward_params = [p for p in forward_operator.parameters() if p.requires_grad]
        if forward_params:
            params.append({"params": forward_params, "lr": args.lr_forward,
                           "initial_lr": args.lr_forward})
    if len(params) == 0:
        raise RuntimeError("No parameters to optimize (check finetune_mode / freeze flags)")

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience,
            verbose=True)
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)

    scaler = make_grad_scaler(device, enabled=(device.type == 'cuda' and args.use_amp))
    ema = ShadowEMA([model, forward_operator], decay=args.ema_decay) if args.use_ema else None

    # Per-parameter weights for the supervised loss [NEW T6]: lens-light
    # labels are weakly identified (best circular-family approximations of an
    # elliptical, slightly offset truth), so they get reduced weight.
    sup_weights = {'lens_flux': 0.25, 'lens_Re': 1, 'lens_n': 0.25}

    loss_history = {
        "train": [], "val": [], "val_ssim": [], "grad_norm": [],
        "forward_fidelity": [], "param_reg": [], "lambda_fwd": [],
        "param_sup": [], "val_src_mse": [], "skipped": [],
    }
    per_param_running = {}

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        ck = torch.load(args.resume_checkpoint, map_location='cpu')
        start_epoch = ck.get('epoch', 0)
        if 'optimizer' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer'])
                print("Loaded optimizer state")
            except Exception as e:
                print(f"Warning: Failed to load optimizer: {e}")
        if 'best_val_loss' in ck:
            best_val_loss = ck['best_val_loss']
        print(f"Resuming from epoch {start_epoch}")

    meta = build_checkpoint_meta(args, psf_range)

    print("\n" + "=" * 80)
    print("STARTING CONDITIONAL TRAINING/FINETUNING")
    print("=" * 80 + "\n")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_idx = epoch + 1
        start_epoch_time = time.time()

        warm_scale = lr_scale_warmup(epoch, args.warmup_epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * warm_scale
        lam_fwd = lambda_fwd_schedule(epoch, args.warmup_epochs,
                                      args.lambda_forward_fidelity)

        model.train()
        forward_operator.train()

        running = dict(loss=0.0, fwd=0.0, reg=0.0, sup=0.0)
        grad_norm_epoch = 0.0
        per_param_epoch = {}
        n_pp = 0
        n_skipped = 0

        loop = tqdm(train_loader,
                    desc=f"Epoch {epoch_idx}/{args.num_epochs} "
                         f"(warm={warm_scale:.2f}, lam_fwd={lam_fwd:.3f})")
        for bidx, batch in enumerate(loop):
            if args.limit_train_batches and bidx >= args.limit_train_batches:
                break
            if use_param_sup:
                obs, gt, tvec, tmask = batch
                tvec = tvec.to(device)
                tmask = tmask.to(device)
            else:
                obs, gt = batch[0], batch[1]
            obs = obs.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            with autocast_ctx(device, enabled=(device.type == 'cuda' and args.use_amp)):
                # --- reconstruction (+ optional deep supervision [NEW T8]) ---
                if args.deep_supervision:
                    recon_raw, inters = model(obs, obs, forward_operator,
                                              return_intermediates=True)
                    recon = apply_positivity(recon_raw, args.source_positivity)
                    T = len(inters)
                    w_sum = 1.0
                    mse_recon = recon_pixel_loss(recon, gt, args.recon_loss,
                                                 args.fg_thresh, args.fg_boost)
                    for t, xt in enumerate(inters):
                        w = (t + 1) / T
                        mse_recon = mse_recon + w * recon_pixel_loss(
                            apply_positivity(xt, args.source_positivity), gt,
                            args.recon_loss, args.fg_thresh, args.fg_boost)
                        w_sum += w
                    mse_recon = mse_recon / w_sum
                else:
                    recon = apply_positivity(model(obs, obs, forward_operator),
                                             args.source_positivity)
                    mse_recon = recon_pixel_loss(recon, gt, args.recon_loss,
                                                 args.fg_thresh, args.fg_boost)

                # --- forward fidelity (params reused for reg + supervision) ---
                lensed_pred, lens_params, subhalo_params = forward_operator(
                    gt, obs, return_params=True)
                if args.multires_fwd:                         # [NEW T10]
                    forward_fidelity_loss = multires_fidelity(lensed_pred, obs)
                else:
                    forward_fidelity_loss = F.mse_loss(lensed_pred, obs)

                # --- PSF residual smoothness (unchanged) ---
                if forward_operator.raw_res_psf is not None:
                    res_psf = forward_operator.raw_res_psf
                    lap = res_psf - F.avg_pool2d(res_psf, 3, 1, padding=1)
                    psf_pen = (lap ** 2).mean()
                else:
                    psf_pen = torch.tensor(0.0, device=device)

                param_reg = forward_operator.compute_regularization_loss(
                    lens_params, subhalo_params)

                # --- supervised parameter loss [NEW T6] ---
                if use_param_sup:
                    sup_loss, per_param = supervised_param_loss(
                        lens_params, tvec, tmask, psf_fwhm_range=psf_range,
                        weights=sup_weights)
                    for k, v in per_param.items():
                        per_param_epoch[k] = per_param_epoch.get(k, 0.0) + v
                    n_pp += 1
                else:
                    sup_loss = torch.tensor(0.0, device=device)

                # --- TV prior [NEW T9] ---
                tv_term = total_variation(recon) if args.lambda_tv > 0 else \
                    torch.tensor(0.0, device=device)

                # --- soft non-negativity on the linear reconstruction [FIX T11] ---
                nonneg_pen = nonneg_penalty(recon) if args.lambda_nonneg > 0 else \
                    torch.tensor(0.0, device=device)

                # --- L1 source sparsity [FIX T14] ---
                # The reconstruction smears flux across the SIE tangential caustic
                # (the high-magnification region) instead of localizing it to the
                # compact true source. L1 on the source penalizes total absolute
                # flux, which preferentially removes the cheap diffuse caustic smear
                # (many pixels x small value) while the foreground-balanced MSE term
                # holds the genuine compact source up. It also pressures the
                # PSF/source degeneracy from the source side: with a compact source
                # forced, the only way to match the observed ring width is a correct
                # (smaller) PSF, rather than the current big-PSF + smeared-source mix.
                l1_term = recon.abs().mean() if args.lambda_l1 > 0 else \
                    torch.tensor(0.0, device=device)

                loss = (mse_recon +
                        lam_fwd * forward_fidelity_loss +
                        args.lambda_psf * psf_pen +
                        args.lambda_param_reg * param_reg +
                        args.lambda_param_sup * sup_loss +
                        args.lambda_tv * tv_term +
                        args.lambda_nonneg * nonneg_pen +
                        args.lambda_l1 * l1_term)

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at epoch {epoch_idx}, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if args.use_amp and device.type == 'cuda':
                scaler.unscale_(optimizer)

            trainable_params = [p for p in list(model.parameters()) +
                                list(forward_operator.parameters()) if p.requires_grad]
            grad_norm = compute_grad_norm(trainable_params)
            grad_finite = math.isfinite(grad_norm)
            amp_on = (args.use_amp and device.type == 'cuda')

            # [FIX T12] Handle non-finite gradients CLEANLY. The old loop clipped
            # first (clip_grad_norm_ with a nan total-norm multiplies EVERY grad
            # by nan), stepped, then tried to nan_to_num parameters afterward --
            # which cannot undo a corrupted AdamW moment buffer, so on a non-AMP
            # run a single bad batch bricked the optimizer for the rest of
            # training. Transient fp16 overflow early in training is expected;
            # we count and report it.
            if not grad_finite:
                n_skipped += 1
                loop.set_postfix({"skipped(non-finite grad)": n_skipped})
                if not amp_on:
                    # Disabled GradScaler does NOT skip the step, so we must, to
                    # avoid poisoning the optimizer state.
                    optimizer.zero_grad(set_to_none=True)
                    continue
                # AMP: fall through WITHOUT clipping (clipping would spread nan).
                # scaler.step() detects the inf/nan and skips the real update;
                # scaler.update() backs the loss scale off. step() must be called
                # after unscale_() to keep the scaler's state consistent.

            if grad_finite:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

            try:
                scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                print(f"[ERROR] Optimizer step failed: {e}. Skipping update.")
                optimizer.zero_grad(set_to_none=True)
                n_skipped += 1
                continue

            if ema is not None:
                ema.update()

            # Keep the learned PSF residual a bounded correction (a legitimate
            # constraint -- not a nan scrubber; non-finite batches are already
            # skipped above).
            if getattr(forward_operator, 'raw_res_psf', None) is not None:
                with torch.no_grad():
                    forward_operator.raw_res_psf.data.clamp_(-1.0, 1.0)

            bsz = obs.size(0)
            running['loss'] += loss.item() * bsz
            running['fwd'] += forward_fidelity_loss.item() * bsz
            running['reg'] += param_reg.item() * bsz
            running['sup'] += float(sup_loss.detach()) * bsz
            # [FIX T13] Skipped (non-finite) batches must not poison the reported
            # epoch grad-norm average -- otherwise a single fp16-overflow batch
            # makes the whole epoch read 'inf' even though the step was skipped.
            if math.isfinite(grad_norm):
                grad_norm_epoch += grad_norm

            loop.set_postfix({
                "loss": f"{loss.item():.4e}",
                "fwd": f"{forward_fidelity_loss.item():.4e}",
                "sup": f"{float(sup_loss):.4e}" if use_param_sup else "-",
                "g": f"{grad_norm:.2f}",
            })

        n_train = max(1, len(train_dataset))
        loss_history['train'].append(running['loss'] / n_train)
        loss_history['forward_fidelity'].append(running['fwd'] / n_train)
        loss_history['param_reg'].append(running['reg'] / n_train)
        loss_history['param_sup'].append(running['sup'] / n_train)
        loss_history['grad_norm'].append(grad_norm_epoch / max(1, len(train_loader)))
        loss_history['lambda_fwd'].append(lam_fwd)
        if n_pp > 0:
            per_param_running = {k: v / n_pp for k, v in per_param_epoch.items()}

        # ----------------------------- validation -----------------------------
        model.eval()
        forward_operator.eval()
        val_loss = val_ssim = val_fwd = 0.0
        val_src_mse = 0.0
        n_val = 0
        backup = ema.swap_in([model, forward_operator]) if ema is not None else None
        try:
            with torch.no_grad():
                for vbidx, batch in enumerate(val_loader):
                    if args.limit_val_batches and vbidx >= args.limit_val_batches:
                        break
                    obs, gt = batch[0].to(device), batch[1].to(device)
                    recon = apply_positivity(model(obs, obs, forward_operator),
                                             args.source_positivity)
                    l = F.mse_loss(recon, gt).item()
                    if not np.isfinite(l):
                        l = 1e6
                    val_loss += l * obs.size(0)

                    # [FIX T11] honest reconstruction signal: MSE only where the
                    # source actually is. All-pixel val MSE is dominated by the
                    # empty background and stays ~1e-3 even for a blank recon, so
                    # it must NOT be used as the 'is it reconstructing' signal.
                    fg = (gt > args.fg_thresh).float()
                    nfg = fg.sum().clamp_min(1.0)
                    smse = ((((recon - gt) ** 2) * fg).sum() / nfg).item()
                    if not np.isfinite(smse):
                        smse = 1e6
                    val_src_mse += smse * obs.size(0)

                    lensed_pred = forward_operator(gt, obs)
                    fwd_fid = F.mse_loss(lensed_pred, obs).item()
                    if not np.isfinite(fwd_fid):
                        fwd_fid = 1e6
                    val_fwd += fwd_fid * obs.size(0)

                    recon_np = recon.cpu().numpy()
                    gt_np = gt.cpu().numpy()
                    for i in range(recon_np.shape[0]):
                        try:
                            val_ssim += ssim(np.clip(gt_np[i, 0], 0, 1),
                                             np.clip(recon_np[i, 0], 0, 1),
                                             data_range=1.0)
                        except Exception:
                            pass
                    n_val += recon_np.shape[0]
        finally:
            if ema is not None and backup is not None:
                ema.restore([model, forward_operator], backup)

        avg_val_loss = val_loss / max(1, n_val)
        avg_val_ssim = val_ssim / max(1, n_val)
        avg_val_src = val_src_mse / max(1, n_val)
        loss_history['val'].append(avg_val_loss)
        loss_history['val_ssim'].append(avg_val_ssim)
        loss_history['val_src_mse'].append(avg_val_src)
        loss_history['skipped'].append(n_skipped)

        epoch_time = time.time() - start_epoch_time
        print(f"\n[Epoch {epoch_idx}]")
        print(f"  Train Loss: {loss_history['train'][-1]:.6e} | "
              f"Fwd Fid: {loss_history['forward_fidelity'][-1]:.6e} | "
              f"Param Reg: {loss_history['param_reg'][-1]:.6e}"
              + (f" | Param Sup: {loss_history['param_sup'][-1]:.6e}" if use_param_sup else ""))
        print(f"  Val Loss:   {avg_val_loss:.6e} (all-px; background-dominated) | "
              f"Val Fwd Fid: {val_fwd / max(1, n_val):.6e}")
        print(f"  >> Val SOURCE MSE: {avg_val_src:.6e}  <-- watch THIS "
              f"(reconstruction quality where the source is)")
        print(f"  Val SSIM:   {avg_val_ssim:.4f} | Grad Norm: {loss_history['grad_norm'][-1]:.3f}"
              f" | Skipped(non-finite grad): {n_skipped} | Time: {epoch_time:.1f}s")
        if use_param_sup and per_param_running:
            terms = "  ".join(f"{k}={v:.3e}" for k, v in sorted(per_param_running.items()))
            print(f"  Per-param sup: {terms}")

        try:
            scheduler.step(avg_val_loss)
        except Exception:
            pass

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.6e}")
            rim_best_path = os.path.join(args.output_dir, args.save_rim_best)
            forward_best_path = os.path.join(args.output_dir, args.save_forward_best)
            # [FIX T5] best checkpoints now carry meta (loaders accept both the
            # new dict layout and old bare state dicts).
            if ema is not None:
                backup = ema.swap_in([model, forward_operator])
            torch.save({'model_state': model.state_dict(),
                        CHECKPOINT_META_KEY: meta}, rim_best_path)
            torch.save({'forward_state': forward_operator.state_dict(),
                        CHECKPOINT_META_KEY: meta}, forward_best_path)
            if ema is not None:
                ema.restore([model, forward_operator], backup)
            print(f"  Saved RIM:     {rim_best_path}")
            print(f"  Saved Forward: {forward_best_path}")

        if epoch_idx % args.checkpoint_every == 0 or epoch_idx == args.num_epochs:
            ck = {
                'epoch': epoch_idx,
                'model_state': model.state_dict(),
                'forward_state': forward_operator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                CHECKPOINT_META_KEY: meta,
            }
            if ema is not None:
                ck['ema_state'] = ema.state_dict()
            ck_path = os.path.join(args.checkpoint_dir,
                                   f"cond_finetune_checkpoint_epoch_{epoch_idx}.pt")
            torch.save(ck, ck_path)
            print(f"  Checkpoint: {ck_path}")

            with torch.no_grad():
                try:
                    obs_sample = next(iter(val_loader))[0].to(device)
                    lp, _ = forward_operator.encode_lens_parameters(obs_sample)
                    psf = forward_operator._get_psf_batched(lp, device).cpu().numpy()
                    plt.figure(figsize=(4, 4))
                    plt.imshow(psf[0, 0], cmap='hot')
                    plt.colorbar()
                    plt.title(f'Example PSF Kernel - Epoch {epoch_idx}')
                    plt.axis('off')
                    plt.tight_layout()
                    psf_path = os.path.join(args.checkpoint_dir,
                                            f'cond_psf_epoch_{epoch_idx}.png')
                    plt.savefig(psf_path, dpi=300)
                    plt.close()
                    print(f"  PSF: {psf_path}")
                except Exception as e:
                    print(f"  PSF visualization failed: {e}")

        if epoch_idx % args.recon_every == 0 or epoch_idx == args.num_epochs:
            try:
                out_recon_dir = os.path.join(args.recon_out_dir, f'epoch_{epoch_idx}')
                save_reconstructions(epoch_idx, model, forward_operator, val_loader,
                                     device, args.source_positivity,
                                     num_examples=args.recon_num_examples,
                                     out_dir=out_recon_dir)
            except Exception as e:
                print(f"Failed to save reconstructions: {e}")

    # ------------------------------ final curves ------------------------------
    print("\nGenerating loss curves...")
    curves = [
        ('cond_mse_curve_finetune.png', [('train', 'Train MSE'), ('val', 'Val MSE (EMA)')],
         'MSE', 'Conditional Reconstruction Loss'),
        ('cond_val_source_mse_curve.png', [('val_src_mse', 'Val SOURCE MSE')],
         'Source-region MSE', 'Reconstruction Quality Where The Source Is [T11]'),
        ('cond_ssim_curve_finetune.png', [('val_ssim', 'Val SSIM (EMA)')],
         'SSIM', 'Structural Similarity'),
        ('cond_forward_fidelity_curve.png', [('forward_fidelity', 'Train Forward Fidelity')],
         'Forward Fidelity MSE', 'Per-Lens Forward Operator Physics Accuracy'),
        ('cond_param_reg_curve.png', [('param_reg', 'Parameter Regularization')],
         'Regularization Loss', 'Per-Lens Parameter Regularization'),
    ]
    if use_param_sup:
        curves.append(('cond_param_sup_curve.png',
                       [('param_sup', 'Supervised Parameter Loss')],
                       'Supervised Param Loss',
                       'Ground-Truth Parameter Supervision [NEW T6]'))
    for fname, series, ylabel, title in curves:
        path = os.path.join(args.output_dir, fname)
        plt.figure(figsize=(8, 5))
        for key, label in series:
            plt.plot(loss_history[key], label=label, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Curve: {path}")

    with open(os.path.join(args.output_dir, 'loss_history.json'), 'w') as fh:
        json.dump(loss_history, fh, indent=2)

    print("\n" + "=" * 80)
    print("CONDITIONAL TRAINING COMPLETE (v2.1)")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6e}")
    if loss_history['forward_fidelity']:
        print(f"Final forward fidelity: {loss_history['forward_fidelity'][-1]:.6e}")
    if use_param_sup and loss_history['param_sup']:
        print(f"Final param supervision: {loss_history['param_sup'][-1]:.6e}")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Conditional RIM + physics forward-operator finetuning (AstroRIM v2.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths (defaults unchanged from v2.0)
    p.add_argument('--dataset-dir', type=str,
                   default='/home/jwalsh/astropy/AstroRIM_V2/AstroRIM_V2.1_scripts/sims_training',
                   help='Directory of simulator FITS files')
    p.add_argument('--output-dir', type=str,
                   default='/home/jwalsh/astropy/AstroRIM_V2/AstroRIM_V2.1_scripts/Model_holder_update',
                   help='Directory for best models, curves, loss_history.json')
    p.add_argument('--checkpoint-dir', type=str,
                   default='/home/jwalsh/astropy/AstroRIM_V2/AstroRIM_V2.1_scripts/Model_holder_update',
                   help='Directory for periodic full checkpoints')
    p.add_argument('--recon-out-dir', type=str, default='/home/jwalsh/astropy/AstroRIM_V2/AstroRIM_V2.1_scripts/Model_holder_update',
                   help='Directory for per-epoch reconstruction previews')
    p.add_argument('--save-rim-best', type=str, default='cond_rim_2.1_best_update.pt')
    p.add_argument('--save-forward-best', type=str, default='cond_forward_2.1_best_update.pt')

    # Warm starts / resume
    p.add_argument('--rim-checkpoint', type=str, default=None,
                   help='Optional RIM warm-start checkpoint')
    p.add_argument('--forward-checkpoint', type=str, default=None,
                   help='Optional forward-operator warm-start checkpoint')
    p.add_argument('--resume-checkpoint', type=str, default=None,
                   help='Full checkpoint to resume epoch/optimizer state from')
    p.add_argument('--allow-partial-load', action='store_true', default=False,
                   help='[FIX T2] Explicit opt-in for loading a checkpoint with '
                        'missing/unexpected keys (the unloaded tensors stay at '
                        'initialization; the loader prints exactly which). '
                        'Without this flag a mismatched checkpoint is an error.')

    # Optimization (defaults unchanged from v2.0)
    p.add_argument('--num-epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--lr-rim', type=float, default=5e-5)
    p.add_argument('--lr-forward', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=1e-6)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-patience', type=int, default=6)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--finetune-mode', type=str, default='both',
                   choices=['both', 'rim', 'forward'])
    p.add_argument('--freeze-psf', action='store_true', default=False,
                   help='Freeze PSF-related parameters in the forward operator')

    # Loss weights (defaults unchanged from v2.0)
    p.add_argument('--lambda-forward-fidelity', type=float, default=0.5)
    p.add_argument('--lambda-psf', type=float, default=1e-3)
    p.add_argument('--lambda-param-reg', type=float, default=1e-3)

    # [NEW T6] auxiliary parameter supervision
    p.add_argument('--lambda-param-sup', type=float, default=0.09,
                   help='Weight of the supervised parameter loss against TR_* '
                        'truth headers (0 = off, exact v2.0 behavior; 0.05 is a '
                        'sensible starting value). Directly targets the '
                        'SIE<->NFW mass-split and PSF hallucinations.')

    # Model architecture 
    p.add_argument('--n-iter', type=int, default=10, help='RIM iterations')
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--kernel-size', type=int, default=21)
    p.add_argument('--n-subhalos', type=int, default=2)

    # [FIX T4] Boolean flags: were action='store_true' with default=True, so
    # they could never be DISABLED from the CLI. Now --x / --no-x.
    p.add_argument('--use-ema', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--use-nfw', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--use-subhalos', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--use-per-obs-psf', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--use-lens-light', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--use-fft-conv', action=argparse.BooleanOptionalAction, default=True,
                   help='[FIX T3] Real FFT convolution in the forward operator')
    p.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True,
                   help='Flips/rotations; truth labels are transformed '
                        'consistently (verified)')

    # [FIX T5] PSF FWHM bounds (legacy v2.0 checkpoints: 1.0 / 25.0)
    p.add_argument('--psf-fwhm-min', type=float, default=DEFAULT_PSF_FWHM_RANGE[0],
                   help='Encoder PSF FWHM lower bound in detector px. '
                        'Use 1.0 to reproduce legacy v2.0 checkpoints exactly.')
    p.add_argument('--psf-fwhm-max', type=float, default=DEFAULT_PSF_FWHM_RANGE[1])

    # [NEW T7] encoder ablation
    p.add_argument('--param-mode', type=str, default='encoder',
                   choices=['encoder', 'global'],
                   help="'global' = ONE shared learned parameter vector instead "
                        "of the per-observation encoder (same bounds, same RIM, "
                        "same schedule): the controlled encoder ablation.")

    # [NEW T8/T9/T10] training enhancements (all default OFF = v2.0 behavior)
    p.add_argument('--deep-supervision', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Supervise every RIM iteration (weight (t+1)/T) in '
                        'addition to the final output')
    p.add_argument('--source-positivity', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='[CHANGED in T11] softplus on the reconstruction. DEFAULT '
                        'IS NOW OFF (linear output). softplus saturates -- its '
                        'gradient is ~0 once the output is driven to large negative '
                        'values -- which is exactly the blank-recon collapse the '
                        'diagnostic found (raw ~ -34 everywhere). Linear output plus '
                        '--lambda-nonneg is the stable default. This flag is recorded '
                        'in checkpoint meta and recon_gen matches it, so set it the '
                        'same at inference. Pass --source-positivity only to A/B test '
                        'the old behaviour.')
    p.add_argument('--lambda-tv', type=float, default=1e-4,
                   help='Isotropic total-variation prior weight on the '
                        'reconstruction')
    p.add_argument('--multires-fwd', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Forward fidelity also at 2x and 4x pooled resolution')

    # [FIX T11] sparse-source reconstruction loss (the blank-recon fix)
    p.add_argument('--recon-loss', type=str, default='balanced',
                   choices=['balanced', 'mse'],
                   help="'balanced' averages reconstruction error over foreground "
                        "and background pixels separately so the ~1%% source pixels "
                        "are not drowned out by the empty background (the cause of "
                        "the blank-recon collapse). 'mse' = legacy plain MSE.")
    p.add_argument('--fg-thresh', type=float, default=0.05,
                   help='Source-pixel threshold on the [0,1]-normalized target; used '
                        'by the balanced loss AND the source-region validation metric.')
    p.add_argument('--fg-boost', type=float, default=3.0,
                   help='Extra weight on source pixels beyond exact fg/bg balance '
                        '(1.0 = equal weight; higher = stronger source emphasis).')
    p.add_argument('--lambda-nonneg', type=float, default=1e-2,
                   help='Soft non-negativity penalty relu(-recon).mean() on the '
                        'linear reconstruction. Replaces saturating softplus '
                        'positivity without its dead-gradient collapse mode. Set 0 '
                        'to disable.')
    p.add_argument('--lambda-l1', type=float, default=8e-3,
                   help='[FIX T14] L1 sparsity recon.abs().mean() on the source. '
                        'Discourages the diffuse caustic smear (flux spread across '
                        'the SIE tangential caustic) so the source localizes, and '
                        'pressures the PSF/source degeneracy. TUNE THIS: raise it '
                        '(0.1-0.3) if the astroid smear persists; lower it if the '
                        'genuine source gets suppressed. Set 0 to disable.')

    # Bookkeeping (defaults unchanged from v2.0)
    p.add_argument('--checkpoint-every', type=int, default=3)
    p.add_argument('--recon-every', type=int, default=1)
    p.add_argument('--recon-num-examples', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)

    # [FIX T11] fast smoke testing without waiting a full (multi-hour) epoch.
    # 0 = no limit (full epoch). e.g. --limit-train-batches 200 --limit-val-batches 50
    # gives a few-minute run on the real data to confirm the recon is no longer
    # blank and Val SOURCE MSE is dropping, before launching the full run.
    p.add_argument('--limit-train-batches', type=int, default=0)
    p.add_argument('--limit-val-batches', type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.psf_fwhm_min <= 0 or args.psf_fwhm_min >= args.psf_fwhm_max:
        raise SystemExit("--psf-fwhm-min must be > 0 and < --psf-fwhm-max")
    conditional_finetune(args)


if __name__ == '__main__':
    main()