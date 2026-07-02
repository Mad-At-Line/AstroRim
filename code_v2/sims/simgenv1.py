"""
simgenv1.py - AstroRIM v2 (patched).

Fixes vs original:
  * Moffat alpha formula corrected (was using Gaussian conversion factor; dormant bug
    in this file since PSF_TYPE is hardcoded to 'GAUSSIAN', fixed for consistency).
  * Lens-light ellipticity now aligned with mass ellipticity (was orthogonal).

Sample filename prefix: sim_v1_{i:05d}.fits

v2.1 changes (see CHANGELOG.md):
  [FIX S1] Moffat PSF radial profile corrected via simgen_truth.make_psf_kernel_fixed.
      The old kernel fed a sigma-NORMALIZED elliptical radius into a Moffat
      whose alpha was in PIXELS, so the realized FWHM was
      requested_FWHM * sigma_pix (e.g. 3 px requested -> 3.83 px realized);
      the earlier 'alpha formula' patch fixed the alpha line but not the
      radius fed into it. Realized FWHM now equals requested FWHM; the
      Gaussian branch is bit-identical to the old (correct) one.
  [NEW S3] Ground-truth TR_* header block in the MODEL parameter convention
      (verified conversions; see simgen_truth.py), enabling predicted-vs-true
      scatter batches and the auxiliary parameter-supervision loss.
Requires simgen_truth.py in the same directory.
"""

import os
import random
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

# USER KNOBS
import simgen_truth as sgt  # [FIX S1, NEW S3] shared verified helpers

OUTPUT_DIR = r"C:\Users\mythi\AstroRIM\AstroRIM_2.1\Sims"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 96          # detector pixels (final)
OVERSAMPLE = 4           # super-grid factor
PIXEL_SCALE = 0.04       # arcsec / detector pixel
DEFAULT_ZP = 25.94       # AB zeropoint (ADU/s) — set to your instrument

NUM_REALIZATIONS = 120
SEED = None              # set an int for reproducibility

# Lens visibility controls
FORCE_STRONG_LENSING = True   # place the main source near caustic so arcs are obvious
THETA_E_RANGE = (0.9, 1.5)    # Einstein radius (arcsec) to favor visible arcs
ELLIPTICITY_RANGE = (0.15, 0.45)  # SIE ellipticity for interesting morphologies
SHEAR_MAX = 0.08

# Photometry & noise (mild)
EXPTIME_RANGE = (800.0, 1600.0)     # seconds
READ_NOISE_RANGE = (0.8, 1.8)       # e- RMS (mild)
GAIN = 1.5                          # e-/ADU
SKY_ADU_RANGE = (0.06, 0.25)        # flat sky level per pixel in ADU (mild)

# Source/lens brightness — tuned so arcs are not swamped by lens light
SRC_MAG_RANGE = (21.0, 23.2)        # brighter -> lower mag number
LENS_MAG_DELTA_RANGE = (-0.3, +1.2) # lens_mag = src_mag + delta

# PSF
PSF_TYPE = 'GAUSSIAN'               # 'GAUSSIAN' or 'MOFFAT'
PSF_FWHM_ARCSEC = (0.06, 0.10)      # space-like, keeps arcs sharp

# Extra source complexity
N_EXTRA_SOURCES_RANGE = (0, 2)      # additional faint sources beyond the main one
USE_CUTOUTS = False                 # if True, set CUTOUT_DIR to a folder of image cutouts
CUTOUT_DIR = None
# Downsampling and super-sampling
SUPER_SIZE = IMAGE_SIZE * OVERSAMPLE
SUP_PIXEL_SCALE = PIXEL_SCALE / OVERSAMPLE


def set_seed(seed=None):
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rnd = int(seed) % (2**32 - 1)
    random.seed(rnd)
    np.random.seed(rnd)


def sigma_to_thetaE_arcsec(sigma_kms, z_lens, z_source):
    c = 299792.458
    D_s = cosmo.angular_diameter_distance(z_source).to(u.km).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.km).value
    theta_rad = 4.0 * np.pi * (sigma_kms**2) / (c**2) * (D_ls / D_s)
    return theta_rad * (180.0 / np.pi) * 3600.0


def make_psf_kernel(fwhm_arcsec, pixel_scale_arcsec, psf_type='GAUSSIAN'):
    """[FIX S1] Delegates to simgen_truth.make_psf_kernel_fixed (corrected
    Moffat radial profile: realized FWHM == requested FWHM; Gaussian branch
    bit-identical to the previous correct one). Returns (kernel, info) --
    info carries realized_fwhm_pix in the pixels of pixel_scale_arcsec.
    No coma perturbation in this variant (matches previous behavior)."""
    return sgt.make_psf_kernel_fixed(fwhm_arcsec, pixel_scale_arcsec,
                                     psf_type=psf_type, ellip=0.0, angle=0.0, beta=3.5,
                                     rng=np.random, coma_prob=0.0)


def downsample(img_sup, factor):
    s = img_sup.shape[0]
    new = s // factor
    return img_sup.reshape(new, factor, new, factor).mean(axis=(1, 3))


def rg_interpolator(x_sup, y_sup, im_sup, method='linear'):
    # RegularGridInterpolator expects axes as (y, x)
    return RegularGridInterpolator((y_sup[:, 0], x_sup[0, :]), im_sup,
                                   method=method, bounds_error=False, fill_value=0.0)


def interp_source_on_deflected(x_sup, y_sup, src_sup, x_src, y_src):
    interp_lin = rg_interpolator(x_sup, y_sup, src_sup, 'linear')
    pts = np.vstack([y_src.ravel(), x_src.ravel()]).T
    out = interp_lin(pts)
    # nearest fallback to fill zeros near edges
    if np.any(out == 0):
        interp_nn = rg_interpolator(x_sup, y_sup, src_sup, 'nearest')
        mask = (out == 0)
        out[mask] = interp_nn(pts[mask])
    return out.reshape(x_src.shape).astype(np.float32)


def normalize_to_counts(img, total_mag, zp, exptime):
    img = np.clip(img, 0.0, None).astype(np.float64)
    s = img.sum()
    if s <= 0:
        return np.zeros_like(img, dtype=np.float32), 0.0
    img_norm = img / s
    total_counts = exptime * 10**(-0.4 * (total_mag - zp))
    return (img_norm * total_counts).astype(np.float32), float(total_counts)


def add_clumps(base, n=3, max_rel=0.8, sig_pix=(1.0, 5.0)):
    sup = base.copy()
    S = sup.shape[0]
    for _ in range(n):
        cx = np.random.uniform(0.25*S, 0.75*S)
        cy = np.random.uniform(0.25*S, 0.75*S)
        amp = np.random.uniform(0.05, max_rel) * (sup.max() if sup.max() > 0 else 1.0)
        sigma = np.random.uniform(*sig_pix)
        X, Y = np.meshgrid(np.arange(S), np.arange(S))
        sup += amp * np.exp(-0.5 * (((X - cx)**2 + (Y - cy)**2) / (sigma**2)))
    return np.clip(sup, 0.0, None)


def build_sources(x_sup, y_sup, theta_E):
    # Main bright source near lens center to encourage arcs
    lm = LightModel(['SERSIC_ELLIPSE'])
    # position within ~0.2 * theta_E of center
    cx = np.random.uniform(-0.2 * theta_E, 0.2 * theta_E)
    cy = np.random.uniform(-0.2 * theta_E, 0.2 * theta_E)
    n_sersic = np.random.uniform(1.0, 3.5)
    e = np.random.uniform(0.0, 0.5)
    phi = np.random.uniform(0, np.pi)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    kwargs_main = {
        'amp': np.random.uniform(2.0, 6.0),
        'R_sersic': np.random.uniform(0.05, 0.25),
        'n_sersic': n_sersic,
        'e1': e1, 'e2': e2,
        'center_x': cx, 'center_y': cy
    }
    main = lm.surface_brightness(x_sup, y_sup, [kwargs_main]).astype(np.float32)

    # Optional extra faint sources
    n_extra = np.random.randint(N_EXTRA_SOURCES_RANGE[0], N_EXTRA_SOURCES_RANGE[1] + 1)
    if n_extra > 0:
        extras = []
        for _ in range(n_extra):
            lm2 = LightModel(['SERSIC_ELLIPSE'])
            cx2 = np.random.uniform(-0.9 * theta_E, 0.9 * theta_E)
            cy2 = np.random.uniform(-0.9 * theta_E, 0.9 * theta_E)
            e2m = np.random.uniform(0.0, 0.5)
            ph = np.random.uniform(0, np.pi)
            e1e = e2m * np.cos(2 * ph)
            e2e = e2m * np.sin(2 * ph)
            extras.append(lm2.surface_brightness(x_sup, y_sup, [{
                'amp': np.random.uniform(1.0, 3.0),
                'R_sersic': np.random.uniform(0.04, 0.18),
                'n_sersic': np.random.uniform(0.8, 4.0),
                'e1': e1e, 'e2': e2e,
                'center_x': cx2, 'center_y': cy2
            }]).astype(np.float32))
        if len(extras):
            main = main + sum(extras)

    # add clumpy features
    if np.random.rand() < 0.7:
        main = add_clumps(main, n=np.random.randint(1, 5), max_rel=np.random.uniform(0.3, 1.0),
                          sig_pix=(OVERSAMPLE * 0.6, OVERSAMPLE * 4.0))
    return np.clip(main, 0.0, None).astype(np.float32)


def generate_one(i, outdir=OUTPUT_DIR):
    # Physical setup
    z_l = float(np.random.uniform(0.25, 0.7))
    z_s = float(np.random.uniform(max(z_l + 0.05, 0.9), 3.0))

    theta_E = float(np.random.uniform(*THETA_E_RANGE))
    e = float(np.random.uniform(*ELLIPTICITY_RANGE))
    phi = np.random.uniform(0, np.pi)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    shear_g = np.random.uniform(0.0, SHEAR_MAX)
    shear_pa = np.random.uniform(0, np.pi)
    g1 = shear_g * np.cos(2 * shear_pa)
    g2 = shear_g * np.sin(2 * shear_pa)

    lens_model_list = ['SIE']
    kwargs_lens = [{
        'theta_E': theta_E,
        'e1': e1, 'e2': e2,
        'center_x': np.random.uniform(-0.04, 0.04),
        'center_y': np.random.uniform(-0.04, 0.04)
    }]
    if shear_g > 0:
        lens_model_list.append('SHEAR')
        kwargs_lens.append({'gamma1': g1, 'gamma2': g2})
    lens = LensModel(lens_model_list)

    # Grids
    grid_lin_sup = np.linspace(-0.5 * IMAGE_SIZE * PIXEL_SCALE,
                               0.5 * IMAGE_SIZE * PIXEL_SCALE, SUPER_SIZE)
    x_sup, y_sup = np.meshgrid(grid_lin_sup, grid_lin_sup)

    # Sources (intrinsic)
    src_sup = build_sources(x_sup, y_sup, theta_E)

    # Photometry and PSF/noise
    exptime = float(np.random.uniform(*EXPTIME_RANGE))
    zp = DEFAULT_ZP
    psf_fwhm = float(np.random.uniform(*PSF_FWHM_ARCSEC))
    psf_sup, psf_kernel_info = make_psf_kernel(psf_fwhm, SUP_PIXEL_SCALE, PSF_TYPE)
    read_noise = float(np.random.uniform(*READ_NOISE_RANGE))
    sky_adu = float(np.random.uniform(*SKY_ADU_RANGE))

    # Choose magnitudes; lens vs source contrast
    src_mag = float(np.random.uniform(*SRC_MAG_RANGE))
    lens_mag = src_mag + float(np.random.uniform(*LENS_MAG_DELTA_RANGE))

    # Scale intrinsic source to counts (ADU) on super-grid
    src_counts_sup, src_total = normalize_to_counts(src_sup, src_mag, zp, exptime)

    # GT (background only, unlensed, detector-sampled, not PSF-convolved) 
    GT = downsample(src_counts_sup, OVERSAMPLE).astype(np.float32)

    # LENSED (lensed + lens-light + PSF + mild noise)
    # Ray-shooting to source plane then interpolate counts
    x_flat, y_flat = x_sup.ravel(), y_sup.ravel()
    xs, ys = lens.ray_shooting(x_flat, y_flat, kwargs_lens)
    xs = xs.reshape(SUPER_SIZE, SUPER_SIZE)
    ys = ys.reshape(SUPER_SIZE, SUPER_SIZE)
    lensed_src_counts_sup = interp_source_on_deflected(x_sup, y_sup, src_counts_sup, xs, ys)

    # Lens galaxy light (on image plane, *not* lensed)
    lens_light_model = LightModel(['SERSIC_ELLIPSE'])
    # [NEW S3] lens-light parameters drawn into named variables so the
    # truth header can record them (same distributions as before).
    ll_Re_arcsec = np.random.uniform(0.25, 1.0)
    ll_n_sersic = np.random.uniform(3.0, 5.0)
    ll_center_x = kwargs_lens[0]['center_x'] + np.random.uniform(-0.02, 0.02)
    ll_center_y = kwargs_lens[0]['center_y'] + np.random.uniform(-0.02, 0.02)
    lens_light_sup = lens_light_model.surface_brightness(x_sup, y_sup, [{
        'amp': 1.0,
        'R_sersic': ll_Re_arcsec,
        'n_sersic': ll_n_sersic,
        # FIXED: lens light PA must be ALIGNED with mass PA, not orthogonal.
        # Previous convention (-0.6 * e_mass) flipped (e1, e2) -> rotation of position
        # angle by 90 degrees, which is physically wrong. In real ellipticals the mass
        # and light PAs agree within ~10 degrees. Keep positive sign; slight random
        # scaling so light and mass axis ratios differ naturally.
        'e1': e1 * np.random.uniform(0.6, 1.0) + np.random.uniform(-0.05, 0.05),
        'e2': e2 * np.random.uniform(0.6, 1.0) + np.random.uniform(-0.05, 0.05),
        'center_x': ll_center_x,
        'center_y': ll_center_y
    }]).astype(np.float32)

    lens_counts_sup, lens_total = normalize_to_counts(lens_light_sup, lens_mag, zp, exptime)

    # Convolve (lensed sources + lens light) with PSF at super-res then downsample
    image_sup = lensed_src_counts_sup + lens_counts_sup
    image_conv_sup = convolve_fft(image_sup, psf_sup, normalize_kernel=True, allow_huge=True)
    LENSED_counts = downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)

    # Ensure arcs aren't swamped: adjust brightness ratio if needed (fast linear rescale)
    peak_src = float(np.percentile(downsample(convolve_fft(lensed_src_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE), 99))
    peak_lens = float(np.percentile(downsample(convolve_fft(lens_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE), 99))
    ratio = (peak_src + 1e-6) / (peak_lens + 1e-6)
    target_ratio = 0.35  # want arcs to be at least ~35% of lens peak
    adjust_tries = 0
    while ratio < target_ratio and adjust_tries < 3:
        # dim lens slightly and/or brighten source
        LENSED_counts *= 0  # keep Poisson stats consistent
        lens_counts_sup *= 0.8
        src_counts_sup *= 1.25
        # recompute lensed/source conv quickly
        lensed_src_counts_sup = interp_source_on_deflected(x_sup, y_sup, src_counts_sup, xs, ys)
        image_sup = lensed_src_counts_sup + lens_counts_sup
        image_conv_sup = convolve_fft(image_sup, psf_sup, normalize_kernel=True, allow_huge=True)
        LENSED_counts = downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)
        peak_src = float(np.percentile(downsample(convolve_fft(lensed_src_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE), 99))
        peak_lens = float(np.percentile(downsample(convolve_fft(lens_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE), 99))
        ratio = (peak_src + 1e-6) / (peak_lens + 1e-6)
        adjust_tries += 1

    # Add mild sky + noise
    LENSED_counts = LENSED_counts + sky_adu
    # convert ADU -> e-, add Poisson + read noise, back to ADU
    e_image = np.clip(LENSED_counts * GAIN, 0.0, None)
    noisy_e = np.random.poisson(e_image).astype(np.float32)
    noisy_e += np.random.normal(0.0, read_noise, noisy_e.shape).astype(np.float32)
    LENSED = (noisy_e / GAIN).astype(np.float32)

    # Save exactly TWO HDUs
    filename = os.path.join(outdir, f"sim_v1_{i:05d}.fits")
    hdu_primary = fits.PrimaryHDU()
    
    # Create image HDUs with proper headers
    hdu_gt = fits.ImageHDU(data=GT.astype(np.float32), name='GT')
    hdu_lensed = fits.ImageHDU(data=LENSED.astype(np.float32), name='LENSED')
    
    # Add essential keywords to image headers
    for hdu in [hdu_gt, hdu_lensed]:
        hdu.header['PIXSCALE'] = (PIXEL_SCALE, 'arcsec/pixel')
        hdu.header['BUNIT'] = ('ADU', 'Brightness unit')
        hdu.header['OVERSAMP'] = (OVERSAMPLE, 'super-sampling factor')
        hdu.header['ZP'] = (DEFAULT_ZP, 'AB zeropoint (ADU/s)')
        hdu.header['EXPTIME'] = (exptime, 'seconds')

    # Primary header metadata
    hdr = hdu_primary.header
    hdr['DATE'] = (datetime.utcnow().isoformat(), 'UTC creation time')
    hdr['GAIN'] = (GAIN, 'e-/ADU')
    hdr['RN_E'] = (read_noise, 'read noise e- RMS')
    hdr['SKYADU'] = (sky_adu, 'sky level (ADU/pixel)')
    hdr['PSF_FWH'] = (psf_fwhm, 'PSF FWHM (arcsec)')
    hdr['PSF_TYP'] = (PSF_TYPE, 'PSF model type')
    hdr['THETA_E'] = (theta_E, 'Einstein radius (arcsec)')
    hdr['ELLIP'] = (e, 'SIE ellipticity')
    hdr['SHEAR'] = (shear_g, 'external shear amplitude')
    hdr['SRCMAG'] = (src_mag, 'source integrated AB mag')
    hdr['LENSMAG'] = (lens_mag, 'lens integrated AB mag')
    hdr['ZLENS'] = (z_l, 'lens redshift')
    hdr['ZSRC'] = (z_s, 'source redshift')


    # ------------------------------------------------------------------
    # [NEW S3] Ground-truth TR_* block in the MODEL's parameter convention
    # (verified conversions; see simgen_truth.py / CHANGELOG.md). Enables
    # predicted-vs-true scatter batches and auxiliary supervision.
    # ------------------------------------------------------------------
    realized_fwhm_arcsec = psf_kernel_info['realized_fwhm_pix'] * SUP_PIXEL_SCALE
    lens_light_det = downsample(lens_counts_sup, OVERSAMPLE).astype(np.float32)
    norm_m = float(max(GT.max(), LENSED.max(), 1e-8))
    ll_amp = sgt.measure_lens_light_amp_at_Re(
        lens_light_det, (ll_center_x, ll_center_y), ll_Re_arcsec,
        PIXEL_SCALE, norm_m)
    truth, maskbits, clean = sgt.truth_from_lenstronomy(
        lens_model_list, kwargs_lens,
        pixel_scale=PIXEL_SCALE, image_size=IMAGE_SIZE,
        psf_info={
            'realized_fwhm_pix': realized_fwhm_arcsec / PIXEL_SCALE,
            'psf_type': PSF_TYPE, 'beta': 3.5,
            'ellip': 0.0, 'angle': 0.0,
            'coma_applied': False,
        },
        lens_light_info={
            'Re_arcsec': ll_Re_arcsec, 'n_sersic': ll_n_sersic,
            'amp_at_Re_normalized': ll_amp,
        })
    sgt.write_truth_header(hdr, truth, maskbits, clean, sim_set='simgenv1',
                           pixel_scale=PIXEL_SCALE, coma_applied=False)

    hdul = fits.HDUList([hdu_primary, hdu_gt, hdu_lensed])
    hdul.writeto(filename, overwrite=True)

    print(f"[{i}] Saved {filename}  | peak(src/lens) ratio≈{ratio:.2f}, noise mild, TWO HDUs only")
    return filename


def main():
    set_seed(SEED)
    for i in range(NUM_REALIZATIONS):
        generate_one(i, OUTPUT_DIR)


if __name__ == '__main__':
    main()
