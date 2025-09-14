#!/usr/bin/env python3
"""
Modified simulator to *primarily* produce hyper-realistic WEAK lenses (i.e. no multiple imaging,
small Einstein radii / low magnification, modest shear) with diverse, realistic observational
effects. The code keeps the original pipeline but:
 - shifts physical ranges downward (lower sigma, smaller theta_E)
 - rejects parameter draws that produce multiple imaging (uses lenstronomy's LensEquationSolver)
 - adds more realistic PSF options (Moffat + Gaussian) and correlated noise
 - adds a faint field galaxy population and occasional cosmetic defects
 - reduces subhalo/group-halo rates (they produce strong small-scale effects)

Usage: run as before. Output FITS files with HDUs: Primary + GT + LENSED.

Notes:
 - This was written to be drop-in with the original dependencies (lenstronomy, astropy, scipy, numpy).
 - The "weak lens" selection uses a point-source image-count test: if solver finds >1 image we
   treat that as "strong" and resample. This is a pragmatic test for multiple imaging.

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
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
# solver for image-position tests (to reject strong lenses)
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

OUTPUT_DIR = r"C:\Users\mythi\.astropy\Code\finetuning_5scripts_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 96          # final detector pixels (same)
OVERSAMPLE = 4           # super-sampling factor
PIXEL_SCALE = 0.04       # arcsec / pixel
DEFAULT_ZP = 25.94

NUM_REALIZATIONS = 55000
SEED = None

# --- Tuned for WEAK LENSING (mostly single-imaged, low magnification) ---
SIGMA_KMS_RANGE = (80.0, 180.0)    # lower velocity dispersions -> smaller Einstein radii
ELLIPTICITY_RANGE = (0.02, 0.45)
SHEAR_MAX = 0.06                    # small external shear (weak regime)

# Keep photometric/noise ranges but can be made slightly deeper for realistic S/N
EXPTIME_RANGE = (900.0, 1800.0)
READ_NOISE_RANGE = (0.8, 1.8)
GAIN = 1.5
SKY_ADU_RANGE = (0.04, 0.22)

# Source / lens brightness - we want sources moderately faint so lensing signatures are subtle
SRC_MAG_RANGE = (21.0, 23.5)
LENS_MAG_DELTA_RANGE = (-0.2, +1.5)

# PSF
PSF_TYPES = ('GAUSSIAN', 'MOFFAT')
PSF_FWHM_ARCSEC = (0.06, 0.12)
PSF_ELLIP_MAX = 0.08
MOFFAT_BETA_RANGE = (2.8, 4.5)

# Extra source complexity
N_EXTRA_SOURCES_RANGE = (0, 2)
USE_CUTOUTS = False
CUTOUT_DIR = None

# Detector artifacts (small)
PRNU_RMS = 0.01        # 1% PRNU
SKY_GRADIENT_MAX = 0.02
COSMIC_RAY_PROB = 0.02
COSMIC_RAY_INTENSITY = (50.0, 300.0)

# Lens model options reduced for weak-lens emphasis
ADD_GROUP_HALO_PROB = 0.08
ADD_SUBHALOS_PROB = 0.25

# Derived
SUPER_SIZE = IMAGE_SIZE * OVERSAMPLE
SUP_PIXEL_SCALE = PIXEL_SCALE / OVERSAMPLE  # arcsec / super-pixel
SUPER_PIX_AREA = SUP_PIXEL_SCALE ** 2

# How many attempts to resample if we accidentally draw a strong lens
MAX_PARAMETER_ATTEMPTS = 8


def set_seed(seed=None):
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rnd = int(seed) % (2**32 - 1)
    random.seed(rnd)
    np.random.seed(rnd)


def sigma_to_thetaE_arcsec(sigma_kms, z_lens, z_source):
    # standard SIS approximation -> Einstein radius in arcsec
    c = 299792.458  # km/s
    D_s = cosmo.angular_diameter_distance(z_source).to(u.km).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.km).value
    theta_rad = 4.0 * np.pi * (sigma_kms**2) / (c**2) * (D_ls / D_s)
    return theta_rad * (180.0 / np.pi) * 3600.0


def make_psf_kernel(fwhm_arcsec, pixel_scale_arcsec, psf_type='GAUSSIAN', ellip=0.0, angle=0.0, beta=3.5):
    """Create a PSF kernel (super-sampled). Supports GAUSSIAN and MOFFAT.
    Ellipticity is parameterised as ellip = 1 - q where q is axis ratio.
    Angle is in radians.
    """
    fwhm_pix = fwhm_arcsec / pixel_scale_arcsec
    sigma_pix = fwhm_pix / 2.3548
    q = max(0.001, 1.0 - ellip)
    sigma_x = sigma_pix / np.sqrt(q)
    sigma_y = sigma_pix * np.sqrt(q)
    k = int(max(15, np.ceil(max(sigma_x, sigma_y) * 8)))
    if k % 2 == 0:
        k += 1
    y, x = np.mgrid[:k, :k] - k // 2
    x_rot =  np.cos(angle)*x + np.sin(angle)*y
    y_rot = -np.sin(angle)*x + np.cos(angle)*y
    r2 = (x_rot**2)/(sigma_x**2 + 1e-12) + (y_rot**2)/(sigma_y**2 + 1e-12)
    if psf_type.upper() == 'GAUSSIAN':
        kern = np.exp(-0.5 * r2)
    else:
        # Moffat profile
        alpha = sigma_pix / np.sqrt(2**(1.0 / beta) - 1.0)
        kern = (1 + (r2 / (alpha**2)))**(-beta)
    kern /= kern.sum()
    return kern.astype(np.float32)


def downsample(img_sup, factor):
    s = img_sup.shape[0]
    new = s // factor
    return img_sup.reshape(new, factor, new, factor).mean(axis=(1, 3))


def rg_interpolator(x_sup, y_sup, im_sup, method='linear'):
    return RegularGridInterpolator((y_sup[:, 0], x_sup[0, :]), im_sup,
                                   method=method, bounds_error=False, fill_value=0.0)


def normalize_to_sb_per_arcsec2(img, total_mag, zp, exptime, pixel_scale_arcsec):
    img = np.clip(img, 0.0, None).astype(np.float64)
    s = img.sum()
    if s <= 0:
        return np.zeros_like(img, dtype=np.float32), 0.0
    img_norm = img / s
    total_counts = exptime * 10**(-0.4 * (total_mag - zp))
    counts_per_superpix = img_norm * total_counts
    pix_area = (pixel_scale_arcsec)**2
    sb_counts_per_arcsec2 = counts_per_superpix / pix_area
    return sb_counts_per_arcsec2.astype(np.float32), float(total_counts)


def add_clumps(base, n=3, max_rel=0.8, sig_pix=(1.0, 5.0)):
    sup = base.copy()
    S = sup.shape[0]
    for _ in range(n):
        cx = np.random.uniform(0.25*S, 0.75*S)
        cy = np.random.uniform(0.25*S, 0.75*S)
        amp = np.random.uniform(0.02, max_rel) * (sup.max() if sup.max() > 0 else 1.0)
        sigma = np.random.uniform(*sig_pix)
        X, Y = np.meshgrid(np.arange(S), np.arange(S))
        sup += amp * np.exp(-0.5 * (((X - cx)**2 + (Y - cy)**2) / (sigma**2)))
    return np.clip(sup, 0.0, None)


def build_sources(x_sup, y_sup, theta_E):
    """Build an extended source with plausible bulge+disk structure, clumps, and optional
    spiral perturbations. Coordinates are in arcsec (super-grid).
    """
    lm = LightModel(['SERSIC_ELLIPSE'])
    cx = np.random.uniform(-0.12 * theta_E, 0.12 * theta_E)
    cy = np.random.uniform(-0.12 * theta_E, 0.12 * theta_E)

    frac_bulge = np.random.beta(1.5, 3.0)
    Rb = np.random.uniform(0.015, 0.09)  # smaller effective sizes (arcsec)
    Rd = Rb * np.random.uniform(1.6, 3.5)
    n_bulge = np.random.uniform(2.0, 4.5)
    amp_b = np.random.uniform(0.7, 3.0)
    amp_d = amp_b * np.random.uniform(0.25, 1.1)

    phi = np.random.uniform(0, np.pi)
    e = np.random.uniform(0.0, 0.6)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)

    kwargs_b = {'amp': amp_b, 'R_sersic': Rb, 'n_sersic': n_bulge,
                'e1': e1, 'e2': e2, 'center_x': cx, 'center_y': cy}
    kwargs_d = {'amp': amp_d, 'R_sersic': Rd, 'n_sersic': 1.0,
                'e1': e1 * np.random.uniform(0.8, 1.0), 'e2': e2 * np.random.uniform(0.8, 1.0),
                'center_x': cx + np.random.uniform(-0.02, 0.02), 'center_y': cy + np.random.uniform(-0.02, 0.02)}

    bulge = lm.surface_brightness(x_sup, y_sup, [kwargs_b]).astype(np.float32)
    disk  = lm.surface_brightness(x_sup, y_sup, [kwargs_d]).astype(np.float32)
    main = frac_bulge * bulge + (1 - frac_bulge) * disk

    n_extra = np.random.randint(N_EXTRA_SOURCES_RANGE[0], N_EXTRA_SOURCES_RANGE[1] + 1)
    if n_extra > 0:
        for _ in range(n_extra):
            cx2 = np.random.uniform(-0.9 * theta_E, 0.9 * theta_E)
            cy2 = np.random.uniform(-0.9 * theta_E, 0.9 * theta_E)
            kwargs_e = {
                'amp': np.random.uniform(0.2, 1.5),
                'R_sersic': np.random.uniform(0.01, 0.08),
                'n_sersic': np.random.uniform(0.8, 3.5),
                'e1': np.random.uniform(-0.4, 0.4),
                'e2': np.random.uniform(-0.4, 0.4),
                'center_x': cx2, 'center_y': cy2
            }
            main += lm.surface_brightness(x_sup, y_sup, [kwargs_e]).astype(np.float32)

    if np.random.rand() < 0.6:
        main = add_clumps(main, n=np.random.randint(1, 6), max_rel=np.random.uniform(0.15, 0.7),
                          sig_pix=(OVERSAMPLE * 0.4, OVERSAMPLE * 3.5))

    if np.random.rand() < 0.25:
        X, Y = np.meshgrid(np.linspace(-1, 1, main.shape[0]), np.linspace(-1, 1, main.shape[1]))
        spiral = 1.0 + 0.06 * np.sin(3.0 * np.arctan2(Y, X) + np.random.uniform(0, 2*np.pi))
        main *= spiral

    return np.clip(main, 0.0, None).astype(np.float32)


def add_field_galaxies(image_sup, n_fake=6):
    """Add faint, randomly placed background/foreground galaxies to the super-sampled image
    to increase realism (they are unlensed in our simulation)."""
    lm = LightModel(['SERSIC_ELLIPSE'])
    S = image_sup.shape[0]
    grid_lin = np.linspace(-0.5 * IMAGE_SIZE * PIXEL_SCALE,
                           0.5 * IMAGE_SIZE * PIXEL_SCALE, S)
    x_sup, y_sup = np.meshgrid(grid_lin, grid_lin)
    for _ in range(n_fake):
        amp = np.random.uniform(0.02, 0.35)
        R = np.random.uniform(0.008, 0.08)
        n = np.random.uniform(0.8, 3.5)
        cx = np.random.uniform(-0.5 * IMAGE_SIZE * PIXEL_SCALE, 0.5 * IMAGE_SIZE * PIXEL_SCALE)
        cy = np.random.uniform(-0.5 * IMAGE_SIZE * PIXEL_SCALE, 0.5 * IMAGE_SIZE * PIXEL_SCALE)
        e1 = np.random.uniform(-0.4, 0.4)
        e2 = np.random.uniform(-0.4, 0.4)
        gal = lm.surface_brightness(x_sup, y_sup, [{
            'amp': amp, 'R_sersic': R, 'n_sersic': n,
            'e1': e1, 'e2': e2, 'center_x': cx, 'center_y': cy
        }])
        image_sup += gal.astype(np.float32)
    return image_sup


def is_strong_for_source(lens_solver, src_x, src_y, kwargs_lens):
    """Return True if the solver finds multiple images for a point source at (src_x, src_y).
    For our weak-lens selection we *reject* parameter draws that generate >1 image.
    """
    try:
        imgs = lens_solver.image_position_from_source(src_x, src_y, kwargs_lens)
        # imgs is a list of (x,y) pairs -> multiple images means strong lensing
        if imgs is None:
            return False
        n_img = len(imgs)
        return n_img > 1
    except Exception:
        # In case the solver fails conservatively treat as strong to be safe
        return True


def generate_one(i, outdir=OUTPUT_DIR):
    # We'll attempt to draw lens params until we get a weak-lens configuration (or hit attempts)
    for attempt in range(MAX_PARAMETER_ATTEMPTS):
        # redshifts
        z_l = float(np.random.uniform(0.25, 0.7))
        z_s = float(np.random.uniform(max(z_l + 0.05, 0.9), 3.0))

        # physical lens strength via sigma
        sigma_kms = float(np.random.uniform(*SIGMA_KMS_RANGE))
        theta_E = float(sigma_to_thetaE_arcsec(sigma_kms, z_l, z_s))

        # ellipticity & shear
        e = float(np.random.uniform(*ELLIPTICITY_RANGE))
        phi = np.random.uniform(0, np.pi)
        e1 = e * np.cos(2 * phi)
        e2 = e * np.sin(2 * phi)
        shear_g = np.random.uniform(0.0, SHEAR_MAX)
        shear_pa = np.random.uniform(0, np.pi)
        g1 = shear_g * np.cos(2 * shear_pa)
        g2 = shear_g * np.sin(2 * shear_pa)

        # lens model list (start simple - SIE + optional shear)
        lens_model_list = ['SIE']
        center_x = np.random.uniform(-0.04, 0.04)
        center_y = np.random.uniform(-0.04, 0.04)
        kwargs_lens = [{
            'theta_E': theta_E,
            'e1': e1, 'e2': e2,
            'center_x': center_x,
            'center_y': center_y
        }]
        if shear_g > 0.0:
            lens_model_list.append('SHEAR')
            kwargs_lens.append({'gamma1': g1, 'gamma2': g2})

        # optionally add mild NFW group halo (rare now)
        if np.random.rand() < ADD_GROUP_HALO_PROB:
            lens_model_list.append('NFW')
            kwargs_lens.append({
                'alpha_Rs': 0.15,
                'Rs': np.random.uniform(10.0, 30.0),
                'center_x': center_x + np.random.uniform(-0.4, 0.4),
                'center_y': center_y + np.random.uniform(-0.4, 0.4)
            })

        # subhalos (small chance, small theta_E)
        if np.random.rand() < ADD_SUBHALOS_PROB:
            n_sub = np.random.poisson(0.7)
            for _ in range(max(0, n_sub)):
                lens_model_list.append('SIS')
                kwargs_lens.append({
                    'theta_E': np.random.uniform(0.004, 0.025),
                    'center_x': np.random.uniform(-0.6 * theta_E, 0.6 * theta_E),
                    'center_y': np.random.uniform(-0.6 * theta_E, 0.6 * theta_E)
                })

        lens = LensModel(lens_model_list)

        # Quick weak-lens safety check: take the main source centre (approx) and see how many images
        # build a trial source position near center (arcsec)
        trial_src_x = np.random.uniform(-0.08 * theta_E, 0.08 * theta_E)
        trial_src_y = np.random.uniform(-0.08 * theta_E, 0.08 * theta_E)
        try:
            lens_solver = LensEquationSolver(lens)
            if is_strong_for_source(lens_solver, trial_src_x, trial_src_y, kwargs_lens):
                # probably strong lens -> try new parameters
                if attempt < MAX_PARAMETER_ATTEMPTS - 1:
                    continue
                # else fall through and accept (rare accept to avoid infinite loop)
        except Exception:
            # if solver import/usage fails for any reason, accept the draw and continue
            pass

        # If we reach here either lens is weak (single-imaged) or we exhausted attempts
        break

    # grids in arcsec (super resolution)
    grid_lin_sup = np.linspace(-0.5 * IMAGE_SIZE * PIXEL_SCALE,
                               0.5 * IMAGE_SIZE * PIXEL_SCALE, SUPER_SIZE)
    x_sup, y_sup = np.meshgrid(grid_lin_sup, grid_lin_sup)

    # build realistic extended source
    src_pattern = build_sources(x_sup, y_sup, theta_E)

    # photometry & PSF & noise parameters
    exptime = float(np.random.uniform(*EXPTIME_RANGE))
    zp = DEFAULT_ZP
    psf_type = np.random.choice(PSF_TYPES)
    psf_fwhm = float(np.random.uniform(*PSF_FWHM_ARCSEC))
    psf_ellip = float(np.random.uniform(0.0, PSF_ELLIP_MAX))
    psf_angle = float(np.random.uniform(0, 2*np.pi))
    psf_beta = float(np.random.uniform(*MOFFAT_BETA_RANGE)) if psf_type == 'MOFFAT' else 3.5
    psf_sup = make_psf_kernel(psf_fwhm, SUP_PIXEL_SCALE, psf_type, ellip=psf_ellip, angle=psf_angle, beta=psf_beta)
    read_noise = float(np.random.uniform(*READ_NOISE_RANGE))
    sky_adu = float(np.random.uniform(*SKY_ADU_RANGE))

    # pick magnitudes
    src_mag = float(np.random.uniform(*SRC_MAG_RANGE))
    lens_mag = src_mag + float(np.random.uniform(*LENS_MAG_DELTA_RANGE))

    sb_src_sup, src_total_counts = normalize_to_sb_per_arcsec2(src_pattern, src_mag, zp, exptime, SUP_PIXEL_SCALE)
    src_counts_sup = (sb_src_sup * SUPER_PIX_AREA).astype(np.float32)

    # GT (unlensed, detector-sampled, NOT PSF-convolved)
    GT = downsample(src_counts_sup, OVERSAMPLE).astype(np.float32)

    # ray-shoot source to lensed plane
    x_flat, y_flat = x_sup.ravel(), y_sup.ravel()
    xs, ys = lens.ray_shooting(x_flat, y_flat, kwargs_lens)
    xs = xs.reshape(SUPER_SIZE, SUPER_SIZE)
    ys = ys.reshape(SUPER_SIZE, SUPER_SIZE)

    interp_sb = rg_interpolator(x_sup, y_sup, sb_src_sup, method='linear')
    pts = np.vstack([ys.ravel(), xs.ravel()]).T
    sb_mapped = interp_sb(pts).reshape(SUPER_SIZE, SUPER_SIZE)
    lensed_src_counts_sup = (sb_mapped * SUPER_PIX_AREA).astype(np.float32)

    # build lens light (on image plane, not lensed)
    lens_light_model = LightModel(['SERSIC_ELLIPSE'])
    lens_light_sup_pattern = lens_light_model.surface_brightness(x_sup, y_sup, [ {
        'amp': 1.0,
        'R_sersic': np.random.uniform(0.18, 0.9),
        'n_sersic': np.random.uniform(2.5, 5.0),
        'e1': -0.6 * e1, 'e2': -0.6 * e2,
        'center_x': center_x + np.random.uniform(-0.02, 0.02),
        'center_y': center_y + np.random.uniform(-0.02, 0.02)
    } ]).astype(np.float32)

    sb_lens_sup, lens_total_counts = normalize_to_sb_per_arcsec2(lens_light_sup_pattern, lens_mag, zp, exptime, SUP_PIXEL_SCALE)
    lens_counts_sup = (sb_lens_sup * SUPER_PIX_AREA).astype(np.float32)

    # combine and add faint field galaxies for realism
    image_sup = lensed_src_counts_sup + lens_counts_sup
    if np.random.rand() < 0.9:
        image_sup = add_field_galaxies(image_sup, n_fake=np.random.randint(3, 9))

    # Convolve with PSF on super-grid
    image_conv_sup = convolve_fft(image_sup, psf_sup, normalize_kernel=True, allow_huge=True)

    # Downsample to detector sampling
    LENSED_counts = downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)

    # measure peaks and adjust source/lens contrast mildly to avoid extremely dim/bright cases
    def peak99(arr):
        return float(np.percentile(arr, 99))

    peak_src = peak99(downsample(convolve_fft(lensed_src_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
    peak_lens = peak99(downsample(convolve_fft(lens_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
    ratio = (peak_src + 1e-9) / (peak_lens + 1e-9)
    target_ratio = 0.35
    tries = 0
    while ratio < target_ratio and tries < 3:
        src_mag -= 0.2
        lens_mag += 0.08
        sb_src_sup, src_total_counts = normalize_to_sb_per_arcsec2(src_pattern, src_mag, zp, exptime, SUP_PIXEL_SCALE)
        src_counts_sup = (sb_src_sup * SUPER_PIX_AREA).astype(np.float32)
        interp_sb = rg_interpolator(x_sup, y_sup, sb_src_sup, method='linear')
        sb_mapped = interp_sb(pts).reshape(SUPER_SIZE, SUPER_SIZE)
        lensed_src_counts_sup = (sb_mapped * SUPER_PIX_AREA).astype(np.float32)
        sb_lens_sup, lens_total_counts = normalize_to_sb_per_arcsec2(lens_light_sup_pattern, lens_mag, zp, exptime, SUP_PIXEL_SCALE)
        lens_counts_sup = (sb_lens_sup * SUPER_PIX_AREA).astype(np.float32)
        image_sup = lensed_src_counts_sup + lens_counts_sup
        image_conv_sup = convolve_fft(image_sup, psf_sup, normalize_kernel=True, allow_huge=True)
        LENSED_counts = downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)
        peak_src = peak99(downsample(convolve_fft(lensed_src_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
        peak_lens = peak99(downsample(convolve_fft(lens_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
        ratio = (peak_src + 1e-9) / (peak_lens + 1e-9)
        tries += 1

    # Add mild sky background with small gradient across image
    xg = np.linspace(-0.5, 0.5, IMAGE_SIZE)
    sky_gradient = (1.0 + SKY_GRADIENT_MAX * (xg - xg.mean()) / (xg.max() - xg.min()))
    sky_map = sky_adu * sky_gradient[np.newaxis, :]
    LENSED_counts = LENSED_counts + sky_map

    # Apply PRNU (multiplicative) BEFORE Poisson
    prnu_map = 1.0 + np.random.normal(0.0, PRNU_RMS, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    LENSED_counts_prnu = LENSED_counts * prnu_map

    # convert ADU -> electrons, Poisson noise, add read noise (electrons)
    e_image = np.clip(LENSED_counts_prnu * GAIN, 0.0, None)
    noisy_e = np.random.poisson(e_image).astype(np.float32)
    noisy_e += np.random.normal(0.0, read_noise, noisy_e.shape).astype(np.float32)

    # correlated noise (small) to mimic resampling/drizzle and detector correlations
    if np.random.rand() < 0.6:
        from scipy.ndimage import gaussian_filter
        corr_sigma = np.random.uniform(0.4, 1.1)
        noisy_e = gaussian_filter(noisy_e, corr_sigma)

    # occasional cosmic ray hits
    if np.random.rand() < COSMIC_RAY_PROB:
        n_hits = np.random.randint(1, 6)
        for _ in range(n_hits):
            cx = np.random.randint(0, IMAGE_SIZE)
            cy = np.random.randint(0, IMAGE_SIZE)
            intensity = np.random.uniform(*COSMIC_RAY_INTENSITY)
            noisy_e[cy, cx] += intensity * GAIN

    # slight vignetting occasionally
    if np.random.rand() < 0.12:
        yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))
        cy, cx = IMAGE_SIZE / 2, IMAGE_SIZE / 2
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        vign = 1.0 - 0.06 * (r / r.max())**1.4
        noisy_e *= vign

    # final back to ADU
    LENSED = (noisy_e / GAIN).astype(np.float32)

    GT_out = GT.astype(np.float32)

    # Save to FITS
    filename = os.path.join(outdir, f"rim_sim_weak_{i:05d}.fits")
    hdu_primary = fits.PrimaryHDU()
    hdu_gt = fits.ImageHDU(data=GT_out, name='GT')
    hdu_lensed = fits.ImageHDU(data=LENSED.astype(np.float32), name='LENSED')

    # add headers
    for hdu in [hdu_gt, hdu_lensed]:
        hdu.header['PIXSCALE'] = (PIXEL_SCALE, 'arcsec/pixel')
        hdu.header['BUNIT'] = ('ADU', 'Brightness unit')
        hdu.header['OVERSAMP'] = (OVERSAMPLE, 'super-sampling factor')
        hdu.header['ZP'] = (DEFAULT_ZP, 'AB zeropoint (ADU/s)')
        hdu.header['EXPTIME'] = (exptime, 'seconds')

    hdr = hdu_primary.header
    hdr['DATE'] = (datetime.utcnow().isoformat(), 'UTC creation time')
    hdr['GAIN'] = (GAIN, 'e-/ADU')
    hdr['RN_E'] = (read_noise, 'read noise e- RMS')
    hdr['SKYADU'] = (sky_adu, 'median sky level (ADU/pixel)')
    hdr['PSF_FWH'] = (psf_fwhm, 'PSF FWHM (arcsec)')
    hdr['PSF_TYP'] = (psf_type, 'PSF model type')
    hdr['THETA_E'] = (theta_E, 'Einstein radius (arcsec)')
    hdr['SIGMA'] = (sigma_kms, 'velocity dispersion km/s used to set thetaE')
    hdr['ELLIP'] = (e, 'SIE ellipticity')
    hdr['SHEAR'] = (shear_g, 'external shear amplitude')
    hdr['SRCMAG'] = (src_mag, 'source integrated AB mag')
    hdr['LENSMAG'] = (lens_mag, 'lens integrated AB mag')
    hdr['ZLENS'] = (z_l, 'lens redshift')
    hdr['ZSRC'] = (z_s, 'source redshift')
    hdr['PRNU'] = (PRNU_RMS, 'PRNU RMS (fractional)')
    hdr['SKYGRAD'] = (SKY_GRADIENT_MAX, 'max fractional sky gradient')

    hdul = fits.HDUList([hdu_primary, hdu_gt, hdu_lensed])
    hdul.writeto(filename, overwrite=True)

    print(f"[{i}] Saved {filename}  | peak(src/lens) ratio≈{ratio:.2f}, attempts={attempt+1}")
    return filename


def main():
    set_seed(SEED)
    for i in range(NUM_REALIZATIONS):
        generate_one(i, OUTPUT_DIR)


if __name__ == '__main__':
    main()
