"""
simgenv6.py - AstroRIM v2.1 (weak-lensing variant).

Fixes vs v2.0 'patched' version (see CHANGELOG.md for full details):
  [FIX S1] Moffat PSF radial profile corrected. The old kernel fed a
      sigma-NORMALIZED elliptical radius into a Moffat whose alpha was in
      PIXELS, so the realized FWHM was requested_FWHM * sigma_pix (e.g.
      3.0 px requested -> 3.83 px realized). The earlier "alpha formula"
      patch fixed the alpha line but not the radius fed into it; the bug
      therefore survived in ALL six variants including this one (the v2.0
      docstring claim that v6 was "already correct" was wrong). PSF
      construction is now delegated to simgen_truth.make_psf_kernel_fixed:
      realized FWHM == requested FWHM, Gaussian branch bit-identical to the
      old (correct) one, Airy unchanged.
  [FIX S2] The NFW group-halo block was dimensionally broken
      ('alpha_Rs': sqrt(4 c^2 M / mu(c)) * 0.001 -- the square root of a
      mass is not an angle; Rs in Mpc * 0.001 is not arcsec; R200 was also
      missing its factor of 3). Replaced with the physically derived
      conversion in simgen_truth.physical_nfw_group_halo (M200 ->
      Rs_arcsec, alpha_Rs via kappa_s; sanity: 1e14 Msun at z=0.5 gives
      kappa_s ~ 0.16, Rs ~ 26", alpha_Rs ~ 5").
  [NEW S3] Ground-truth TR_* header block written in the MODEL's parameter
      convention (astrorim_core.TRUTH_ORDER), with TR_MASK supervision bits
      and TR_CLEAN model-family flag. Conversions verified numerically
      against the forward operator (SIE exact in the rc->0 limit, SHEAR and
      NFW exact). Enables the predicted-vs-true scatter batch and the
      auxiliary parameter-supervision loss in training.py.

Earlier (v2.0) fixes retained: lens-light PA aligned with mass PA;
cosmology unified to Planck15.

Requires simgen_truth.py in the same directory.
Sample filename prefix: rim_simv6_{i:05d}.fits
"""

import os
import random
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy import units as u
from astropy import constants as const
# FIXED: was FlatLambdaCDM(H0=70, Om0=0.3). Unified with simgenv1-v5 which all use
# Planck15, giving ~1-1.5% consistent theta_E for the same velocity dispersion.
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import rotate, gaussian_filter
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import scipy.special
import scipy

import simgen_truth as sgt  # [FIX S1/S2, NEW S3] shared verified helpers

OUTPUT_DIR = r"C:\Users\mythi\AstroRIM\AstroRIM_2.1\Sims"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 96          # final detector pixels
OVERSAMPLE = 4           # super-sampling factor
PIXEL_SCALE = 0.04       # arcsec / pixel
DEFAULT_ZP = 25.94

NUM_REALIZATIONS = 150
SEED = None

# Use a more modern cosmology
# FIXED: previously instantiated FlatLambdaCDM(H0=70, Om0=0.3) here. Cosmo is now
# imported directly from Planck15 at the top of the file to match simgenv1-v5.

# Tuned for WEAK LENSING (mostly single-imaged, low magnification)
SIGMA_KMS_RANGE = (80.0, 180.0)    # lower velocity dispersions -> smaller Einstein radii
ELLIPTICITY_RANGE = (0.02, 0.45)
SHEAR_MAX = 0.06                    # small external shear (weak regime)

# More realistic exposure times for weak lensing surveys
EXPTIME_RANGE = (1200.0, 1800.0)
READ_NOISE_RANGE = (1.2, 2.2)      # More realistic for current instruments
GAIN = 1.5
SKY_ADU_RANGE = (0.06, 0.28)       # Adjusted for more realistic sky levels

# Source brightness - fainter sources more typical for weak lensing
SRC_MAG_RANGE = (21.5, 24.0)
LENS_MAG_DELTA_RANGE = (-0.1, +1.2)

# More realistic PSF parameters
PSF_TYPES = ('GAUSSIAN', 'MOFFAT', 'AIRY')  # Added Airy for diffraction-limited case
PSF_FWHM_ARCSEC = (0.07, 0.15)     # More realistic range for ground-based observations
PSF_ELLIP_MAX = 0.12               # Increased for more realistic optical aberrations
MOFFAT_BETA_RANGE = (2.5, 4.8)

# Extra source complexity
N_EXTRA_SOURCES_RANGE = (0, 3)     # More field galaxies
USE_CUTOUTS = False
CUTOUT_DIR = None

# More realistic detector artifacts
PRNU_RMS = 0.015                   # 1.5% PRNU more realistic
SKY_GRADIENT_MAX = 0.03
COSMIC_RAY_PROB = 0.03
COSMIC_RAY_INTENSITY = (40.0, 350.0)
CTI_EFFECT = True                  # Charge Transfer Inefficiency

# Lens model options for weak-lens emphasis
ADD_GROUP_HALO_PROB = 0.05
ADD_SUBHALOS_PROB = 0.15

# Derived
SUPER_SIZE = IMAGE_SIZE * OVERSAMPLE
SUP_PIXEL_SCALE = PIXEL_SCALE / OVERSAMPLE  # arcsec / super-pixel
SUPER_PIX_AREA = SUP_PIXEL_SCALE ** 2

# How many attempts to resample if we accidentally draw a strong lens
MAX_PARAMETER_ATTEMPTS = 10

# Precompute a realistic noise correlation kernel (from HSC measurements)
NOISE_CORR_KERNEL = np.array([[0.004, 0.016, 0.026, 0.016, 0.004],
                              [0.016, 0.086, 0.134, 0.086, 0.016],
                              [0.026, 0.134, 1.000, 0.134, 0.026],
                              [0.016, 0.086, 0.134, 0.086, 0.016],
                              [0.004, 0.016, 0.026, 0.016, 0.004]])


def set_seed(seed=None):
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rnd = int(seed) % (2**32 - 1)
    random.seed(rnd)
    np.random.seed(rnd)


def sigma_to_thetaE_arcsec(sigma_kms, z_lens, z_source):
    # More accurate calculation using the full cosmology
    D_d = cosmo.angular_diameter_distance(z_lens)
    D_s = cosmo.angular_diameter_distance(z_source)
    D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

    # Critical surface density (kept for completeness) using astropy.constants
    sigma_crit = (D_s / (D_d * D_ds) * (const.c**2) / (4 * np.pi * const.G)).to(u.kg / u.m**2)

    # Einstein radius for an SIS: theta_E = 4*pi*(sigma^2 / c^2) * (D_ds / D_s)  (radians)
    theta_E_rad = (4 * np.pi * (sigma_kms * 1000.0 * u.m / u.s) ** 2
                   / (const.c**2) * (D_ds / D_s))

    # Explicitly interpret as an angle in radians, then convert to arcseconds
    theta_E_arcsec = (theta_E_rad * u.rad).to(u.arcsec).value

    return float(theta_E_arcsec)


def make_psf_kernel(fwhm_arcsec, pixel_scale_arcsec, psf_type='GAUSSIAN', ellip=0.0, angle=0.0, beta=3.5):
    """[FIX S1] Delegates to simgen_truth.make_psf_kernel_fixed.

    The previous in-file implementation realized a Moffat of width
    sigma_pix * alpha instead of alpha (sigma-normalized radius divided by a
    pixel-unit alpha^2), so the realized FWHM differed from the requested one
    by the FWHM-dependent factor sigma_pix = FWHM_px/2.3548. The fixed
    version realizes the requested FWHM exactly; the Gaussian branch is
    bit-identical to the old (already-correct) one and Airy is unchanged.

    Returns (kernel, info) -- info carries realized_fwhm_pix (in the pixels
    of `pixel_scale_arcsec`) and coma_applied, needed by the truth header.
    """
    return sgt.make_psf_kernel_fixed(fwhm_arcsec, pixel_scale_arcsec,
                                     psf_type=psf_type, ellip=ellip,
                                     angle=angle, beta=beta,
                                     rng=np.random, coma_prob=0.4)


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
    """Build more realistic extended sources with better morphology diversity."""
    # Decide on source type: spiral, elliptical, or irregular
    source_type = np.random.choice(['spiral', 'elliptical', 'irregular'],
                                  p=[0.5, 0.3, 0.2])

    cx = np.random.uniform(-0.15 * theta_E, 0.15 * theta_E)
    cy = np.random.uniform(-0.15 * theta_E, 0.15 * theta_E)

    if source_type == 'elliptical':
        # Single Sersic profile for ellipticals
        lm = LightModel(['SERSIC_ELLIPSE'])
        Reff = np.random.uniform(0.02, 0.08)  # Effective radius in arcsec
        n_sersic = np.random.uniform(2.5, 6.0)  # De Vaucouleurs-like

        e = np.random.uniform(0.0, 0.7)
        phi = np.random.uniform(0, np.pi)
        e1 = e * np.cos(2 * phi)
        e2 = e * np.sin(2 * phi)

        kwargs = {'amp': 1.0, 'R_sersic': Reff, 'n_sersic': n_sersic,
                 'e1': e1, 'e2': e2, 'center_x': cx, 'center_y': cy}
        main = lm.surface_brightness(x_sup, y_sup, [kwargs]).astype(np.float32)

    elif source_type == 'spiral':
        # Bulge + disk for spirals
        # Use two single-component LightModel instances (one per component)
        lm_bulge = LightModel(['SERSIC_ELLIPSE'])
        lm_disk = LightModel(['SERSIC_ELLIPSE'])

        # Bulge parameters
        Reff_bulge = np.random.uniform(0.01, 0.04)
        n_bulge = np.random.uniform(2.0, 4.0)

        # Disk parameters
        Reff_disk = np.random.uniform(1.5, 3.0) * Reff_bulge
        n_disk = 1.0  # Exponential disk

        # Ellipticity and orientation
        e = np.random.uniform(0.4, 0.8)
        phi = np.random.uniform(0, np.pi)
        e1 = e * np.cos(2 * phi)
        e2 = e * np.sin(2 * phi)

        # Slight misalignment between bulge and disk
        phi_disk = phi + np.random.uniform(-0.2, 0.2)
        e1_disk = e * np.cos(2 * phi_disk)
        e2_disk = e * np.sin(2 * phi_disk)

        kwargs_bulge = {'amp': np.random.uniform(0.3, 1.0), 'R_sersic': Reff_bulge,
                       'n_sersic': n_bulge, 'e1': e1*0.7, 'e2': e2*0.7,
                       'center_x': cx, 'center_y': cy}
        kwargs_disk = {'amp': np.random.uniform(0.5, 1.5), 'R_sersic': Reff_disk,
                      'n_sersic': n_disk, 'e1': e1_disk, 'e2': e2_disk,
                      'center_x': cx + np.random.uniform(-0.01, 0.01),
                      'center_y': cy + np.random.uniform(-0.01, 0.01)}

        # Evaluate bulge and disk separately using their own LightModel instances
        bulge = lm_bulge.surface_brightness(x_sup, y_sup, [kwargs_bulge]).astype(np.float32)
        disk = lm_disk.surface_brightness(x_sup, y_sup, [kwargs_disk]).astype(np.float32)
        main = bulge + disk

        # Add spiral arms to some spirals
        if np.random.rand() < 0.7:
            X, Y = np.meshgrid(np.linspace(-1, 1, main.shape[0]),
                              np.linspace(-1, 1, main.shape[1]))
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            spiral = 1.0 + 0.1 * np.sin(3.0 * theta + 5 * r + np.random.uniform(0, 2*np.pi))
            main *= np.clip(spiral, 0.7, 1.3)

    else:  # irregular
        lm = LightModel(['SERSIC_ELLIPSE'])
        Reff = np.random.uniform(0.03, 0.09)
        n_sersic = np.random.uniform(0.8, 2.5)  # Lower Sersic index for irregulars

        e = np.random.uniform(0.0, 0.5)
        phi = np.random.uniform(0, np.pi)
        e1 = e * np.cos(2 * phi)
        e2 = e * np.sin(2 * phi)

        kwargs = {'amp': 1.0, 'R_sersic': Reff, 'n_sersic': n_sersic,
                 'e1': e1, 'e2': e2, 'center_x': cx, 'center_y': cy}
        main = lm.surface_brightness(x_sup, y_sup, [kwargs]).astype(np.float32)

        # Make irregulars more clumpy
        main = add_clumps(main, n=np.random.randint(3, 8),
                         max_rel=np.random.uniform(0.2, 0.8),
                         sig_pix=(OVERSAMPLE * 0.3, OVERSAMPLE * 2.5))

    # Add star-forming regions to some galaxies
    if np.random.rand() < 0.6:
        main = add_clumps(main, n=np.random.randint(1, 5),
                         max_rel=np.random.uniform(0.1, 0.4),
                         sig_pix=(OVERSAMPLE * 0.2, OVERSAMPLE * 1.5))

    # Add companion galaxies
    n_extra = np.random.randint(N_EXTRA_SOURCES_RANGE[0], N_EXTRA_SOURCES_RANGE[1] + 1)
    if n_extra > 0:
        lm_extra = LightModel(['SERSIC_ELLIPSE'])
        for _ in range(n_extra):
            cx2 = np.random.uniform(-1.2 * theta_E, 1.2 * theta_E)
            cy2 = np.random.uniform(-1.2 * theta_E, 1.2 * theta_E)
            kwargs_e = {
                'amp': np.random.uniform(0.1, 0.8),
                'R_sersic': np.random.uniform(0.008, 0.06),
                'n_sersic': np.random.uniform(0.8, 3.5),
                'e1': np.random.uniform(-0.4, 0.4),
                'e2': np.random.uniform(-0.4, 0.4),
                'center_x': cx2, 'center_y': cy2
            }
            main += lm_extra.surface_brightness(x_sup, y_sup, [kwargs_e]).astype(np.float32)

    return np.clip(main, 0.0, None).astype(np.float32)


def add_field_galaxies(image_sup, n_fake=8):
    """Add more realistic field galaxies and stars."""
    lm_gal = LightModel(['SERSIC_ELLIPSE'])
    S = image_sup.shape[0]
    grid_lin = np.linspace(-0.5 * IMAGE_SIZE * PIXEL_SCALE,
                           0.5 * IMAGE_SIZE * PIXEL_SCALE, S)
    x_sup, y_sup = np.meshgrid(grid_lin, grid_lin)

    n_galaxies = int(n_fake * 0.8)  # 80% galaxies, 20% stars
    n_stars = n_fake - n_galaxies

    # Add galaxies
    for _ in range(n_galaxies):
        amp = np.random.uniform(0.01, 0.3)
        R = np.random.uniform(0.006, 0.07)
        n = np.random.uniform(0.8, 3.5)
        cx = np.random.uniform(-0.5 * IMAGE_SIZE * PIXEL_SCALE, 0.5 * IMAGE_SIZE * PIXEL_SCALE)
        cy = np.random.uniform(-0.5 * IMAGE_SIZE * PIXEL_SCALE, 0.5 * IMAGE_SIZE * PIXEL_SCALE)
        e1 = np.random.uniform(-0.5, 0.5)
        e2 = np.random.uniform(-0.5, 0.5)
        gal = lm_gal.surface_brightness(x_sup, y_sup, [{
            'amp': amp, 'R_sersic': R, 'n_sersic': n,
            'e1': e1, 'e2': e2, 'center_x': cx, 'center_y': cy
        }])
        image_sup += gal.astype(np.float32)

    # Add stars (Moffat profiles)
    for _ in range(n_stars):
        amp = np.random.uniform(0.05, 0.4)
        fwhm = np.random.uniform(0.04, 0.12)  # arcsec
        beta = np.random.uniform(2.5, 4.5)
        cx = np.random.uniform(-0.5 * IMAGE_SIZE * PIXEL_SCALE, 0.5 * IMAGE_SIZE * PIXEL_SCALE)
        cy = np.random.uniform(-0.5 * IMAGE_SIZE * PIXEL_SCALE, 0.5 * IMAGE_SIZE * PIXEL_SCALE)

        # Create Moffat kernel
        alpha = fwhm / (2 * np.sqrt(2**(1.0/beta) - 1))
        r2 = ((x_sup - cx)**2 + (y_sup - cy)**2) / (alpha**2)
        star = amp * (1 + r2)**(-beta)
        image_sup += star.astype(np.float32)

    return image_sup


def is_strong_for_source(lens_solver, src_x, src_y, kwargs_lens):
    """More robust check for strong lensing with multiple test positions."""
    try:
        imgs = lens_solver.image_position_from_source(src_x, src_y, kwargs_lens)
        if imgs is None:
            return False
        n_img = len(imgs)
        return n_img > 1
    except Exception:
        return True


def apply_cti_effect(image, direction='vertical', trap_density=0.1, trap_release_time=0.5):
    """Apply Charge Transfer Inefficiency effect to simulate real CCDs."""
    if not CTI_EFFECT:
        return image

    result = image.copy()
    ny, nx = image.shape

    if direction == 'vertical':
        for x in range(nx):
            column = image[:, x].copy()
            for y in range(1, ny):
                # Simulate charge trapping and release
                trapped = column[y] * trap_density
                column[y] -= trapped
                # Release previously trapped charge
                released = trap_release_time * trapped
                column[y] += released
            result[:, x] = column
    else:  # horizontal
        for y in range(ny):
            row = image[y, :].copy()
            for x in range(1, nx):
                trapped = row[x] * trap_density
                row[x] -= trapped
                released = trap_release_time * trapped
                row[x] += released
            result[y, :] = row

    return result


def generate_one(i, outdir=OUTPUT_DIR):
    for attempt in range(MAX_PARAMETER_ATTEMPTS):
        # Redshifts with more realistic distribution
        z_l = float(np.random.uniform(0.3, 0.8))
        z_s = float(np.random.uniform(z_l + 0.1, 2.5))  # More typical source redshifts

        # Physical lens strength via sigma with more realistic distribution
        sigma_kms = float(np.random.normal(130, 25))
        sigma_kms = max(SIGMA_KMS_RANGE[0], min(SIGMA_KMS_RANGE[1], sigma_kms))
        theta_E = float(sigma_to_thetaE_arcsec(sigma_kms, z_l, z_s))

        # More realistic ellipticity distribution
        e = float(np.random.beta(1.5, 3) * (ELLIPTICITY_RANGE[1] - ELLIPTICITY_RANGE[0]) + ELLIPTICITY_RANGE[0])
        phi = np.random.uniform(0, np.pi)
        e1 = e * np.cos(2 * phi)
        e2 = e * np.sin(2 * phi)

        # External shear with more realistic amplitude distribution
        shear_g = float(np.random.gamma(1.5, 0.02))
        shear_g = min(shear_g, SHEAR_MAX)
        shear_pa = np.random.uniform(0, np.pi)
        g1 = shear_g * np.cos(2 * shear_pa)
        g2 = shear_g * np.sin(2 * shear_pa)

        # Lens model with more realistic parameters
        lens_model_list = ['SIE']
        center_x = np.random.uniform(-0.03, 0.03)
        center_y = np.random.uniform(-0.03, 0.03)
        kwargs_lens = [{
            'theta_E': theta_E,
            'e1': e1, 'e2': e2,
            'center_x': center_x,
            'center_y': center_y
        }]

        if shear_g > 0.0:
            lens_model_list.append('SHEAR')
            kwargs_lens.append({'gamma1': g1, 'gamma2': g2})

        # [FIX S2] Group halo with PHYSICAL parameter derivation.
        # The previous block was dimensionally broken: 'alpha_Rs' was set to
        # sqrt(4 c^2 M / mu(c)) * 0.001 (the square root of a mass is not a
        # deflection angle), 'Rs' was R200/c in Mpc times 0.001 (not arcsec),
        # and the R200 formula was missing its factor of 3. The 5% of v6
        # images carrying a group halo therefore had physically meaningless
        # halo deflections. simgen_truth.physical_nfw_group_halo derives
        # (Rs_arcsec, alpha_Rs) from M200 via the standard chain
        # (R200 -> c(M) -> Rs -> rho_s -> Sigma_crit -> kappa_s -> alpha_Rs).
        if np.random.rand() < ADD_GROUP_HALO_PROB:
            lens_model_list.append('NFW')
            mass = 10**np.random.uniform(13.0, 14.5)  # M_sun (M200c)
            nfw_kwargs, _nfw_info = sgt.physical_nfw_group_halo(
                mass, z_l, z_s, cosmo,
                center_x + np.random.uniform(-0.3, 0.3),
                center_y + np.random.uniform(-0.3, 0.3))
            kwargs_lens.append(nfw_kwargs)

        # Subhalos with more realistic mass function
        if np.random.rand() < ADD_SUBHALOS_PROB:
            n_sub = np.random.poisson(0.5)  # Fewer subhalos
            for _ in range(max(0, n_sub)):
                lens_model_list.append('SIS')
                sub_theta_E = np.random.uniform(0.003, 0.02)
                kwargs_lens.append({
                    'theta_E': sub_theta_E,
                    'center_x': np.random.uniform(-0.5 * theta_E, 0.5 * theta_E),
                    'center_y': np.random.uniform(-0.5 * theta_E, 0.5 * theta_E)
                })

        lens = LensModel(lens_model_list)

        # Check for strong lensing at multiple positions
        lens_solver = LensEquationSolver(lens)
        strong_lensing = False
        for _ in range(5):  # Test multiple positions
            trial_src_x = np.random.uniform(-0.1 * theta_E, 0.1 * theta_E)
            trial_src_y = np.random.uniform(-0.1 * theta_E, 0.1 * theta_E)
            if is_strong_for_source(lens_solver, trial_src_x, trial_src_y, kwargs_lens):
                strong_lensing = True
                break

        if strong_lensing and attempt < MAX_PARAMETER_ATTEMPTS - 1:
            continue

        break

    # Grids in arcsec (super resolution)
    grid_lin_sup = np.linspace(-0.5 * IMAGE_SIZE * PIXEL_SCALE,
                               0.5 * IMAGE_SIZE * PIXEL_SCALE, SUPER_SIZE)
    x_sup, y_sup = np.meshgrid(grid_lin_sup, grid_lin_sup)

    # Build realistic extended source
    src_pattern = build_sources(x_sup, y_sup, theta_E)

    # Photometry and PSF & noise parameters
    exptime = float(np.random.uniform(*EXPTIME_RANGE))
    zp = DEFAULT_ZP
    psf_type = np.random.choice(PSF_TYPES)
    psf_fwhm = float(np.random.uniform(*PSF_FWHM_ARCSEC))
    psf_ellip = float(np.random.uniform(0.0, PSF_ELLIP_MAX))
    psf_angle = float(np.random.uniform(0, 2*np.pi))
    psf_beta = float(np.random.uniform(*MOFFAT_BETA_RANGE)) if psf_type == 'MOFFAT' else 3.5
    # [FIX S1] kernel + realized-PSF info (needed for the truth header)
    psf_sup, psf_kernel_info = make_psf_kernel(psf_fwhm, SUP_PIXEL_SCALE, psf_type, ellip=psf_ellip, angle=psf_angle, beta=psf_beta)
    read_noise = float(np.random.uniform(*READ_NOISE_RANGE))
    sky_adu = float(np.random.uniform(*SKY_ADU_RANGE))

    # Pick magnitudes with more realistic distributions
    src_mag = float(np.random.normal(22.5, 0.8))
    src_mag = max(SRC_MAG_RANGE[0], min(SRC_MAG_RANGE[1], src_mag))
    lens_mag = src_mag + float(np.random.uniform(*LENS_MAG_DELTA_RANGE))

    sb_src_sup, src_total_counts = normalize_to_sb_per_arcsec2(src_pattern, src_mag, zp, exptime, SUP_PIXEL_SCALE)
    src_counts_sup = (sb_src_sup * SUPER_PIX_AREA).astype(np.float32)

    # GT (unlensed, detector-sampled, NOT PSF-convolved)
    GT = downsample(src_counts_sup, OVERSAMPLE).astype(np.float32)

    # Ray-shoot source to lensed plane
    x_flat, y_flat = x_sup.ravel(), y_sup.ravel()
    xs, ys = lens.ray_shooting(x_flat, y_flat, kwargs_lens)
    xs = xs.reshape(SUPER_SIZE, SUPER_SIZE)
    ys = ys.reshape(SUPER_SIZE, SUPER_SIZE)

    interp_sb = rg_interpolator(x_sup, y_sup, sb_src_sup, method='linear')
    pts = np.vstack([ys.ravel(), xs.ravel()]).T
    sb_mapped = interp_sb(pts).reshape(SUPER_SIZE, SUPER_SIZE)
    lensed_src_counts_sup = (sb_mapped * SUPER_PIX_AREA).astype(np.float32)

    # Build lens light with more realistic profile
    # [NEW S3] parameters drawn into named variables so the truth header can
    # record them (same distributions as before).
    lens_light_model = LightModel(['SERSIC_ELLIPSE'])
    ll_Re_arcsec = np.random.uniform(0.2, 1.0)
    ll_n_sersic = np.random.uniform(2.0, 4.5)
    ll_center_x = center_x + np.random.uniform(-0.015, 0.015)
    ll_center_y = center_y + np.random.uniform(-0.015, 0.015)
    lens_light_sup_pattern = lens_light_model.surface_brightness(x_sup, y_sup, [ {
        'amp': 1.0,
        'R_sersic': ll_Re_arcsec,
        'n_sersic': ll_n_sersic,
        # FIXED: lens light PA must be ALIGNED with mass PA, not orthogonal.
        # Previous convention (-0.8 * e_mass) flipped (e1, e2) -> rotation of position
        # angle by 90 degrees, which is physically wrong. In real ellipticals the mass
        # and light PAs agree within ~10 degrees. Keep positive sign; slight random
        # scaling so light and mass axis ratios differ naturally.
        'e1': e1 * np.random.uniform(0.6, 1.0) + np.random.uniform(-0.05, 0.05),
        'e2': e2 * np.random.uniform(0.6, 1.0) + np.random.uniform(-0.05, 0.05),
        'center_x': ll_center_x,
        'center_y': ll_center_y
    } ]).astype(np.float32)

    sb_lens_sup, lens_total_counts = normalize_to_sb_per_arcsec2(lens_light_sup_pattern, lens_mag, zp, exptime, SUP_PIXEL_SCALE)
    lens_counts_sup = (sb_lens_sup * SUPER_PIX_AREA).astype(np.float32)

    # Combine and add field galaxies
    image_sup = lensed_src_counts_sup + lens_counts_sup
    if np.random.rand() < 0.95:  # Almost always add field objects
        image_sup = add_field_galaxies(image_sup, n_fake=np.random.randint(4, 12))

    # Convolve with PSF on super-grid
    image_conv_sup = convolve_fft(image_sup, psf_sup, normalize_kernel=True, allow_huge=True)

    # Downsample to detector sampling
    LENSED_counts = downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)

    # Measure peaks and adjust source/lens contrast
    def peak99(arr):
        return float(np.percentile(arr, 99))

    peak_src = peak99(downsample(convolve_fft(lensed_src_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
    peak_lens = peak99(downsample(convolve_fft(lens_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
    ratio = (peak_src + 1e-9) / (peak_lens + 1e-9)

    # More sophisticated contrast adjustment
    target_ratio = np.random.uniform(0.25, 0.45)
    tries = 0
    while abs(ratio - target_ratio) > 0.15 and tries < 4:
        if ratio < target_ratio:
            src_mag -= 0.15
            lens_mag += 0.06
        else:
            src_mag += 0.15
            lens_mag -= 0.06

        src_mag = max(SRC_MAG_RANGE[0], min(SRC_MAG_RANGE[1], src_mag))
        lens_mag = src_mag + max(LENS_MAG_DELTA_RANGE[0],
                                min(LENS_MAG_DELTA_RANGE[1], lens_mag - src_mag))

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

    # Add sky background with gradient
    xg, yg = np.meshgrid(np.linspace(-0.5, 0.5, IMAGE_SIZE),
                         np.linspace(-0.5, 0.5, IMAGE_SIZE))
    sky_gradient = (1.0 + SKY_GRADIENT_MAX * (xg * np.random.uniform(-1, 1) +
                     yg * np.random.uniform(-1, 1)))
    sky_map = sky_adu * sky_gradient
    LENSED_counts = LENSED_counts + sky_map

    # Apply PRNU (multiplicative) BEFORE Poisson
    prnu_map = 1.0 + np.random.normal(0.0, PRNU_RMS, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    LENSED_counts_prnu = LENSED_counts * prnu_map

    # Convert ADU -> electrons, Poisson noise, add read noise (electrons)
    e_image = np.clip(LENSED_counts_prnu * GAIN, 0.0, None)
    noisy_e = np.random.poisson(e_image).astype(np.float32)
    noisy_e += np.random.normal(0.0, read_noise, noisy_e.shape).astype(np.float32)

    # Apply correlated noise using a realistic kernel
    if np.random.rand() < 0.8:
        noisy_e = convolve_fft(noisy_e, NOISE_CORR_KERNEL, normalize_kernel=True)

    # Apply CTI effect
    if CTI_EFFECT and np.random.rand() < 0.7:
        noisy_e = apply_cti_effect(noisy_e, direction='vertical' if np.random.rand() < 0.5 else 'horizontal')

    # Cosmic ray hits
    if np.random.rand() < COSMIC_RAY_PROB:
        n_hits = np.random.randint(1, 5)
        for _ in range(n_hits):
            cx = np.random.randint(2, IMAGE_SIZE-2)
            cy = np.random.randint(2, IMAGE_SIZE-2)
            intensity = np.random.uniform(*COSMIC_RAY_INTENSITY)
            # Cosmic rays affect multiple pixels
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= cx+dx < IMAGE_SIZE and 0 <= cy+dy < IMAGE_SIZE:
                        noisy_e[cy+dy, cx+dx] += intensity * GAIN * (0.3 + 0.7 * np.random.rand())

    # Vignetting
    if np.random.rand() < 0.15:
        yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))
        cy, cx = IMAGE_SIZE / 2, IMAGE_SIZE / 2
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        vign = 1.0 - 0.08 * (r / r.max())**1.8
        noisy_e *= vign

    # Final back to ADU
    LENSED = (noisy_e / GAIN).astype(np.float32)

    GT_out = GT.astype(np.float32)

    # Save to FITS
    filename = os.path.join(outdir, f"rim_simv6_{i:05d}.fits")
    hdu_primary = fits.PrimaryHDU()
    hdu_gt = fits.ImageHDU(data=GT_out, name='GT')
    hdu_lensed = fits.ImageHDU(data=LENSED.astype(np.float32), name='LENSED')

    # Add headers with more metadata
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
    hdr['COSMO_H0'] = (cosmo.H0.value, 'Hubble constant [km/s/Mpc]')
    hdr['COSMO_OM'] = (cosmo.Om0, 'Matter density parameter')

    # ------------------------------------------------------------------
    # [NEW S3] Ground-truth TR_* block in the MODEL's parameter convention.
    # Conversions are the numerically verified ones in simgen_truth (SIE
    # b == theta_E/half_fov exact in the rc->0 limit; SHEAR and NFW exact).
    # The lens-light amplitude label is measured from the final (post
    # contrast-adjustment) pre-PSF detector lens-light image in the SAME
    # normalized units the training dataset will produce.
    # ------------------------------------------------------------------
    realized_fwhm_arcsec = psf_kernel_info['realized_fwhm_pix'] * SUP_PIXEL_SCALE
    lens_light_det = downsample(lens_counts_sup, OVERSAMPLE).astype(np.float32)
    norm_m = float(max(GT_out.max(), LENSED.max(), 1e-8))
    ll_amp = sgt.measure_lens_light_amp_at_Re(
        lens_light_det, (ll_center_x, ll_center_y), ll_Re_arcsec,
        PIXEL_SCALE, norm_m)
    truth, maskbits, clean = sgt.truth_from_lenstronomy(
        lens_model_list, kwargs_lens,
        pixel_scale=PIXEL_SCALE, image_size=IMAGE_SIZE,
        psf_info={
            'realized_fwhm_pix': realized_fwhm_arcsec / PIXEL_SCALE,
            'psf_type': psf_type, 'beta': psf_beta,
            'ellip': psf_ellip, 'angle': psf_angle,
            'coma_applied': psf_kernel_info['coma_applied'],
        },
        lens_light_info={
            'Re_arcsec': ll_Re_arcsec, 'n_sersic': ll_n_sersic,
            'amp_at_Re_normalized': ll_amp,
        })
    sgt.write_truth_header(hdr, truth, maskbits, clean, sim_set='simgenv6',
                           pixel_scale=PIXEL_SCALE,
                           coma_applied=psf_kernel_info['coma_applied'])

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
