import os
import random
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift as nd_shift
from scipy.signal import fftconvolve
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel


OUTPUT_DIR = r"YOUR OUTPUT PATH HERE"  
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 96        
OVERSAMPLE = 4         
PIXEL_SCALE = 0.04     
DEFAULT_ZP = 25.94     

NUM_REALIZATIONS = 150
SEED = None                


INSTRUMENTS = [
    (0.45, (0.06, 0.12), (0.03, 0.20), (0.8, 1.8), 0.04),  # space-like
    (0.45, (0.5, 1.2), (0.2, 3.0), (3.0, 8.0), 0.20),     # ground-like (worse seeing, larger pixel)
    (0.10, (0.12, 0.35), (0.05, 0.8), (1.0, 4.0), 0.08),   # intermediate
]

# Lens physics ranges
SIGMA_KMS_RANGE = (100.0, 330.0)  
ELLIPTICITY_RANGE = (0.0, 0.5)
SHEAR_MAX = 0.12

# Source/lens brightness & contrast
SRC_MAG_RANGE = (20.0, 25.5)
LENS_MAG_DELTA_RANGE = (-2.0, +3.0) 

# PSF parameters
PSF_TYPES = ['GAUSSIAN', 'MOFFAT']
PSF_MOFFAT_BETA_RANGE = (2.5, 4.5)
PSF_ELLIP_MAX = 0.35

# Noise & artifacts knobs
GAIN = 1.0
READ_NOISE_RANGE = (0.8, 8.0)
SKY_ADU_RANGE = (0.02, 3.0)
PRNU_RMS = 0.01               
SKY_GRADIENT_MAX = 0.03       
COSMIC_RAY_PROB = 0.03
COSMIC_RAY_INTENSITY = (30.0, 400.0)  # ADU

# Substructures & group halos
ADD_GROUP_HALO_PROB = 0.18
ADD_SUBHALOS_PROB = 0.7

# Realism toggles
ADD_CORRELATED_NOISE_PROB = 0.6
CORRELATED_NOISE_SIGMA_RANGE = (0.8, 4.0)  
ADD_SKY_SUBTRACTION_RESIDUALS_PROB = 0.6
ADD_SUBPIXEL_SHIFT_PROB = 0.85
ADD_PSF_MISMATCH_PROB = 0.6

# source complexity
N_EXTRA_SOURCES_RANGE = (0, 3)

# derived
SUPER_SIZE = IMAGE_SIZE * OVERSAMPLE
SUP_PIXEL_SCALE = PIXEL_SCALE / OVERSAMPLE
SUPER_PIX_AREA = SUP_PIXEL_SCALE ** 2



def set_seed(seed=None):
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rnd = int(seed) % (2**32 - 1)
    random.seed(rnd)
    np.random.seed(rnd)

def sigma_to_thetaE_arcsec(sigma_kms, z_lens, z_source):
    c = 299792.458  # km/s
    D_s = cosmo.angular_diameter_distance(z_source).to(u.km).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.km).value
    theta_rad = 4.0 * np.pi * (sigma_kms**2) / (c**2) * (D_ls / D_s)
    return theta_rad * (180.0 / np.pi) * 3600.0

def make_psf_kernel(fwhm_arcsec, pixel_scale_arcsec, psf_type='GAUSSIAN', ellip=0.0, angle=0.0, beta=3.5):
    """
    Elliptical Gaussian or Moffat kernel on super-grid.
    Returns normalized kernel (sum=1) as float32.
    """
    fwhm_pix = fwhm_arcsec / pixel_scale_arcsec
    sigma_pix = fwhm_pix / 2.3548
    # axis ratio q from ellipticity (ellip = 1 - q)
    q = max(0.02, 1.0 - ellip)
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
    """
    Normalize pattern to counts per arcsec^2 on super-grid (so it scales correctly with pixel scale ofc).
    """
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

def add_clumps(base, n=3, max_rel=0.6, sig_pix=(1.0, 5.0)):
    sup = base.copy()
    S = sup.shape[0]
    for _ in range(n):
        cx = np.random.uniform(0.2*S, 0.8*S)
        cy = np.random.uniform(0.2*S, 0.8*S)
        amp = np.random.uniform(0.02, max_rel) * (sup.max() if sup.max() > 0 else 1.0)
        sigma = np.random.uniform(*sig_pix)
        X, Y = np.meshgrid(np.arange(S), np.arange(S))
        sup += amp * np.exp(-0.5 * (((X - cx)**2 + (Y - cy)**2) / (sigma**2)))
    return np.clip(sup, 0.0, None)

def build_sources(x_sup, y_sup, theta_E):
    """
    Build a multi-component source (bulge+disk possibly spiral + clumps + extras).
    Position sampling allows both near-caustic and off-caustic sources to produce a wide range of lens morphologies.
    """
    lm = LightModel(['SERSIC_ELLIPSE'])
    # position: sometimes near caustic, often off-caustic
    if np.random.rand() < 0.28:   # 28% strong-arc placement
        cx = np.random.uniform(-0.12 * theta_E, 0.12 * theta_E)
        cy = np.random.uniform(-0.12 * theta_E, 0.12 * theta_E)
    else:
        cx = np.random.uniform(-0.6 * theta_E, 0.6 * theta_E)
        cy = np.random.uniform(-0.6 * theta_E, 0.6 * theta_E)

    frac_bulge = np.random.beta(1.6, 3.0)
    Rb = np.random.uniform(0.03, 0.14)
    Rd = Rb * np.random.uniform(1.4, 3.6)
    n_bulge = np.random.uniform(1.5, 4.5)
    amp_b = np.random.uniform(0.8, 5.0)
    amp_d = amp_b * np.random.uniform(0.2, 1.1)

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

    # extras (small satellites / star forming clumps)
    n_extra = np.random.randint(N_EXTRA_SOURCES_RANGE[0], N_EXTRA_SOURCES_RANGE[1] + 1)
    for _ in range(n_extra):
        cx2 = np.random.uniform(-0.9 * theta_E, 0.9 * theta_E)
        cy2 = np.random.uniform(-0.9 * theta_E, 0.9 * theta_E)
        kwargs_e = {
            'amp': np.random.uniform(0.2, 2.0),
            'R_sersic': np.random.uniform(0.015, 0.10),
            'n_sersic': np.random.uniform(0.6, 3.5),
            'e1': np.random.uniform(-0.4, 0.4),
            'e2': np.random.uniform(-0.4, 0.4),
            'center_x': cx2, 'center_y': cy2
        }
        main += lm.surface_brightness(x_sup, y_sup, [kwargs_e]).astype(np.float32)

    # clumpy structure and optional spiral multipliers
    if np.random.rand() < 0.75:
        main = add_clumps(main, n=np.random.randint(1, 6), max_rel=np.random.uniform(0.15, 0.7),
                          sig_pix=(OVERSAMPLE * 0.4, OVERSAMPLE * 4.0))

    if np.random.rand() < 0.35:
        X, Y = np.meshgrid(np.linspace(-1, 1, main.shape[0]), np.linspace(-1, 1, main.shape[1]))
        spiral = 1.0 + 0.08 * np.sin(3.0 * np.arctan2(Y, X) + np.random.uniform(0, 2*np.pi))
        main *= spiral

    return np.clip(main, 0.0, None).astype(np.float32)


def generate_one(i, outdir=OUTPUT_DIR):
    # choose instrument regime probabilistically
    r = np.random.rand()
    cum = 0.0
    for p, pfwhm, psky, prn, pscale in INSTRUMENTS:
        cum += p
        if r <= cum:
            inst_psf_fwhm_range = pfwhm
            inst_sky_range = psky
            inst_read_noise_range = prn
            inst_pixel_scale = pscale
            break

    PIXEL_SCALE_THIS = inst_pixel_scale
    SUP_PIXEL_SCALE = PIXEL_SCALE_THIS / OVERSAMPLE
    SUPER_PIX_AREA = SUP_PIXEL_SCALE ** 2

    # redshifts
    z_l = float(np.random.uniform(0.25, 0.7))
    z_s = float(np.random.uniform(max(z_l + 0.05, 0.85), 3.0))

    # physical lens strength via sigma -> theta_E
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

    # lens model list (SIE always)
    lens_model_list = ['SIE']
    kwargs_lens = [ {
        'theta_E': theta_E,
        'e1': e1, 'e2': e2,
        'center_x': np.random.uniform(-0.04, 0.04),
        'center_y': np.random.uniform(-0.04, 0.04)
    } ]
    if shear_g > 0.0:
        lens_model_list.append('SHEAR')
        kwargs_lens.append({'gamma1': g1, 'gamma2': g2})

    # optional group halo (NFW)
    if np.random.rand() < ADD_GROUP_HALO_PROB:
        lens_model_list.append('NFW')
        kwargs_lens.append({
            'alpha_Rs': 0.5,
            'Rs': np.random.uniform(5.0, 25.0),
            'center_x': kwargs_lens[0]['center_x'] + np.random.uniform(-0.3, 0.3),
            'center_y': kwargs_lens[0]['center_y'] + np.random.uniform(-0.3, 0.3)
        })

    # optional subhalos (SIS)
    if np.random.rand() < ADD_SUBHALOS_PROB:
        n_sub = np.random.poisson(1.0)
        for _ in range(max(0, n_sub)):
            lens_model_list.append('SIS')
            kwargs_lens.append({
                'theta_E': np.random.uniform(0.008, 0.06),
                'center_x': np.random.uniform(-0.5 * theta_E, 0.5 * theta_E),
                'center_y': np.random.uniform(-0.5 * theta_E, 0.5 * theta_E)
            })

    lens = LensModel(lens_model_list)

    # grids (super resolution)
    grid_lin_sup = np.linspace(-0.5 * IMAGE_SIZE * PIXEL_SCALE_THIS,
                               0.5 * IMAGE_SIZE * PIXEL_SCALE_THIS, SUPER_SIZE)
    x_sup, y_sup = np.meshgrid(grid_lin_sup, grid_lin_sup)

    # build sources pattern
    src_pattern = build_sources(x_sup, y_sup, theta_E)

    # photometry & PSF & noise
    exptime = float(np.random.uniform(800.0, 1600.0))
    zp = DEFAULT_ZP
    psf_fwhm = float(np.random.uniform(*inst_psf_fwhm_range))
    psf_type = np.random.choice(PSF_TYPES)
    psf_ellip = float(np.random.uniform(0.0, PSF_ELLIP_MAX))
    psf_angle = float(np.random.uniform(0, 2*np.pi))
    psf_beta = float(np.random.uniform(*PSF_MOFFAT_BETA_RANGE))
    read_noise = float(np.random.uniform(*inst_read_noise_range))
    sky_adu = float(np.random.uniform(*inst_sky_range))

    # pick magnitudes
    src_mag = float(np.random.uniform(*SRC_MAG_RANGE))
    lens_mag = src_mag + float(np.random.uniform(*LENS_MAG_DELTA_RANGE))


    sb_src_sup, src_total_counts = normalize_to_sb_per_arcsec2(src_pattern, src_mag, zp, exptime, SUP_PIXEL_SCALE)
    src_counts_sup = (sb_src_sup * SUPER_PIX_AREA).astype(np.float32)


    GT = downsample(src_counts_sup, OVERSAMPLE).astype(np.float32)

    # Ray-shooting mapping for lensed source
    x_flat, y_flat = x_sup.ravel(), y_sup.ravel()
    xs, ys = lens.ray_shooting(x_flat, y_flat, kwargs_lens)
    xs = xs.reshape(SUPER_SIZE, SUPER_SIZE)
    ys = ys.reshape(SUPER_SIZE, SUPER_SIZE)


    interp_sb = rg_interpolator(x_sup, y_sup, sb_src_sup, method='linear')
    pts = np.vstack([ys.ravel(), xs.ravel()]).T
    sb_mapped = interp_sb(pts).reshape(SUPER_SIZE, SUPER_SIZE)
    lensed_src_counts_sup = (sb_mapped * SUPER_PIX_AREA).astype(np.float32)

    # lens light (on image plane, not lensed)
    lens_light_model = LightModel(['SERSIC_ELLIPSE'])
    lens_light_sup_pattern = lens_light_model.surface_brightness(x_sup, y_sup, [ {
        'amp': 1.0,
        'R_sersic': np.random.uniform(0.25, 1.2),
        'n_sersic': np.random.uniform(2.5, 5.0),
        'e1': -0.6 * e1, 'e2': -0.6 * e2,
        'center_x': kwargs_lens[0]['center_x'] + np.random.uniform(-0.02, 0.02),
        'center_y': kwargs_lens[0]['center_y'] + np.random.uniform(-0.02, 0.02)
    } ]).astype(np.float32)
    sb_lens_sup, lens_total_counts = normalize_to_sb_per_arcsec2(lens_light_sup_pattern, lens_mag, zp, exptime, SUP_PIXEL_SCALE)
    lens_counts_sup = (sb_lens_sup * SUPER_PIX_AREA).astype(np.float32)

    # combine on super-grid
    image_sup = lensed_src_counts_sup + lens_counts_sup

    # create "true" PSF and optionally a slightly mismatched "measured" PSF
    true_psf_sup = make_psf_kernel(psf_fwhm, SUP_PIXEL_SCALE, psf_type, ellip=psf_ellip, angle=psf_angle, beta=psf_beta)
    if np.random.rand() < ADD_PSF_MISMATCH_PROB:
        meas_fwhm = psf_fwhm * np.random.uniform(0.9, 1.30)
        meas_ellip = min(PSF_ELLIP_MAX, max(0.0, psf_ellip + np.random.normal(0.0, 0.06)))
        meas_angle = psf_angle + np.random.uniform(-0.4, 0.4)
        measured_psf_sup = make_psf_kernel(meas_fwhm, SUP_PIXEL_SCALE, psf_type, ellip=meas_ellip, angle=meas_angle, beta=psf_beta)
    else:
        measured_psf_sup = true_psf_sup.copy()
        meas_fwhm = psf_fwhm

    # convolve with true PSF on super-grid
    image_conv_sup = convolve_fft(image_sup, true_psf_sup, normalize_kernel=True, allow_huge=True)

    # sub-pixel shift (simulate dithering / misalignment)
    if np.random.rand() < ADD_SUBPIXEL_SHIFT_PROB:
        # shift in super-pixel units (fractional)
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        image_conv_sup = nd_shift(image_conv_sup, shift=(dy, dx), order=3, mode='reflect')

    # downsample to detector sampling
    LENSED_counts = downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)

    # check arc vs lens peaks - soft stochastic adjustment (do NOT force unrealistic arcs)
    def peak99(arr):
        return float(np.percentile(arr.flatten(), 99))
    peak_src = peak99(downsample(convolve_fft(lensed_src_counts_sup, true_psf_sup, normalize_kernel=True), OVERSAMPLE))
    peak_lens = peak99(downsample(convolve_fft(lens_counts_sup, true_psf_sup, normalize_kernel=True), OVERSAMPLE))
    ratio = (peak_src + 1e-9) / (peak_lens + 1e-9)

    if np.random.rand() < 0.20 and ratio < 0.25:
        # sometimes enhance source mildly so arcs remain visible in some subsets
        LENSED_counts *= 0  # rebuild
        lens_counts_sup *= np.random.uniform(0.8, 1.0)
        lensed_src_counts_sup *= np.random.uniform(1.05, 1.6)
        image_sup = lensed_src_counts_sup + lens_counts_sup
        image_conv_sup = convolve_fft(image_sup, true_psf_sup, normalize_kernel=True, allow_huge=True)
        if np.random.rand() < ADD_SUBPIXEL_SHIFT_PROB:
            dx = np.random.uniform(-0.5, 0.5)
            dy = np.random.uniform(-0.5, 0.5)
            image_conv_sup = nd_shift(image_conv_sup, shift=(dy, dx), order=3, mode='reflect')
        LENSED_counts = downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)
        peak_src = peak99(downsample(convolve_fft(lensed_src_counts_sup, true_psf_sup, normalize_kernel=True), OVERSAMPLE))
        peak_lens = peak99(downsample(convolve_fft(lens_counts_sup, true_psf_sup, normalize_kernel=True), OVERSAMPLE))
        ratio = (peak_src + 1e-9) / (peak_lens + 1e-9)

    # Add sky baseline + small sky gradient (on detector grid)
    xg = np.linspace(-0.5, 0.5, IMAGE_SIZE)
    sky_gradient = (1.0 + SKY_GRADIENT_MAX * (xg - xg.mean()) / (xg.max() - xg.min()))
    sky_map = sky_adu * sky_gradient[np.newaxis, :]
    LENSED_counts = LENSED_counts + sky_map

    # PRNU (multiplicative pattern) applied before Poisson
    prnu_map = 1.0 + np.random.normal(0.0, PRNU_RMS, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    LENSED_counts_prnu = LENSED_counts * prnu_map


    if np.random.rand() < ADD_CORRELATED_NOISE_PROB:
        sigma_corr = np.random.uniform(*CORRELATED_NOISE_SIGMA_RANGE)
        kern_size = int(max(5, np.ceil(6 * sigma_corr)))
        if kern_size % 2 == 0:
            kern_size += 1
        yk, xk = np.mgrid[:kern_size, :kern_size] - kern_size//2
        gkern = np.exp(-0.5*(xk**2 + yk**2)/sigma_corr**2)
        gkern /= gkern.sum()
        white = np.random.normal(0, 1.0, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        corr = fftconvolve(white, gkern, mode='same')
        corr *= np.random.uniform(0.002, 0.03) * max(1.0, np.median(LENSED_counts_prnu + 1e-12))
        LENSED_counts_prnu += corr

    # sky-subtraction residual (low-order tilt) optional
    if np.random.rand() < ADD_SKY_SUBTRACTION_RESIDUALS_PROB:
        ax = np.linspace(-1, 1, IMAGE_SIZE)
        bx = np.random.uniform(-0.003, 0.003)
        by = np.random.uniform(-0.003, 0.003)
        sky_resid = bx * ax[np.newaxis, :] + by * ax[:, np.newaxis]
        LENSED_counts_prnu += sky_resid * np.median(LENSED_counts_prnu + 1e-12)

    # convert ADU -> electrons, add Poisson noise, add read noise
    e_image = np.clip(LENSED_counts_prnu * GAIN, 0.0, None)
    # Poisson can be expensive for large arrays; but we need realistic stats
    noisy_e = np.random.poisson(e_image).astype(np.float32)
    noisy_e += np.random.normal(0.0, read_noise, noisy_e.shape).astype(np.float32)

    # cosmic rays
    if np.random.rand() < COSMIC_RAY_PROB:
        n_hits = np.random.randint(1, 6)
        for _ in range(n_hits):
            cx = np.random.randint(0, IMAGE_SIZE)
            cy = np.random.randint(0, IMAGE_SIZE)
            intensity = np.random.uniform(*COSMIC_RAY_INTENSITY)
            noisy_e[cy, cx] += intensity * GAIN

    LENSED = (noisy_e / GAIN).astype(np.float32)


    filename = os.path.join(outdir, f"rim_simv3_{i:05d}.fits")
    hdu_primary = fits.PrimaryHDU(data=GT.astype(np.float32))
    hdu_lensed = fits.ImageHDU(data=LENSED.astype(np.float32), name='LENSED')

    # Primary header: include metadata (we keep GT-related keys here)
    phdr = hdu_primary.header
    phdr['DATE'] = (datetime.utcnow().isoformat(), 'UTC creation time')
    phdr['PIXSCALE'] = (PIXEL_SCALE_THIS, 'arcsec/pixel')
    phdr['BUNIT'] = ('ADU', 'Brightness unit')
    phdr['OVERSAMP'] = (OVERSAMPLE, 'super-sampling factor')
    phdr['ZP'] = (DEFAULT_ZP, 'AB zeropoint (ADU/s)')
    phdr['EXPTIME'] = (exptime, 'seconds')
    phdr['GAIN'] = (GAIN, 'e-/ADU')
    phdr['RN_E'] = (read_noise, 'read noise e- RMS')
    phdr['SKYADU'] = (sky_adu, 'median sky level (ADU/pixel)')
    phdr['PSF_FWH_TRUE'] = (psf_fwhm, 'simulated true PSF FWHM (arcsec)')
    phdr['PSF_FWH_MEAS'] = (meas_fwhm, 'measured (mismatched) PSF FWHM (arcsec)')
    phdr['PSF_TYP'] = (psf_type, 'PSF model type')
    phdr['THETA_E'] = (theta_E, 'Einstein radius (arcsec)')
    phdr['SIGMA'] = (sigma_kms, 'velocity dispersion km/s used to set thetaE')
    phdr['ELLIP'] = (e, 'SIE ellipticity')
    phdr['SHEAR'] = (shear_g, 'external shear amplitude')
    phdr['SRCMAG'] = (src_mag, 'source integrated AB mag')
    phdr['LENSMAG'] = (lens_mag, 'lens integrated AB mag')
    phdr['ZLENS'] = (z_l, 'lens redshift')
    phdr['ZSRC'] = (z_s, 'source redshift')
    phdr['PRNU'] = (PRNU_RMS, 'PRNU RMS (fractional)')
    phdr['SKYGRAD'] = (SKY_GRADIENT_MAX, 'max fractional sky gradient')
    # instrument descriptor
    phdr['INSTRUM'] = ('SPACE' if PIXEL_SCALE_THIS <= 0.05 else 'GROUND', 'instrument regime')

    # Lensed header entries (mirror a few keys in image HDU too)
    for k, v in [('PIXSCALE', PIXEL_SCALE_THIS), ('ZP', DEFAULT_ZP), ('EXPTIME', exptime), ('PSF_FWH', psf_fwhm), ('PSF_TYP', psf_type)]:
        hdu_lensed.header[k] = v

    hdul = fits.HDUList([hdu_primary, hdu_lensed])
    hdul.writeto(filename, overwrite=True)

    print(f"[{i}] Saved {filename}  | peak(src/lens) ratio≈{ratio:.3f}  | instr_pixel_scale={PIXEL_SCALE_THIS:.3f}\"  | psf_fwhm_true={psf_fwhm:.3f}\"")

    return filename

def main():
    set_seed(SEED)
    for i in range(NUM_REALIZATIONS):
        generate_one(i, OUTPUT_DIR)

if __name__ == '__main__':
    main()
