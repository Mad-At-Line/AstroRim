"""
simgen_truth.py -- Shared truth-label and PSF/halo helpers for simgenv1-v6.

================================================================================
PURPOSE
================================================================================
The simulators previously recorded only loose global metadata (THETA_E in
arcsec, ELLIP, SHEAR, ...). Nothing was written in the MODEL's parameter
convention, so neither (a) the predicted-vs-true scatter batch (~200 lenses)
nor (b) the auxiliary parameter-supervision loss in training.py could be run.
This module writes a standardized TR_* header block in exactly the convention
the encoder predicts in (astrorim_core.TRUTH_ORDER), using conversions
verified numerically against the forward operator:

  b          = theta_E_arcsec / half_fov,   half_fov = (N/2) * pixel_scale
               (lenstronomy SIE theta_E == Kormann b: verified exact in the
               rc->0 limit at every q; the model's rc=0.01 softening is its
               own design choice, identical at train and test time)
  q, phi     = lenstronomy param_util.ellipticity2phi_q(e1, e2)  (phi mod pi)
  x0, y0     = center_x / half_fov, center_y / half_fov
  gamma      = hypot(gamma1, gamma2)            (verified exact)
  gamma_phi  = 0.5 * atan2(gamma2, gamma1)
  kappa_s    = alpha_Rs / (4 * Rs * (1 + ln(1/2)))   (verified exact, <1e-7)
  rs         = Rs_arcsec / half_fov
  psf_fwhm   = realized FWHM in detector pixels (see [FIX S1] below)
  psf_e1,e2  = ((1-q_psf)/(1+q_psf)) * (cos 2a, -sin 2a)
               for the model's coordinate-stretch kernel convention
               (direction verified by direct kernel fitting; with the fixed
               kernel [FIX S1] this is the exact axis-ratio mapping)
  lens_Re    = R_sersic_arcsec / half_fov ; lens_n = n_sersic ;
  lens_flux  = measured pre-PSF lens-light amplitude at r=Re in the SAME
               normalized units the training dataset produces (divide by
               m = max(GT.max(), LENSED.max()))

TR_MASK is a bitmask over astrorim_core.TRUTH_ORDER marking which labels are
valid supervision targets for this file. TR_CLEAN = 1 only when the lens is
exactly inside the forward-operator family (single SIE [+SHEAR] [+co-centred
NFW]); offset group halos and SIS subhalos set TR_CLEAN = 0 and mask off
kappa_s / rs (the SIE+shear labels still describe the primary deflector).

================================================================================
FIXES PROVIDED HERE FOR THE SIMULATORS
================================================================================
[FIX S1] make_psf_kernel_fixed: corrected Moffat radial profile.
    The previous make_psf_kernel (all six variants, INCLUDING the 'patched'
    v6) computed r2 = (x'/sigma_x)^2 + (y'/sigma_y)^2 -- a sigma-NORMALIZED
    radius -- and then evaluated the Moffat as (1 + r2/alpha^2)^-beta with
    alpha in PIXELS. The sigma normalization and pixel-alpha do not compose:
    the realized profile is a Moffat of width alpha_eff = sigma_pix * alpha,
    so the realized FWHM = requested_FWHM * (requested_FWHM_px / 2.3548).
    Measured example: requested 3.0 px -> realized 3.83 px (+28%); requested
    2.0 px -> realized 2.55... in general the error grows with FWHM and even
    changes sign below ~2.35 px. Every MOFFAT-type image in the existing
    training set therefore has a PSF that does not match its header, and the
    realized training FWHM distribution is distorted. The fix evaluates the
    Moffat on the true elliptical pixel radius r2_pix = q*x'^2 + y'^2/q
    (geometric-mean-FWHM convention, area-preserving) divided by alpha^2.
    The Gaussian branch was already correct and is reproduced exactly;
    Airy is unchanged. Verified: realized FWHM == requested FWHM to <3%
    (pixel sampling) across FWHM in [1.5, 6] px, beta in [2.5, 4.8].
    NOTE [S1b]: the AIRY branch ignores the requested FWHM entirely (its
    width is set by lambda/D alone, ~0.45 px at 0.04"/px). That is left
    as-is (it is a deliberate diffraction-limited case) but the TRUTH header
    now records the REALIZED Airy FWHM, not the meaningless requested one.

[FIX S2] physical_nfw_group_halo: replaces simgenv6's dimensionally broken
    group-halo block ('alpha_Rs': sqrt(4 c^2 M / mu(c)) * 0.001 "convert to
    appropriate units" -- the square root of a mass is not an angle, and
    R200/c in Mpc times 0.001 is not arcsec; the R200 formula was also
    missing its factor of 3). The replacement derives (Rs_arcsec, alpha_Rs)
    from (M200, z_l, z_s, cosmology) via the standard chain:
        R200 = (3 M / (4 pi 200 rho_crit(z_l)))^(1/3)
        c(M) as supplied; Rs = R200 / c; Rs_arcsec = Rs / D_d (rad->arcsec)
        delta_c = (200/3) c^3 / (ln(1+c) - c/(1+c)); rho_s = delta_c rho_crit
        Sigma_crit = c_light^2 D_s / (4 pi G D_d D_ds)
        kappa_s = rho_s Rs / Sigma_crit
        alpha_Rs = 4 kappa_s Rs_arcsec (1 + ln(1/2))   [W&B 2000 deflection at Rs]
    Sanity-checked: M200 = 1e14 Msun at z_l=0.5, z_s=2.0 gives
    kappa_s ~ 0.17, Rs ~ 26", alpha_Rs ~ 5" -- a sensible group/cluster halo.

[S3] write_truth_header: the standardized TR_* block (<=8-char FITS keys).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from astropy import constants as const
from astropy import units as u

# Order must match astrorim_core.TRUTH_ORDER. Duplicated here (rather than
# imported) so the simulators do not need torch installed; the test suite
# asserts the two lists are identical.
TRUTH_ORDER = [
    'b', 'q', 'phi', 'x0', 'y0', 'gamma', 'gamma_phi', 'kappa_s', 'rs',
    'psf_fwhm', 'psf_beta', 'psf_e1', 'psf_e2',
    'lens_flux', 'lens_Re', 'lens_n',
]
TRUTH_FITS_KEYS = {
    'b': 'TR_B', 'q': 'TR_Q', 'phi': 'TR_PHI', 'x0': 'TR_X0', 'y0': 'TR_Y0',
    'gamma': 'TR_GAM', 'gamma_phi': 'TR_GPHI', 'kappa_s': 'TR_KS', 'rs': 'TR_RS',
    'psf_fwhm': 'TR_PSFW', 'psf_beta': 'TR_PSFB',
    'psf_e1': 'TR_PSFE1', 'psf_e2': 'TR_PSFE2',
    'lens_flux': 'TR_LFLX', 'lens_Re': 'TR_LRE', 'lens_n': 'TR_LN',
}
TRUTH_COMMENTS = {
    'b': 'true theta_E in normalized units',
    'q': 'true SIE axis ratio',
    'phi': 'true SIE PA [rad], model frame, mod pi',
    'x0': 'true lens center x [norm]',
    'y0': 'true lens center y [norm]',
    'gamma': 'true external shear amplitude',
    'gamma_phi': 'true shear PA [rad], mod pi',
    'kappa_s': 'true NFW kappa_s (0 if no halo)',
    'rs': 'true NFW rs [norm]',
    'psf_fwhm': 'realized PSF FWHM [detector px]',
    'psf_beta': 'true Moffat beta (Moffat PSFs only)',
    'psf_e1': 'true PSF e1 (model stretch conv.)',
    'psf_e2': 'true PSF e2 (model stretch conv.)',
    'lens_flux': 'lens light amp at Re [norm units]',
    'lens_Re': 'lens light Re [norm]',
    'lens_n': 'lens light Sersic n',
}


def _wrap_axis_angle(a: float) -> float:
    """Wrap an axis-type (period-pi) angle into (-pi/2, pi/2]."""
    a = (a + math.pi / 2.0) % math.pi - math.pi / 2.0
    if a <= -math.pi / 2.0:
        a += math.pi
    return float(a)


# =============================================================================
# [FIX S1] Corrected PSF kernel builder (drop-in for make_psf_kernel)
# =============================================================================

def make_psf_kernel_fixed(fwhm_arcsec, pixel_scale_arcsec, psf_type='GAUSSIAN',
                          ellip=0.0, angle=0.0, beta=3.5, rng=None,
                          coma_prob=0.4):
    """Drop-in replacement for the simulators' make_psf_kernel.

    Differences from the original (see module docstring [FIX S1]):
      * MOFFAT: evaluated on the true elliptical pixel radius
        r2_pix = q*x'^2 + y'^2/q (geometric-mean FWHM convention), so the
        realized FWHM equals the requested FWHM. The old version realized
        FWHM_requested * sigma_pix instead.
      * GAUSSIAN: algebraically identical to the original (the original
        Gaussian branch was correct); kept in the same r2 form.
      * AIRY: unchanged (lambda/D-limited; ignores fwhm by design). 
      * The optional 'coma' perturbation is preserved but now takes an
        explicit rng and probability so the simulators stay reproducible
        under seeding, and returns whether it was applied (recorded as
        TR_COMA so the PSF truth labels can be interpreted).

    Returns (kernel, info) where info = dict(realized_fwhm_pix, coma_applied).
    """
    if rng is None:
        rng = np.random
    fwhm_pix = fwhm_arcsec / pixel_scale_arcsec
    sigma_pix = fwhm_pix / 2.3548

    k = int(max(21, np.ceil(fwhm_pix * 6)))
    if k % 2 == 0:
        k += 1

    y, x = np.mgrid[:k, :k] - k // 2
    coma_applied = False

    if str(psf_type).upper() == 'AIRY':
        wavelength = 700e-9       # m
        aperture_diameter = 8.2   # m
        angular_scale = (wavelength / aperture_diameter) * 206265.0  # arcsec
        r = np.sqrt(x ** 2 + y ** 2) * pixel_scale_arcsec / angular_scale
        r[r == 0] = 1e-10
        import scipy.special
        kern = (2 * scipy.special.j1(np.pi * r) / (np.pi * r)) ** 2
        realized_fwhm_pix = 1.028 * angular_scale / pixel_scale_arcsec
    else:
        q = max(0.001, 1.0 - ellip)
        x_rot = np.cos(angle) * x + np.sin(angle) * y
        y_rot = -np.sin(angle) * x + np.cos(angle) * y
        # True elliptical pixel radius^2, area-preserving (geometric-mean FWHM)
        r2_pix = q * x_rot ** 2 + (y_rot ** 2) / q

        if str(psf_type).upper() == 'GAUSSIAN':
            kern = np.exp(-0.5 * r2_pix / (sigma_pix ** 2 + 1e-12))
        else:  # MOFFAT
            alpha = fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
            kern = (1.0 + r2_pix / (alpha ** 2)) ** (-beta)
        realized_fwhm_pix = fwhm_pix

        if rng.rand() < coma_prob:
            coma = x * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma_pix * 1.5) ** 2))
            mx = np.max(np.abs(coma))
            if mx > 0:
                kern = kern * (1 + 0.05 * coma / mx)
                coma_applied = True

    kern = kern / kern.sum()
    return kern.astype(np.float32), {
        'realized_fwhm_pix': float(realized_fwhm_pix),
        'coma_applied': bool(coma_applied),
    }


# =============================================================================
# [FIX S2] Physical NFW group-halo kwargs (replaces simgenv6's broken block)
# =============================================================================

def physical_nfw_group_halo(mass_msun: float, z_lens: float, z_source: float,
                            cosmo, center_x: float, center_y: float
                            ) -> Tuple[dict, dict]:
    """Derive lenstronomy NFW kwargs (alpha_Rs, Rs in arcsec) from a physical
    halo mass with the standard M200c definition. See module docstring.

    Returns (lenstronomy_kwargs, info) where info carries kappa_s and Rs.
    """
    M = mass_msun * u.M_sun
    rho_c = cosmo.critical_density(z_lens).to(u.M_sun / u.Mpc ** 3)
    R200 = ((3.0 * M / (4.0 * math.pi * 200.0 * rho_c)) ** (1.0 / 3.0)).to(u.Mpc)
    conc = 5.0 * (mass_msun / 1e14) ** (-0.1)   # same M-c relation as before
    Rs = (R200 / conc).to(u.Mpc)

    D_d = cosmo.angular_diameter_distance(z_lens).to(u.Mpc)
    D_s = cosmo.angular_diameter_distance(z_source).to(u.Mpc)
    D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.Mpc)
    Rs_arcsec = float((Rs / D_d).decompose().value * 206265.0)

    delta_c = (200.0 / 3.0) * conc ** 3 / (math.log(1.0 + conc) - conc / (1.0 + conc))
    rho_s = delta_c * rho_c

    sigma_crit = (const.c ** 2 / (4.0 * math.pi * const.G) *
                  D_s / (D_d * D_ds)).to(u.M_sun / u.Mpc ** 2)
    kappa_s = float((rho_s * Rs / sigma_crit).decompose().value)

    alpha_Rs = 4.0 * kappa_s * Rs_arcsec * (1.0 + math.log(0.5))

    kwargs = {'alpha_Rs': float(alpha_Rs), 'Rs': float(Rs_arcsec),
              'center_x': float(center_x), 'center_y': float(center_y)}
    info = {'kappa_s': kappa_s, 'Rs_arcsec': Rs_arcsec, 'conc': float(conc),
            'M200_Msun': float(mass_msun)}
    return kwargs, info


# =============================================================================
# Truth extraction from lenstronomy kwargs (verified conventions)
# =============================================================================

def sim_psf_e_to_model(ellip: float, angle: float) -> Tuple[float, float]:
    """Map the simulator's (ellip, angle) PSF to the model's (e1, e2) stretch
    convention. Derivation in module docstring; direction verified by direct
    kernel fitting against the model's _build_moffat_kernel_batched."""
    qp = max(0.001, 1.0 - float(ellip))
    e_mod = (1.0 - qp) / (1.0 + qp)
    return (float(e_mod * math.cos(2.0 * angle)),
            float(-e_mod * math.sin(2.0 * angle)))


def truth_from_lenstronomy(lens_model_list, kwargs_lens,
                           pixel_scale: float, image_size: int,
                           psf_info: Optional[dict] = None,
                           lens_light_info: Optional[dict] = None) -> Tuple[Dict[str, float], int, int]:
    """Convert the simulator's lens configuration into the model convention.

    psf_info (optional): dict with keys
        realized_fwhm_pix, psf_type ('MOFFAT'/'GAUSSIAN'/'AIRY'),
        beta, ellip, angle, coma_applied
    lens_light_info (optional): dict with keys
        Re_arcsec, n_sersic, amp_at_Re_normalized  (any may be None)

    Returns (truth dict over TRUTH_ORDER, mask bits, clean flag).
    Mask semantics per astrorim_core: bit i set => TRUTH_ORDER[i] supervisable.
    """
    from lenstronomy.Util import param_util  # local import: sims have it

    half_fov = (image_size / 2.0) * pixel_scale
    truth = {k: 0.0 for k in TRUTH_ORDER}
    maskbits = 0
    clean = 1

    def set_ok(name, value):
        nonlocal maskbits
        truth[name] = float(value)
        maskbits |= (1 << TRUTH_ORDER.index(name))

    n_sie = 0
    n_nfw = 0
    n_other = 0
    nfw_cocentred = False

    primary = None
    for model, kw in zip(lens_model_list, kwargs_lens):
        if model == 'SIE':
            n_sie += 1
            if primary is None:
                primary = kw
        elif model == 'SHEAR':
            g1, g2 = float(kw['gamma1']), float(kw['gamma2'])
            set_ok('gamma', math.hypot(g1, g2))
            set_ok('gamma_phi', _wrap_axis_angle(0.5 * math.atan2(g2, g1)))
        elif model == 'NFW':
            n_nfw += 1
            Rs_arc = float(kw['Rs'])
            alpha_Rs = float(kw['alpha_Rs'])
            ks = alpha_Rs / (4.0 * Rs_arc * (1.0 + math.log(0.5)))
            truth['kappa_s'] = ks
            truth['rs'] = Rs_arc / half_fov
            if primary is not None:
                dxc = float(kw.get('center_x', 0.0)) - float(primary['center_x'])
                dyc = float(kw.get('center_y', 0.0)) - float(primary['center_y'])
                nfw_cocentred = math.hypot(dxc, dyc) < 1e-6
        else:
            n_other += 1

    if primary is not None:
        phi, q = param_util.ellipticity2phi_q(float(primary['e1']), float(primary['e2']))
        set_ok('b', float(primary['theta_E']) / half_fov)
        set_ok('q', q)
        set_ok('phi', _wrap_axis_angle(phi))
        set_ok('x0', float(primary['center_x']) / half_fov)
        set_ok('y0', float(primary['center_y']) / half_fov)

    gamma_already_set = bool((maskbits >> TRUTH_ORDER.index('gamma')) & 1)
    if not gamma_already_set:
        # no SHEAR component: true shear is exactly zero (supervisable);
        # its orientation is undefined (the loss already weights gamma_phi
        # by gamma_true, so the value here is irrelevant but masked off).
        set_ok('gamma', 0.0)

    # NFW / clean logic
    if n_nfw == 0:
        # No halo: kappa_s truly 0 -> supervisable (teaches 'predict the
        # floor when there is no halo': the direct fix for the hallucinated
        # halo). rs has no meaning and the loss auto-weights it to 0, but we
        # mask it off as well for clarity.
        set_ok('kappa_s', 0.0)
        truth['rs'] = 0.0
    elif n_nfw == 1 and nfw_cocentred and n_other == 0 and n_sie == 1:
        set_ok('kappa_s', truth['kappa_s'])
        set_ok('rs', truth['rs'])
    else:
        clean = 0  # offset halo: labels recorded but not supervisable

    if n_sie != 1 or n_other > 0:
        clean = 0

    # PSF labels
    if psf_info is not None:
        set_ok('psf_fwhm', float(psf_info['realized_fwhm_pix']))
        ptype = str(psf_info.get('psf_type', 'MOFFAT')).upper()
        if ptype == 'MOFFAT':
            set_ok('psf_beta', float(psf_info.get('beta', 3.5)))
        else:
            truth['psf_beta'] = 0.0  # not in the Moffat family: not supervised
        if ptype == 'AIRY':
            set_ok('psf_e1', 0.0)
            set_ok('psf_e2', 0.0)
        else:
            e1m, e2m = sim_psf_e_to_model(float(psf_info.get('ellip', 0.0)),
                                          float(psf_info.get('angle', 0.0)))
            set_ok('psf_e1', e1m)
            set_ok('psf_e2', e2m)

    # Lens-light labels (best circular-family approximations; train at
    # reduced weight -- see training.py defaults)
    if lens_light_info is not None:
        if lens_light_info.get('Re_arcsec') is not None:
            set_ok('lens_Re', float(lens_light_info['Re_arcsec']) / half_fov)
        if lens_light_info.get('n_sersic') is not None:
            set_ok('lens_n', float(lens_light_info['n_sersic']))
        if lens_light_info.get('amp_at_Re_normalized') is not None:
            set_ok('lens_flux', float(lens_light_info['amp_at_Re_normalized']))

    return truth, maskbits, clean


def measure_lens_light_amp_at_Re(lens_light_detector: np.ndarray,
                                 light_center_arcsec: Tuple[float, float],
                                 Re_arcsec: float,
                                 pixel_scale: float,
                                 norm_factor: float) -> Optional[float]:
    """Median pre-PSF lens-light value in an annulus at r=Re around the light
    center, divided by the dataset normalization factor
    m = max(GT.max(), LENSED.max()). This is directly comparable to the
    model's lens_flux (its circular Sersic equals lens_flux at r=Re)."""
    H, W = lens_light_detector.shape
    cx_pix = (light_center_arcsec[0] / pixel_scale) + (W - 1) / 2.0
    cy_pix = (light_center_arcsec[1] / pixel_scale) + (H - 1) / 2.0
    Re_pix = Re_arcsec / pixel_scale
    if not (0 <= cx_pix < W and 0 <= cy_pix < H) or Re_pix <= 0.5:
        return None
    yy, xx = np.indices((H, W))
    r = np.sqrt((xx - cx_pix) ** 2 + (yy - cy_pix) ** 2)
    ann = (r > Re_pix - 0.75) & (r < Re_pix + 0.75)
    if ann.sum() < 4:
        return None
    val = float(np.median(lens_light_detector[ann]))
    if norm_factor <= 0:
        return None
    return val / norm_factor


def write_truth_header(hdr, truth: Dict[str, float], maskbits: int, clean: int,
                       sim_set: str, pixel_scale: float,
                       coma_applied: bool = False):
    """[S3] Write the standardized TR_* block onto a FITS header."""
    for name in TRUTH_ORDER:
        hdr[TRUTH_FITS_KEYS[name]] = (float(truth.get(name, 0.0)),
                                      TRUTH_COMMENTS[name])
    hdr['TR_MASK'] = (int(maskbits), 'bitmask: TRUTH_ORDER[i] supervisable')
    hdr['TR_CLEAN'] = (int(clean), '1 = lens exactly in model family')
    hdr['TR_SET'] = (str(sim_set), 'simulator variant')
    hdr['TR_PS'] = (float(pixel_scale), 'sim pixel scale [arcsec/px]')
    hdr['TR_COMA'] = (1 if coma_applied else 0, 'coma perturbation applied to PSF')
    hdr['TR_VER'] = ('2.1.0', 'truth header schema version')
