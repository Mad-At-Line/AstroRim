
import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from astropy.io import fits
from scipy.ndimage import (
    center_of_mass, shift as nd_shift, median_filter, gaussian_filter
)
from skimage.transform import resize
from skimage.filters import sobel
from skimage.feature import peak_local_max

DEFAULT_INPUT_DIR = r"C:\Users\mythi\AstroRIM\Castles_raw\Castles\RXJ1131\FIT"
DEFAULT_OUTPUT_DIR = r"C:\Users\mythi\AstroRIM\AstroRIM_2.1\Test_outputs\rxj1131\Normalized"
TARGET_SIZE = 96

# Sky and background

def estimate_sky(data, method='edge_median'):
    """Estimate the sky background level. Returns a scalar.

    Methods:
      edge_median - median of pixels on the outer border (robust if FOV > lens).
      percentile  - 10th percentile of all pixels (robust if most of the image is sky).
      sigma_clip  - iterative sigma-clipped median.
    """
    if method == 'edge_median':
        edges = np.concatenate([data[0, :], data[-1, :], data[:, 0], data[:, -1]])
        return float(np.median(edges))
    elif method == 'percentile':
        return float(np.percentile(data, 10))
    elif method == 'sigma_clip':
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        std = 1.4826 * mad if mad > 0 else np.std(data)
        mask = np.abs(data - med) < 3.0 * std
        if mask.sum() > 0:
            return float(np.median(data[mask]))
        return float(med)
    return 0.0

def gentle_denoise(data, method='none', **params):
    """Optional very gentle denoising. Default is 'none' because the forward model
    has its own noise handling via Poisson + Gaussian likelihood.

    Use 'median3' for a single pass of 3x3 median (removes isolated hot pixels only).
    Use 'gauss05' for sigma=0.5 Gaussian (sub-pixel smoothing, use sparingly).
    """
    if method == 'none':
        return data
    elif method == 'median3':
        return median_filter(data, size=3)
    elif method == 'gauss05':
        return gaussian_filter(data, sigma=0.5)
    return data

# Centering utilities

def find_lens_center(data, method='brightness_weighted', **params):
    """Find the centroid of the lens in input-image coordinates.

    Returns (cy, cx). Falls back to the geometric center if detection fails.
    """
    h, w = data.shape

    if method == 'peak':
        try:
            # brightest pixel of smoothed image (robust to single-pixel noise)
            sm = gaussian_filter(data, sigma=1.0)
            cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
            return float(cy), float(cx)
        except Exception:
            return h / 2.0, w / 2.0

    if method == 'ring_detection':
        # Detect ring-like structures using edge response; return edge-weighted centroid.
        try:
            edges = sobel(data)
            edges = np.clip(edges, 0, None)
            if edges.sum() > 0:
                cy, cx = center_of_mass(edges)
                if np.isfinite(cy) and np.isfinite(cx):
                    return float(cy), float(cx)
        except Exception:
            pass
        return h / 2.0, w / 2.0

    # default: brightness_weighted (robust, matches v1 default)
    try:
        # Subtract a floor to avoid centering on a global offset
        floor = np.percentile(data, 50)
        d = np.clip(data - floor, 0, None)
        if d.sum() > 0:
            cy, cx = center_of_mass(d)
            if np.isfinite(cy) and np.isfinite(cx):
                return float(cy), float(cx)
    except Exception:
        pass
    return h / 2.0, w / 2.0

def resize_and_pad(data, target_size):
    """Downscale proportionally so the larger dimension fits, then zero-pad to square.
    Preserves the full field of view - never crops."""
    h, w = data.shape
    if h == target_size and w == target_size:
        return data.astype(np.float32)

    scale = min(target_size / h, target_size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Preserve linear range during resize (anti_aliasing=True uses Gaussian pre-filter,
    # which slightly blurs but preserves total flux locally; acceptable for lens inference).
    resized = resize(data, (new_h, new_w),
                     anti_aliasing=True, preserve_range=True, order=1).astype(np.float32)

    padded = np.zeros((target_size, target_size), dtype=np.float32)
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    padded[y0:y0 + new_h, x0:x0 + new_w] = resized
    return padded


def extract_pixel_scale(header):
    """[NEW N1] Input pixel scale in arcsec/px from common header conventions.
    Returns (value, source_string) or (None, None). Priority: explicit
    PIXSCALE/PIXSCAL1 keywords, then the WCS CD matrix (rotation-safe row
    norms), then CDELT."""
    if header is None:
        return None, None
    for key in ('PIXSCALE', 'PIXSCAL1'):
        if key in header:
            try:
                v = float(header[key])
                if np.isfinite(v) and 1e-4 < v < 30.0:
                    return v, key
            except Exception:
                pass
    try:
        if 'CD1_1' in header:
            cd11 = float(header['CD1_1'])
            cd21 = float(header.get('CD2_1', 0.0))
            cd12 = float(header.get('CD1_2', 0.0))
            cd22 = float(header.get('CD2_2', cd11))
            sx = 3600.0 * (cd11 ** 2 + cd21 ** 2) ** 0.5
            sy = 3600.0 * (cd12 ** 2 + cd22 ** 2) ** 0.5
            if sx > 0 and sy > 0:
                v = float((sx * sy) ** 0.5)
                if np.isfinite(v) and 1e-4 < v < 30.0:
                    return v, 'CD_matrix'
    except Exception:
        pass
    for key in ('CDELT1',):
        if key in header:
            try:
                v = abs(float(header[key])) * 3600.0
                if np.isfinite(v) and 1e-4 < v < 30.0:
                    return v, key + '*3600'
            except Exception:
                pass
    return None, None


def resample_scale_factor(original_shape, target_size):
    """[NEW N1] The isotropic factor applied by resize_and_pad (padding does
    not change the pixel scale). ps_new = ps_orig / scale."""
    h, w = original_shape
    if h == target_size and w == target_size:
        return 1.0
    return min(target_size / h, target_size / w)


def center_and_resize(data, target_size=96, centering_method='brightness_weighted', verbose=False):
    """Shift the lens to the image center, then downscale and pad to target_size."""
    h, w = data.shape
    cy, cx = find_lens_center(data, method=centering_method)

    if verbose:
        print(f"   Detected lens center: ({cy:.1f}, {cx:.1f}); image center: ({h / 2:.1f}, {w / 2:.1f})")

    shift_y = h / 2.0 - cy
    shift_x = w / 2.0 - cx
    shifted = nd_shift(data, shift=(shift_y, shift_x), order=3, mode='constant', cval=0.0)

    return resize_and_pad(shifted, target_size)

# Main processing

def process_real_fits(input_path, output_path,
                      target_size=96,
                      subtract_sky=True,
                      sky_method='edge_median',
                      denoise_method='none',
                      centering_method='brightness_weighted',
                      skip_centering=False,
                      band_index=0,
                      pixel_scale_override=None,
                      psf_fwhm_arcsec=None,
                      psf_fwhm_bounds=(0.3, 25.0),
                      verbose=True):
    """Process a single real FITS file for AstroRIM inference.

    Parameters
    ----------
    input_path : str
        Path to the input FITS file.
    output_path : str
        Path where the processed FITS will be written.
    target_size : int
        Final image size (square). Must match the training image size (96 for v2).
    subtract_sky : bool
        If True, subtract an estimated sky background before further processing.
        Recommended True for HST/ground-based observations, which typically have a
        non-zero sky level.
    sky_method : str
        'edge_median', 'percentile', or 'sigma_clip'. See estimate_sky docstring.
    denoise_method : str
        'none' (recommended), 'median3', or 'gauss05'. See gentle_denoise docstring.
        The forward operator handles noise natively; aggressive denoising hurts
        reconstruction quality.
    centering_method : str
        'brightness_weighted' (default), 'ring_detection', or 'peak'.
    skip_centering : bool
        If True, just resize+pad without recentering.
    band_index : int
        For 3D input arrays (multi-band), which band to select.
    pixel_scale_override : float or None
        [NEW N1] Input pixel scale in arcsec/px; overrides anything read from
        the source header. Use when the FITS lacks WCS/PIXSCALE keywords.
    psf_fwhm_arcsec : float or None
        [NEW N2] PSF FWHM of the observation in arcsec, used (with the
        post-resampling scale) to verify the PSF lies inside the encoder's
        representable FWHM range. If None, a PSF_FWH header keyword is used
        when present.
    psf_fwhm_bounds : (float, float)
        [NEW N2] Encoder FWHM bounds in (new) detector pixels. v2.1 default
        (0.3, 25.0); pass (1.0, 25.0) when targeting a legacy v2.0 checkpoint.
    verbose : bool
        If True, print progress.
    """
    try:
        if verbose:
            print(f"\nProcessing: {Path(input_path).name}")

        with fits.open(input_path) as hdul:
            # Try primary HDU first; fall back to first HDU with 2D/3D data.
            data = None
            src_header = None
            for h in hdul:
                if h.data is None:
                    continue
                try:
                    arr = h.data.astype(np.float32)
                except Exception:
                    continue
                if arr.ndim in (2, 3):
                    data = arr
                    src_header = h.header.copy()
                    break

            if data is None:
                raise ValueError(f"No 2D or 3D image data found in any HDU of {input_path}")

        # Handle multi-band
        if data.ndim == 3:
            n_bands = data.shape[0]
            if band_index >= n_bands:
                if verbose:
                    print(f"   Warning: requested band {band_index} but only {n_bands} available; using band 0")
                band_index = 0
            data = data[band_index, :, :]
            if verbose:
                print(f"   Selected band {band_index} of {n_bands}")
        elif data.ndim > 3:
            raise ValueError(f"Unsupported image dimensions: {data.ndim}")

        # NaN/inf cleanup
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        original_shape = data.shape
        original_min = float(data.min())
        original_max = float(data.max())
        if verbose:
            print(f"   Original shape: {original_shape}, range: [{original_min:.4g}, {original_max:.4g}]")

        # Optional sky subtraction (linear; preserves photometry)
        sky_level = 0.0
        if subtract_sky:
            sky_level = estimate_sky(data, method=sky_method)
            data = data - sky_level
            if verbose:
                print(f"   Subtracted sky ({sky_method}): {sky_level:.4g}")

        # Clip negatives (not strictly necessary but avoids numerical issues downstream)
        # Note: this is a simple nonneg projection, not a nonlinear remapping
        data = np.clip(data, 0.0, None).astype(np.float32)

        # Optional gentle denoise (default off)
        if denoise_method != 'none':
            data = gentle_denoise(data, method=denoise_method).astype(np.float32)
            if verbose:
                print(f"   Applied gentle denoise: {denoise_method}")

        # Center + resize to target
        if skip_centering:
            data = resize_and_pad(data, target_size)
            if verbose:
                print(f"   Resized to {target_size}x{target_size} (centering skipped)")
        else:
            data = center_and_resize(data, target_size=target_size,
                                     centering_method=centering_method, verbose=verbose)
            if verbose:
                print(f"   Lens centered ({centering_method}) and resized to {target_size}x{target_size}")

        # Final range report (no normalization - inference script handles it)
        final_min = float(data.min())
        final_max = float(data.max())
        if verbose:
            print(f"   Output range: [{final_min:.4g}, {final_max:.4g}]")

        # Build output FITS
        # Convention: primary HDU metadata, LENSED image extension (matches simgen format)
        prih = fits.PrimaryHDU()
        # Preserve any useful keys from the source header on the primary
        if src_header is not None:
            for key in ('INSTRUME', 'TELESCOP', 'DETECTOR', 'FILTER', 'EXPTIME',
                        'PIXSCALE', 'PIXSCAL1', 'PIXSCAL2', 'CD1_1', 'CD2_2',
                        'RA', 'DEC', 'CRVAL1', 'CRVAL2', 'OBJECT', 'DATE-OBS'):
                if key in src_header:
                    try:
                        prih.header[key] = src_header[key]
                    except Exception:
                        pass

        # Record what we did
        prih.header['PREPROC'] = ('linear_v2', 'AstroRIM v2 linear preprocessing')
        prih.header['PP_DATE'] = (datetime.utcnow().isoformat(), 'Preprocessing UTC timestamp')
        prih.header['PP_SKY'] = (subtract_sky, 'Sky subtraction applied')
        if subtract_sky:
            prih.header['PP_SKY_M'] = (sky_method, 'Sky estimation method')
            prih.header['PP_SKY_V'] = (float(sky_level), 'Subtracted sky level (input units)')
        prih.header['PP_DENS'] = (denoise_method, 'Denoise method (none recommended)')
        prih.header['PP_CTR'] = (not skip_centering, 'Lens centering applied')
        if not skip_centering:
            prih.header['PP_CTR_M'] = (centering_method, 'Centering method')
        prih.header['PP_SIZE'] = (target_size, 'Final image size')
        prih.header['PP_BAND'] = (band_index, 'Band index selected from input')

        # ------------------------------------------------------------------
        # [NEW N1] Pixel-scale extraction and propagation. Downstream mass
        # analysis needs the POST-resampling scale; previously nothing
        # recorded it. The pad step does not change the scale; the isotropic
        # resize factor does: ps_new = ps_orig / scale.
        # ------------------------------------------------------------------
        if pixel_scale_override is not None:
            ps_orig, ps_src = float(pixel_scale_override), 'cli:--pixel-scale'
        else:
            ps_orig, ps_src = extract_pixel_scale(src_header)
        ps_new = None
        if ps_orig is not None:
            scale = resample_scale_factor(original_shape, target_size)
            ps_new = ps_orig / scale
            prih.header['PP_PS_O'] = (float(ps_orig), 'Original pixel scale [arcsec/px]')
            prih.header['PP_PS_N'] = (float(ps_new), 'Pixel scale AFTER resize+pad [arcsec/px]')
            prih.header['PP_PSSRC'] = (str(ps_src), 'Source of original pixel scale')
            if verbose:
                print(f"   Pixel scale: {ps_orig:.5f} -> {ps_new:.5f} arcsec/px "
                      f"(source: {ps_src}, resize factor {scale:.4f})")
        else:
            if verbose:
                print("   WARNING: no pixel scale found (PIXSCALE/PIXSCAL1/CD/"
                      "CDELT absent) and --pixel-scale not given. PP_PS_N will "
                      "be missing; mass_analysis.py will fall back to its "
                      "--pixel-scale assumption. Strongly consider re-running "
                      "with --pixel-scale <arcsec/px>.")

        # ------------------------------------------------------------------
        # [NEW N2] PSF-coverage check against the encoder's FWHM bounds.
        # ------------------------------------------------------------------
        psf_fwhm = psf_fwhm_arcsec
        if psf_fwhm is None and src_header is not None and 'PSF_FWH' in src_header:
            try:
                psf_fwhm = float(src_header['PSF_FWH'])
            except Exception:
                psf_fwhm = None
        if psf_fwhm is not None and ps_new is not None:
            fwhm_px_new = psf_fwhm / ps_new
            prih.header['PP_PSFPX'] = (float(fwhm_px_new),
                                       'PSF FWHM in POST-resample pixels')
            lo, hi = psf_fwhm_bounds
            if not (lo <= fwhm_px_new <= hi):
                msg = (f"   WARNING: PSF FWHM after resampling = {fwhm_px_new:.2f} px, "
                       f"OUTSIDE the encoder range [{lo}, {hi}] px. The model "
                       f"cannot represent this PSF: predictions will saturate at "
                       f"the bound and reconstructions will be systematically "
                       f"mis-blurred. Options: (a) change --target-size so the "
                       f"resampled PSF lands in range, (b) train/evaluate with "
                       f"--psf-fwhm-min/--psf-fwhm-max covering it, (c) fix the "
                       f"PSF externally rather than free-fitting it."
                       + (f" (Note: legacy v2.0 checkpoints have floor 1.0 px.)"
                          if lo != 1.0 else ""))
                print(msg)
            elif verbose:
                print(f"   PSF FWHM after resampling: {fwhm_px_new:.2f} px "
                      f"(inside encoder range [{lo}, {hi}])")

        lensed_hdu = fits.ImageHDU(data=data.astype(np.float32), name='LENSED')
        lensed_hdu.header['BUNIT'] = ('ADU', 'Brightness unit (linear, sky-subtracted if PP_SKY=T)')
        lensed_hdu.header['ORIGSHP0'] = (int(original_shape[0]), 'Original height in pixels')
        lensed_hdu.header['ORIGSHP1'] = (int(original_shape[1]), 'Original width in pixels')
        # [NEW N1] mirror the scale keys on the image HDU as well
        if ps_orig is not None:
            lensed_hdu.header['PP_PS_O'] = (float(ps_orig), 'Original pixel scale [arcsec/px]')
            lensed_hdu.header['PP_PS_N'] = (float(ps_new), 'Pixel scale AFTER resize+pad [arcsec/px]')

        hdul_out = fits.HDUList([prih, lensed_hdu])
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        hdul_out.writeto(output_path, overwrite=True)

        if verbose:
            print(f"   Saved: {Path(output_path).name}")

        return output_path

    except Exception as e:
        if verbose:
            print(f"   Error processing {input_path}: {e}")
        return None

def process_directory(input_dir, output_dir,
                      target_size=96,
                      subtract_sky=True,
                      sky_method='edge_median',
                      denoise_method='none',
                      centering_method='brightness_weighted',
                      skip_centering=False,
                      recursive=True,
                      band_index=0,
                      pixel_scale_override=None,
                      psf_fwhm_arcsec=None,
                      psf_fwhm_bounds=(0.3, 25.0)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)

    if recursive:
        fits_files = list(input_path.rglob("*.fits")) + \
                     list(input_path.rglob("*.FITS")) + \
                     list(input_path.rglob("*.fit")) + \
                     list(input_path.rglob("*.FIT"))
    else:
        fits_files = list(input_path.glob("*.fits")) + \
                     list(input_path.glob("*.FITS")) + \
                     list(input_path.glob("*.fit")) + \
                     list(input_path.glob("*.FIT"))

    fits_files = sorted(set(fits_files))

    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return 0, 0

    print(f"\nProcessing {len(fits_files)} FITS files from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pipeline: sky_subtract={subtract_sky} ({sky_method if subtract_sky else 'off'}), "
          f"denoise={denoise_method}, centering={'off' if skip_centering else centering_method}, "
          f"target_size={target_size}")

    n_ok = 0
    n_fail = 0
    for in_path in fits_files:
        # Mirror relative path in output
        try:
            rel = in_path.relative_to(input_path)
        except ValueError:
            rel = Path(in_path.name)
        out_file = output_path / rel.with_name(f"{rel.stem}_processed.fits")

        result = process_real_fits(
            str(in_path), str(out_file),
            target_size=target_size,
            subtract_sky=subtract_sky,
            sky_method=sky_method,
            denoise_method=denoise_method,
            centering_method=centering_method,
            skip_centering=skip_centering,
            band_index=band_index,
            pixel_scale_override=pixel_scale_override,
            psf_fwhm_arcsec=psf_fwhm_arcsec,
            psf_fwhm_bounds=psf_fwhm_bounds,
            verbose=True,
        )
        if result is not None:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone. {n_ok} processed, {n_fail} failed.")
    return n_ok, n_fail

# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Prepare real FITS observations for AstroRIM v2 inference "
                    "(linear, no nonlinear remapping).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                       help='Input directory with FITS files')
    group.add_argument('--input-file', type=str,
                       help='Single FITS file to process')

    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--target-size', type=int, default=TARGET_SIZE,
                        help='Target output size (must match training image size: 96 for v2)')

    parser.add_argument('--no-sky-subtract', dest='subtract_sky', action='store_false',
                        help='Skip sky subtraction (not recommended for real observations)')
    parser.set_defaults(subtract_sky=True)
    parser.add_argument('--sky-method', type=str, default='edge_median',
                        choices=['edge_median', 'percentile', 'sigma_clip'],
                        help='Sky estimation method')

    parser.add_argument('--denoise', type=str, default='none',
                        choices=['none', 'median3', 'gauss05'],
                        help='Denoise method (none recommended - forward operator handles noise)')

    parser.add_argument('--centering', type=str, default='brightness_weighted',
                        choices=['brightness_weighted', 'ring_detection', 'peak'],
                        help='Centering method')
    parser.add_argument('--skip-centering', action='store_true', default=False,
                        help='Skip lens centering')

    parser.add_argument('--band', type=int, default=0,
                        help='For 3D (multi-band) inputs, which band to select')

    # [NEW N1/N2]
    parser.add_argument('--pixel-scale', type=float, default=None,
                        help='Input pixel scale in arcsec/px; overrides header '
                             'PIXSCALE/WCS. Required for files without WCS if you '
                             'want PP_PS_N written (mass_analysis depends on it).')
    parser.add_argument('--psf-fwhm-arcsec', type=float, default=None,
                        help='PSF FWHM of the observation in arcsec, for the '
                             'encoder-coverage sanity check (falls back to a '
                             'PSF_FWH header keyword when present)')
    parser.add_argument('--psf-fwhm-min', type=float, default=0.3,
                        help='Encoder FWHM lower bound in NEW pixels for the '
                             'coverage check (legacy v2.0 checkpoints: 1.0)')
    parser.add_argument('--psf-fwhm-max', type=float, default=25.0)

    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not search subdirectories')

    args = parser.parse_args()

    if args.input_file:
        out_file = Path(args.output_dir) / f"{Path(args.input_file).stem}_processed.fits"
        process_real_fits(
            args.input_file, str(out_file),
            target_size=args.target_size,
            subtract_sky=args.subtract_sky,
            sky_method=args.sky_method,
            denoise_method=args.denoise,
            centering_method=args.centering,
            skip_centering=args.skip_centering,
            band_index=args.band,
            pixel_scale_override=args.pixel_scale,
            psf_fwhm_arcsec=args.psf_fwhm_arcsec,
            psf_fwhm_bounds=(args.psf_fwhm_min, args.psf_fwhm_max),
            verbose=True,
        )
    else:
        process_directory(
            args.input_dir, args.output_dir,
            target_size=args.target_size,
            subtract_sky=args.subtract_sky,
            sky_method=args.sky_method,
            denoise_method=args.denoise,
            centering_method=args.centering,
            skip_centering=args.skip_centering,
            recursive=not args.no_recursive,
            band_index=args.band,
            pixel_scale_override=args.pixel_scale,
            psf_fwhm_arcsec=args.psf_fwhm_arcsec,
            psf_fwhm_bounds=(args.psf_fwhm_min, args.psf_fwhm_max),
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided, using defaults")
        print(f"   Input:  {DEFAULT_INPUT_DIR}")
        print(f"   Output: {DEFAULT_OUTPUT_DIR}\n")
        process_directory(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)
    else:
        main()
