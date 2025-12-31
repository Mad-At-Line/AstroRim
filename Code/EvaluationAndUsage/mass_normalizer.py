#!/usr/bin/env python3
"""
Real Data Preprocessor for Gravitational Lensing
Handles noise reduction, feature enhancement, background filtering, and lens centering
"""
import os
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from skimage.transform import resize
from skimage.filters import gaussian, median, unsharp_mask, sobel
from skimage.exposure import adjust_gamma, equalize_adapthist
from scipy.ndimage import center_of_mass, shift as nd_shift
from scipy.ndimage import median_filter, gaussian_filter
from skimage import measure, morphology

DEFAULT_INPUT_DIR = r"YOUR INPUT DIRECTORY"
DEFAULT_OUTPUT_DIR = r"YOUR OUTPUT DIRECTORY"
TARGET_SIZE = 96


def estimate_background(data, method='percentile', percentile=10, sigma=3.0):
    if method == 'percentile':
        return np.percentile(data, percentile)
    elif method == 'median':
        h, w = data.shape
        edges = np.concatenate([data[0, :], data[-1, :], data[:, 0], data[:, -1]])
        return np.median(edges)
    elif method == 'sigma_clip':
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        std = 1.4826 * mad
        mask = np.abs(data - med) < sigma * std
        return np.median(data[mask])
    return 0.0


def denoise_image(data, method='combined', **params):
    if method == 'gaussian':
        sigma = params.get('sigma', 0.5)
        return gaussian_filter(data, sigma=sigma)
    elif method == 'median':
        size = params.get('size', 3)
        return median_filter(data, size=size)
    elif method == 'bilateral':
        sigma_spatial = params.get('sigma_spatial', 1.0)
        sigma_intensity = params.get('sigma_intensity', 0.1)
        smoothed = gaussian_filter(data, sigma=sigma_spatial)
        diff = np.abs(data - smoothed)
        weight = np.exp(-(diff ** 2) / (2 * sigma_intensity ** 2))
        return weight * data + (1 - weight) * smoothed
    elif method == 'combined':
        bg = estimate_background(data, method='sigma_clip')
        data_sub = data - bg
        data_sub = np.clip(data_sub, 0, None)
        denoised = median_filter(data_sub, size=3)
        denoised = gaussian_filter(denoised, sigma=0.5)
        return denoised + bg
    return data


def enhance_features(data, method='adaptive_ring', **params):
    if method == 'adaptive_ring':
        clip_limit = params.get('clip_limit', 0.01)
        enhanced = equalize_adapthist(data, clip_limit=clip_limit, kernel_size=None)
        gamma = params.get('gamma', 0.75)
        enhanced = adjust_gamma(enhanced, gamma=gamma)
        radius = params.get('radius', 1.2)
        amount = params.get('amount', 1.8)
        enhanced = unsharp_mask(enhanced, radius=radius, amount=amount)
        sigmoid_gain = params.get('sigmoid_gain', 7)
        enhanced = 1.0 / (1.0 + np.exp(-sigmoid_gain * (enhanced - 0.5)))
        return enhanced
    elif method == 'multiscale':
        scales = [0.5, 1.0, 2.0, 4.0]
        components = []
        for sigma in scales:
            smoothed = gaussian_filter(data, sigma=sigma)
            detail = data - smoothed
            components.append(detail * 1.5)
        enhanced = data + sum(components) * 0.3
        enhanced = np.clip(enhanced, 0, 1)
        return enhanced
    elif method == 'unsharp_strong':
        radius = params.get('radius', 3.0)
        amount = params.get('amount', 4.0)
        enhanced = unsharp_mask(data, radius=radius, amount=amount)
        enhanced = adjust_gamma(enhanced, gamma=0.5)
        return enhanced
    return data


def filter_background_objects(data, thresh_sigma=3.0, min_area=50, edge_overlap_thresh=0.2, bg_method='sigma_clip'):
    """
    Remove small compact bright objects (e.g. stars, artifacts) while keeping
    extended ring-like / edge-rich structures.
    
    Strategy:
      - Estimate background and std
      - Threshold to find bright candidate objects
      - Label connected components
      - Compute area and edge-overlap fraction for each component
      - Remove components that are small and *not* edge-rich
      - Do NOT overdo this - in tests I found that the models perform better with original data vs over-filtered data
    """
    # Estimate background and robust std
    bg = estimate_background(data, method=bg_method)
    mad = np.median(np.abs(data - np.median(data)))
    robust_std = 1.4826 * mad if mad > 0 else np.std(data)
    thresh = bg + thresh_sigma * (robust_std + 1e-12)

    bright_mask = data > thresh
    if bright_mask.sum() == 0:
        return data  # nothing to filter

    # Edge map to detect ring-like structures
    edges = sobel(data)
    edges_norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-12)

    labeled = measure.label(bright_mask, connectivity=2)
    props = measure.regionprops(labeled)

    # Prepare a copy we will modify
    filtered = data.copy()

    for prop in props:
        area = prop.area
        coords = prop.coords
        # Compute fraction of pixels in this component that overlap with strong edges
        edge_vals = edges_norm[coords[:, 0], coords[:, 1]]
        edge_overlap_fraction = np.mean(edge_vals > np.percentile(edges_norm, 75))

        # If area is small and it does not overlap strongly with edges -> remove
        if (area < min_area) and (edge_overlap_fraction < edge_overlap_thresh):
            # set to background (interpolate via local median)
            for (r, c) in coords:
                filtered[r, c] = bg
    return filtered


def find_lens_center(data, method='brightness_weighted', **params):
    if method == 'brightness_weighted':
        threshold = params.get('threshold', 0.001)
        weights = np.clip(data - threshold * np.max(data), 0, None)
        weights = weights ** 2
        if weights.sum() > 0:
            cy, cx = center_of_mass(weights)
            return cy, cx
        else:
            return data.shape[0] / 2, data.shape[1] / 2
    elif method == 'ring_detection':
        edges = sobel(data)
        threshold = np.percentile(edges, 90)
        ring_mask = edges > threshold
        if ring_mask.sum() > 0:
            cy, cx = center_of_mass(ring_mask)
            return cy, cx
        else:
            return data.shape[0] / 2, data.shape[1] / 2
    elif method == 'peak':
        cy, cx = np.unravel_index(np.argmax(data), data.shape)
        return cy, cx
    return data.shape[0] / 2, data.shape[1] / 2


def _resize_and_pad_to_target(img, target_size):
    """
    Resize full image to fit within target_size while preserving field-of-view
    (no cropping). The larger image dimension is scaled down to target_size,
    the other dimension is scaled accordingly, then the result is centered
    in a target_size x target_size canvas (zero-padded).
    """
    h, w = img.shape
    if h == target_size and w == target_size:
        return img

    # scale to fit max dimension to target_size
    scale = target_size / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = resize(img, (new_h, new_w), preserve_range=True, anti_aliasing=True)

    # pad to target_size
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0.0)

    # in rare rounding cases, crop to exact target
    if padded.shape != (target_size, target_size):
        padded = padded[:target_size, :target_size]
    return padded


def center_lens(data, target_size=96, centering_method='brightness_weighted', **params):
    """
    Center the lens system but DO NOT crop/zoom. Steps:
      - find center in original resolution
      - shift image so lens center lies at geometric center
      - downscale the whole shifted image so largest dim == target_size
      - pad to exact target_size
    """
    h, w = data.shape
    cy, cx = find_lens_center(data, method=centering_method, **params)
    if params.get('verbose', True):
        print(f"   Detected center: ({cy:.1f}, {cx:.1f}), Image center: ({h/2:.1f}, {w/2:.1f})")

    shift_y = h / 2 - cy
    shift_x = w / 2 - cx
    shifted = nd_shift(data, shift=(shift_y, shift_x), order=3, mode='constant', cval=0.0)

    # Resize + pad to target without cropping (preserve full FOV)
    final = _resize_and_pad_to_target(shifted, target_size)
    return final


def process_real_fits(input_path, output_path,
                     target_size=96,
                     denoise_method='combined',
                     denoise_params=None,
                     enhance_method='adaptive_ring',
                     enhance_params=None,
                     centering_method='brightness_weighted',
                     centering_params=None,
                     skip_centering=False,
                     filter_bg=True,
                     bg_thresh_sigma=3.0,
                     bg_min_area=50,
                     bg_edge_thresh=0.2,
                     verbose=True):
    try:
        if verbose:
            print(f"\n📂 Processing: {Path(input_path).name}")

        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()

        if data.ndim == 3:
            data = data[0, :, :]
        elif data.ndim > 3:
            raise ValueError(f"Unsupported dimensions: {data.ndim}")

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Initial normalization to [0, 1]
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

        if verbose:
            print(f"   Original shape: {data.shape}, Range: [{data.min():.3f}, {data.max():.3f}]")

        # Step 1: Denoise
        if denoise_params is None:
            denoise_params = {}
        data = denoise_image(data, method=denoise_method, **denoise_params)
        if verbose:
            print(f"Denoising applied ({denoise_method})")

        # Step 1.5: Optional background object filtering (before enhancement)
        if filter_bg:
            data = filter_background_objects(data,
                                             thresh_sigma=bg_thresh_sigma,
                                             min_area=bg_min_area,
                                             edge_overlap_thresh=bg_edge_thresh)
            if verbose:
                print(f"Background filtering applied (sigma={bg_thresh_sigma}, min_area={bg_min_area})")

        # Step 2: Enhance features
        if enhance_params is None:
            enhance_params = {}
        data = enhance_features(data, method=enhance_method, **enhance_params)
        if verbose:
            print(f"Enhancement applied ({enhance_method})")
            if enhance_method == 'adaptive_ring' and enhance_params:
                print(f"     Parameters: gamma={enhance_params.get('gamma', 0.7):.2f}, "
                      f"radius={enhance_params.get('radius', 1.5):.1f}, "
                      f"amount={enhance_params.get('amount', 2.0):.1f}")

        # Step 3: Center lens (optional) — but do NOT crop/zoom, I prefer to downscale full image instead
        if not skip_centering:
            if centering_params is None:
                centering_params = {}
            centering_params['verbose'] = verbose
            data = center_lens(data, target_size=target_size,
                               centering_method=centering_method,
                               **centering_params)
            if verbose:
                print(f"Lens centered ({centering_method})")
        else:
            # Resize to target preserving FOV (no cropping)
            if data.shape != (target_size, target_size):
                data = _resize_and_pad_to_target(data, target_size)
            if verbose:
                print(f"Resized to {target_size}x{target_size} (centering skipped)")

        # Final normalization
        data = np.clip(data, 0, 1)
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

        if verbose:
            print(f"   Final range: [{data.min():.3f}, {data.max():.3f}]")

        # Update header
        header['PREPROC'] = ('enhanced_v2', 'Preprocessing version')
        header['DENOISE'] = (denoise_method, 'Denoising method')
        header['ENHANCE'] = (enhance_method, 'Enhancement method')
        header['CENTERED'] = (not skip_centering, 'Lens centered')
        header['FILTERBG'] = (filter_bg, 'Background filter applied')
        header['BG_SIG'] = (bg_thresh_sigma, 'BG threshold sigma')
        header['BG_MINA'] = (bg_min_area, 'BG min area')

        if enhance_method == 'adaptive_ring' and enhance_params:
            header['ENH_GAM'] = (enhance_params.get('gamma', 0.75), 'Enhancement gamma')
            header['ENH_RAD'] = (enhance_params.get('radius', 1.3), 'Enhancement radius')
            header['ENH_AMT'] = (enhance_params.get('amount', 1.8), 'Enhancement amount')
            header['ENH_CLIP'] = (enhance_params.get('clip_limit', 0.015), 'CLAHE clip limit')
            header['ENH_SIG'] = (enhance_params.get('sigmoid_gain', 7), 'Sigmoid gain')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fits.writeto(output_path, data.astype(np.float32), header, overwrite=True)

        if verbose:
            print(f"Saved to: {Path(output_path).name}\n")

        return output_path

    except Exception as e:
        if verbose:
            print(f"Error: {e}\n")
        return None


def process_directory(input_dir, output_dir, target_size=96,
                     denoise_method='combined',
                     enhance_method='adaptive_ring',
                     enhance_params=None,
                     centering_method='brightness_weighted',
                     skip_centering=False,
                     recursive=True,
                     filter_bg=True,
                     bg_thresh_sigma=3.0,
                     bg_min_area=50,
                     bg_edge_thresh=0.2):
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

    fits_files = list(set(fits_files))

    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return 0, 0

    print(f"\n{'='*70}")
    print(f"REAL DATA PREPROCESSING PIPELINE")
    print(f"{'='*70}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Denoising: {denoise_method}")
    print(f"Enhancement: {enhance_method}")
    print(f"Centering: {centering_method if not skip_centering else 'DISABLED'}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Files found: {len(fits_files)}")
    print(f"{'='*70}")

    success_count = 0
    failure_count = 0

    for fits_file in fits_files:
        stem = fits_file.stem
        suffix = fits_file.suffix

        output_file = output_path / f"{stem}_processed{suffix}"
        counter = 1
        while output_file.exists():
            output_file = output_path / f"{stem}_processed_{counter}{suffix}"
            counter += 1

        result = process_real_fits(
            str(fits_file), str(output_file),
            target_size=target_size,
            denoise_method=denoise_method,
            denoise_params=None,
            enhance_method=enhance_method,
            enhance_params=enhance_params,
            centering_method=centering_method,
            centering_params=None,
            skip_centering=skip_centering,
            filter_bg=filter_bg,
            bg_thresh_sigma=bg_thresh_sigma,
            bg_min_area=bg_min_area,
            bg_edge_thresh=bg_edge_thresh
        )

        if result is not None:
            success_count += 1
        else:
            failure_count += 1

    print(f"\n{'='*70}")
    print(f"Successfully processed: {success_count} file(s)")
    if failure_count > 0:
        print(f"Failed to process: {failure_count} file(s)")
    print(f"{'='*70}\n")

    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced preprocessing for real gravitational lens FITS images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                      help='Input directory with FITS files')
    group.add_argument('--input-file', type=str,
                      help='Single FITS file to process')

    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--target-size', type=int, default=TARGET_SIZE,
                       help='Target output size (default: 96)')

    parser.add_argument('--denoise', type=str, default='combined',
                       choices=['gaussian', 'median', 'bilateral', 'combined'],
                       help='Denoising method (default: combined)')

    parser.add_argument('--enhance', type=str, default='adaptive_ring',
                       choices=['adaptive_ring', 'multiscale', 'unsharp_strong'],
                       help='Enhancement method (default: adaptive_ring)')

    parser.add_argument('--gamma', type=float, default=0.75,
                       help='Gamma for brightening (0.5-0.9, lower=brighter, default: 0.75)')
    parser.add_argument('--radius', type=float, default=1.2,
                       help='Unsharp mask radius (1.0-3.0, lower=finer edges, default: 1.2)')
    parser.add_argument('--amount', type=float, default=1.8,
                       help='Unsharp mask strength (1.0-4.0, higher=sharper, default: 1.8)')
    parser.add_argument('--clip-limit', type=float, default=0.01,
                       help='CLAHE clip limit (0.01-0.05, lower=gentler, default: 0.01)')
    parser.add_argument('--sigmoid-gain', type=float, default=7.0,
                       help='Sigmoid contrast gain (5-15, higher=more contrast, default: 7.0)')

    parser.add_argument('--centering', type=str, default='brightness_weighted',
                       choices=['brightness_weighted', 'ring_detection', 'peak'],
                       help='Centering method (default: brightness_weighted)')

    # IMPORTANT: default centering is ON; you can set this flag to skip centering, I recommend centering only for specific cases, (e.g. low resolution images)
    parser.add_argument('--skip-centering', action='store_true', default=False,
                        help='Skip automatic centering (default: False)')

    parser.add_argument('--filter-bg', action='store_true', default=True,
                        help='Apply background object filtering to remove compact bright sources (default: True)')
    parser.add_argument('--bg-thresh-sigma', type=float, default=3.0,
                        help='Sigma threshold above background for object detection (default: 3.0)')
    parser.add_argument('--bg-min-area', type=int, default=50,
                        help='Minimum area (pixels) to keep a bright object (default: 50)')
    parser.add_argument('--bg-edge-thresh', type=float, default=0.2,
                        help='Edge-overlap threshold to preserve a small bright object if it is edge-rich (default: 0.2)')

    parser.add_argument('--no-recursive', action='store_true',
                       help='Do not search subdirectories')

    args = parser.parse_args()

    # Build enhancement params
    enhance_params = {
        'gamma': args.gamma,
        'radius': args.radius,
        'amount': args.amount,
        'clip_limit': args.clip_limit,
        'sigmoid_gain': args.sigmoid_gain
    }

    if args.input_file:
        output_file = Path(args.output_dir) / f"{Path(args.input_file).stem}_processed.fits"
        process_real_fits(
            args.input_file, str(output_file),
            target_size=args.target_size,
            denoise_method=args.denoise,
            denoise_params=None,
            enhance_method=args.enhance,
            enhance_params=enhance_params,
            centering_method=args.centering,
            centering_params=None,
            skip_centering=args.skip_centering,
            filter_bg=args.filter_bg,
            bg_thresh_sigma=args.bg_thresh_sigma,
            bg_min_area=args.bg_min_area,
            bg_edge_thresh=args.bg_edge_thresh
        )
    else:
        recursive = not args.no_recursive
        process_directory(
            args.input_dir, args.output_dir,
            target_size=args.target_size,
            denoise_method=args.denoise,
            enhance_method=args.enhance,
            enhance_params=enhance_params,
            centering_method=args.centering,
            skip_centering=args.skip_centering,
            recursive=recursive,
            filter_bg=args.filter_bg,
            bg_thresh_sigma=args.bg_thresh_sigma,
            bg_min_area=args.bg_min_area,
            bg_edge_thresh=args.bg_edge_thresh
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
