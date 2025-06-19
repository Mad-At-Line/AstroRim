import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from scipy.interpolate import griddata
import os
import random
from scipy.ndimage import gaussian_filter

def normalize_img(img):
    """Normalize image for display or saving (0–1 range)"""
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    return np.clip(img / np.max(img), 0, 1) if np.max(img) != 0 else img

def add_gaussian_star(image, x, y, amp, sigma):
    """Adds a Gaussian star to the image at position (x, y)"""
    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    star = amp * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return image + star

def generate_lens_model():
    """Randomly selects lens models and their parameters"""
    lens_types = random.sample(['SIE', 'NFW', 'SIS', 'SHEAR'], k=random.randint(1, 3))
    kwargs_lens = []
    for lens in lens_types:
        if lens == 'SIE':
            kwargs_lens.append({
                'theta_E': np.random.uniform(1.0, 1.6),
                'e1': np.random.uniform(-0.2, 0.2),
                'e2': np.random.uniform(-0.2, 0.2),
                'center_x': 0.0,
                'center_y': 0.0
            })
        elif lens == 'NFW':
            kwargs_lens.append({
                'Rs': np.random.uniform(0.2, 0.6),
                'alpha_Rs': np.random.uniform(0.5, 1.2),
                'center_x': 0.0,
                'center_y': 0.0
            })
        elif lens == 'SIS':
            kwargs_lens.append({
                'theta_E': np.random.uniform(0.6, 1.2),
                'center_x': 0.0,
                'center_y': 0.0
            })
        elif lens == 'SHEAR':
            kwargs_lens.append({
                'gamma1': np.random.uniform(-0.05, 0.05),
                'gamma2': np.random.uniform(-0.05, 0.05)
            })
    return lens_types, kwargs_lens

def generate_source_params(n_sources):
    """Generates source galaxy parameters (same for all channels)"""
    sources = []
    for _ in range(n_sources):
        cx, cy = np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)
        sources.append({
            'amp': np.random.uniform(2.0, 5.5),
            'R_sersic': np.random.uniform(0.1, 0.3),
            'n_sersic': np.random.uniform(1.0, 4.0),
            'e1': np.random.uniform(-0.3, 0.3),
            'e2': np.random.uniform(-0.3, 0.3),
            'center_x': cx,
            'center_y': cy
        })
    return sources

def generate_realistic_simulation_rgb(image_size=64, num_sources=2, num_stars=4, num_bg_galaxies=3, add_noise=True):
    """Full RGB gravitational lensing simulation with consistent features"""
    grid = np.linspace(-2.5, 2.5, image_size)
    x_grid, y_grid = np.meshgrid(grid, grid)

    lens_types, kwargs_lens = generate_lens_model()
    lens_model = LensModel(lens_model_list=lens_types)

    x_src, y_src = lens_model.ray_shooting(x_grid.flatten(), y_grid.flatten(), kwargs_lens)
    x_src = x_src.reshape(image_size, image_size)
    y_src = y_src.reshape(image_size, image_size)

    gt_rgb = np.zeros((3, image_size, image_size), dtype=np.float32)
    lensed_rgb = np.zeros_like(gt_rgb)

    # Generate sources (same parameters for all channels)
    kwargs_src = generate_source_params(num_sources)
    model = LightModel(light_model_list=['SERSIC_ELLIPSE'] * num_sources)
    
    for c in range(3):
        # Apply slight color variation (10%) while keeping structure identical
        color_factor = 0.9 + 0.2 * c/2  # R:0.9, G:1.0, B:1.1
        colored_kwargs = [{
            **kw, 
            'amp': kw['amp'] * color_factor
        } for kw in kwargs_src]
        
        gt = model.surface_brightness(x_grid, y_grid, colored_kwargs)
        lensed = griddata(
            (x_grid.flatten(), y_grid.flatten()), gt.flatten(),
            (x_src, y_src), method='linear', fill_value=0.0
        )
        gt_rgb[c] = gt
        lensed_rgb[c] = np.nan_to_num(lensed)

    # Background galaxies (added to BOTH images)
    bg_galaxy_params = []
    for _ in range(num_bg_galaxies):
        params = {
            'amp': np.random.uniform(0.5, 1.5),
            'R_sersic': np.random.uniform(0.05, 0.2),
            'n_sersic': 1.0,
            'e1': np.random.uniform(-0.1, 0.1),
            'e2': np.random.uniform(-0.1, 0.1),
            'center_x': np.random.uniform(-1.2, 1.2),
            'center_y': np.random.uniform(-1.2, 1.2)
        }
        bg_galaxy_params.append(params)
        
    for params in bg_galaxy_params:
        model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        for c in range(3):
            gt_rgb[c] += model.surface_brightness(x_grid, y_grid, [params])
            lensed_rgb[c] += model.surface_brightness(x_grid, y_grid, [params])

    # Stars (added to BOTH images with same positions)
    star_params = []
    for _ in range(num_stars):
        params = {
            'x': np.random.randint(5, image_size-5),
            'y': np.random.randint(5, image_size-5),
            'sigma': np.random.uniform(0.6, 1.2),
            'amp': np.random.uniform(3.0, 8.0),
            'color': np.random.dirichlet([1, 1, 1])  # Random color profile
        }
        star_params.append(params)
    
    for params in star_params:
        for c in range(3):
            gt_rgb[c] = add_gaussian_star(
                gt_rgb[c], params['x'], params['y'], 
                params['amp'] * params['color'][c], params['sigma']
            )
            lensed_rgb[c] = add_gaussian_star(
                lensed_rgb[c], params['x'], params['y'], 
                params['amp'] * params['color'][c], params['sigma']
            )

    # Add noise only to lensed image (observational noise)
    if add_noise:
        for c in range(3):
            lensed_rgb[c] += np.abs(np.random.normal(scale=0.01, size=(image_size, image_size)))
            
    # Global normalization
    max_val = max(np.max(gt_rgb), np.max(lensed_rgb), 1e-6)
    gt_rgb /= max_val
    lensed_rgb /= max_val

    return gt_rgb, lensed_rgb, lens_types, kwargs_lens

# ========== MAIN EXECUTION ==========
if __name__ == '__main__':
    output_dir = r'C:\Users\mythi\.astropy\Code\COLOR\colordata2'
    os.makedirs(output_dir, exist_ok=True)

    num_examples = 5 
    print(f"Generating {num_examples} simulations...")
    
    # Create a visualization of the first 5 examples
    num_visualize = 5
    fig, axs = plt.subplots(num_visualize, 3, figsize=(12, 4 * num_visualize))
    
    for i in range(num_examples):
        if i < num_visualize:
            # Get lens model info for visualization
            gt, lensed, lens_types, kwargs_lens = generate_realistic_simulation_rgb()
            
            # Add lens model info to filename
            lens_str = "_".join(lens_types)
            filename = f'simulation_rgb_{i:04d}_{lens_str}.npz'
            
            # Visualization
            axs[i, 0].imshow(normalize_img(gt))
            axs[i, 0].set_title(f"GT ({i+1})")
            axs[i, 1].imshow(normalize_img(lensed))
            axs[i, 1].set_title(f"Lensed ({lens_str})")
            
            # Difference image
            diff = np.sum(np.abs(gt - lensed), axis=0)
            axs[i, 2].imshow(diff, cmap='hot')
            axs[i, 2].set_title("Difference")
            
            for j in range(3):
                axs[i, j].axis("off")
        else:
            gt, lensed, _, _ = generate_realistic_simulation_rgb()
            filename = f'simulation_rgb_{i:04d}.npz'
        
        np.savez(os.path.join(output_dir, filename), gt=gt, lensed=lensed)
        
        if i % 100 == 0:
            print(f"Saved simulation {i+1} of {num_examples}: {filename}")

    # Save and show visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_samples.png"), dpi=150)
    plt.show()
    print("Dataset generation complete!")