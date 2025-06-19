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

def generate_source_params(n_sources, color_channel):
    """Generates diverse source galaxy parameters per channel"""
    sources = []
    for _ in range(n_sources):
        cx, cy = np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)
        sources.append({
            'amp': np.random.uniform(2.0, 5.5) * (1 + 0.3 * color_channel),
            'R_sersic': np.random.uniform(0.1, 0.3),
            'n_sersic': np.random.uniform(1.0, 4.0),
            'e1': np.random.uniform(-0.3, 0.3),
            'e2': np.random.uniform(-0.3, 0.3),
            'center_x': cx,
            'center_y': cy
        })
    return sources

def generate_realistic_simulation_rgb(image_size=64, num_sources=2, num_stars=4, num_bg_galaxies=3, add_noise=True):
    """Full RGB gravitational lensing simulation with realistic features"""
    grid = np.linspace(-2.5, 2.5, image_size)
    x_grid, y_grid = np.meshgrid(grid, grid)

    lens_types, kwargs_lens = generate_lens_model()
    lens_model = LensModel(lens_model_list=lens_types)

    x_src, y_src = lens_model.ray_shooting(x_grid.flatten(), y_grid.flatten(), kwargs_lens)
    x_src = x_src.reshape(image_size, image_size)
    y_src = y_src.reshape(image_size, image_size)

    gt_rgb = np.zeros((3, image_size, image_size), dtype=np.float32)
    lensed_rgb = np.zeros_like(gt_rgb)

    for c in range(3):
        kwargs_src = generate_source_params(num_sources, c)
        model = LightModel(light_model_list=['SERSIC_ELLIPSE'] * num_sources)
        gt = model.surface_brightness(x_grid, y_grid, kwargs_src)
        lensed = griddata(
            (x_grid.flatten(), y_grid.flatten()), gt.flatten(),
            (x_src, y_src), method='linear', fill_value=0.0
        )
        gt_rgb[c] = gt
        lensed_rgb[c] = np.nan_to_num(lensed)

    # Background galaxies
    for _ in range(num_bg_galaxies):
        cx, cy = np.random.uniform(-1.2, 1.2), np.random.uniform(-1.2, 1.2)
        model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        for c in range(3):
            kwargs = [{
                'amp': np.random.uniform(0.5, 1.5),
                'R_sersic': np.random.uniform(0.05, 0.2),
                'n_sersic': 1.0,
                'e1': np.random.uniform(-0.1, 0.1),
                'e2': np.random.uniform(-0.1, 0.1),
                'center_x': cx,
                'center_y': cy
            }]
            lensed_rgb[c] += model.surface_brightness(x_grid, y_grid, kwargs)

    # Stars
    for _ in range(num_stars):
        x_star, y_star = np.random.randint(5, image_size - 5, 2)
        sigma = np.random.uniform(0.6, 1.2)
        amp = np.random.uniform(3.0, 8.0)
        color = np.random.dirichlet([1, 1, 1])
        for c in range(3):
            lensed_rgb[c] = add_gaussian_star(lensed_rgb[c], x_star, y_star, amp * color[c], sigma)

    if add_noise:
        for c in range(3):
            lensed_rgb[c] += np.random.normal(scale=0.01, size=(image_size, image_size))

    return np.clip(gt_rgb, 0, None), np.clip(lensed_rgb, 0, None)

# ========== MAIN EXECUTION ==========
if __name__ == '__main__':
    output_dir = r'C:\Users\mythi\.astropy\Code\COLOR\Colordata1'
    os.makedirs(output_dir, exist_ok=True)

    num_examples = 5
    fig, axs = plt.subplots(num_examples, 2, figsize=(6, 3 * num_examples))

    for i in range(num_examples):
        gt, lensed = generate_realistic_simulation_rgb()
        filename = f'simulation_rgb7_{i:04d}.npz'
        np.savez(os.path.join(output_dir, filename), gt=gt, lensed=lensed)
        print(f"Saved simulation {i+1} of {num_examples}: {filename}")

        axs[i, 0].imshow(normalize_img(gt))
        axs[i, 0].set_title("Ground Truth (RGB)")
        axs[i, 1].imshow(normalize_img(lensed))
        axs[i, 1].set_title("Lensed + BG + Stars (RGB)")
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
