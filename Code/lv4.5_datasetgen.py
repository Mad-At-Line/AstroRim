import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from scipy.interpolate import griddata
import os
import random

def generate_realistic_simulation(image_size=64):
    """
    Generates a realistic gravitational lensing simulation with diverse lens models
    and 1 to 3 extended source galaxies, without artificial noise.
    """
    # Coordinate grid in arcseconds
    grid_lin = np.linspace(-2.5, 2.5, image_size)
    x_grid, y_grid = np.meshgrid(grid_lin, grid_lin)

    # Lens models used
    possible_lens_types = ['SIE', 'NFW', 'SIS', 'SHEAR']
    num_lenses = random.randint(1, 3)
    lens_types = random.sample(possible_lens_types, num_lenses)
    lens_model = LensModel(lens_model_list=lens_types)

    # Number of sources (1 to 3)
    num_sources = random.randint(1, 3)
    source_list = ['SERSIC_ELLIPSE'] * num_sources
    source_model = LightModel(light_model_list=source_list)

    # Lens parameters
    kwargs_lens = []
    for lens_type in lens_types:
        if lens_type == 'SIE':
            kwargs_lens.append({
                'theta_E': np.random.uniform(0.8, 1.8),
                'e1': np.random.uniform(-0.2, 0.2),
                'e2': np.random.uniform(-0.2, 0.2),
                'center_x': np.random.uniform(-0.2, 0.2),
                'center_y': np.random.uniform(-0.2, 0.2)
            })
        elif lens_type == 'NFW':
            kwargs_lens.append({
                'Rs': np.random.uniform(0.2, 0.6),
                'alpha_Rs': np.random.uniform(0.1, 1.0),
                'center_x': np.random.uniform(-0.2, 0.2),
                'center_y': np.random.uniform(-0.2, 0.2)
            })
        elif lens_type == 'SIS':
            kwargs_lens.append({
                'theta_E': np.random.uniform(0.5, 1.4),
                'center_x': np.random.uniform(-0.2, 0.2),
                'center_y': np.random.uniform(-0.2, 0.2)
            })
        elif lens_type == 'SHEAR':
            kwargs_lens.append({
                'gamma1': np.random.uniform(-0.1, 0.1),
                'gamma2': np.random.uniform(-0.1, 0.1)
            })

    # Source galaxy parameters
    kwargs_sources = []
    for _ in range(num_sources):
        kwargs_sources.append({
            'amp': np.random.uniform(2.5, 8.0),
            'R_sersic': np.random.uniform(0.05, 0.3),
            'n_sersic': np.random.uniform(1.0, 4.0),
            'e1': np.random.uniform(-0.4, 0.4),
            'e2': np.random.uniform(-0.4, 0.4),
            'center_x': np.random.uniform(-0.8, 0.8),
            'center_y': np.random.uniform(-0.8, 0.8)
        })

    # Generate source light profile
    gt = source_model.surface_brightness(x_grid, y_grid, kwargs_sources).astype(np.float32)

    # Ray-shooting
    x_src, y_src = lens_model.ray_shooting(x_grid.flatten(), y_grid.flatten(), kwargs_lens)
    gt_interp = griddata(
        points=(x_grid.flatten(), y_grid.flatten()),
        values=gt.flatten(),
        xi=(x_src.reshape(image_size, image_size), y_src.reshape(image_size, image_size)),
        method='linear',
        fill_value=0.0
    )
    lensed = np.nan_to_num(gt_interp)

    return gt, lensed

if __name__ == '__main__':
    output_dir = r'C:\Users\mythi\.astropy\Code\Hopes_and_dreams\lv4.5_dataset'
    os.makedirs(output_dir, exist_ok=True)

    num_simulations = 20000
    for i in range(num_simulations):
        gt, lensed = generate_realistic_simulation(image_size=64)
        filename = f'simulation_{i:04d}.npz'
        np.savez(os.path.join(output_dir, filename), gt=gt, lensed=lensed)
        print(f"Simulation {i + 1} of {num_simulations} completed.")
