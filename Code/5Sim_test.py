import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from scipy.interpolate import griddata
import random

def generate_realistic_simulation(image_size=64):
    """
    Generates a realistic gravitational lensing simulation with diverse lens models
    and 1 to 3 extended source galaxies, without artificial noise.
    
    Returns:
        gt (np.ndarray): The ground truth (unlensed) source image
        lensed (np.ndarray): The lensed image
        metadata (dict): Dictionary containing lens and source information
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
    lens_info = []
    for lens_type in lens_types:
        if lens_type == 'SIE':
            theta_E = np.random.uniform(0.8, 1.8)
            e1 = np.random.uniform(-0.2, 0.2)
            e2 = np.random.uniform(-0.2, 0.2)
            center_x = np.random.uniform(-0.2, 0.2)
            center_y = np.random.uniform(-0.2, 0.2)
            kwargs_lens.append({
                'theta_E': theta_E,
                'e1': e1,
                'e2': e2,
                'center_x': center_x,
                'center_y': center_y
            })
            lens_info.append(f"SIE: θE={theta_E:.2f}, e1={e1:.2f}, e2={e2:.2f}")
        elif lens_type == 'NFW':
            Rs = np.random.uniform(0.2, 0.6)
            alpha_Rs = np.random.uniform(0.1, 1.0)
            center_x = np.random.uniform(-0.2, 0.2)
            center_y = np.random.uniform(-0.2, 0.2)
            kwargs_lens.append({
                'Rs': Rs,
                'alpha_Rs': alpha_Rs,
                'center_x': center_x,
                'center_y': center_y
            })
            lens_info.append(f"NFW: Rs={Rs:.2f}, α={alpha_Rs:.2f}")
        elif lens_type == 'SIS':
            theta_E = np.random.uniform(0.5, 1.4)
            center_x = np.random.uniform(-0.2, 0.2)
            center_y = np.random.uniform(-0.2, 0.2)
            kwargs_lens.append({
                'theta_E': theta_E,
                'center_x': center_x,
                'center_y': center_y
            })
            lens_info.append(f"SIS: θE={theta_E:.2f}")
        elif lens_type == 'SHEAR':
            gamma1 = np.random.uniform(-0.1, 0.1)
            gamma2 = np.random.uniform(-0.1, 0.1)
            kwargs_lens.append({
                'gamma1': gamma1,
                'gamma2': gamma2
            })
            lens_info.append(f"SHEAR: γ1={gamma1:.2f}, γ2={gamma2:.2f}")

    # Source galaxy parameters
    kwargs_sources = []
    source_info = []
    for i in range(num_sources):
        amp = np.random.uniform(2.5, 8.0)
        R_sersic = np.random.uniform(0.05, 0.3)
        n_sersic = np.random.uniform(1.0, 4.0)
        e1 = np.random.uniform(-0.4, 0.4)
        e2 = np.random.uniform(-0.4, 0.4)
        center_x = np.random.uniform(-0.8, 0.8)
        center_y = np.random.uniform(-0.8, 0.8)
        
        kwargs_sources.append({
            'amp': amp,
            'R_sersic': R_sersic,
            'n_sersic': n_sersic,
            'e1': e1,
            'e2': e2,
            'center_x': center_x,
            'center_y': center_y
        })
        source_info.append(f"Source {i+1}: R={R_sersic:.2f}, n={n_sersic:.1f}, pos=({center_x:.2f},{center_y:.2f})")

    # Generate source light profile
    gt = source_model.surface_brightness(x_grid, y_grid, kwargs_sources).astype(np.float32)

    # Calculate magnification map
    mag_map = np.zeros_like(x_grid)
    try:
        mag = lens_model.magnification(x_grid.flatten(), y_grid.flatten(), kwargs_lens)
        mag_map = mag.reshape(image_size, image_size)
    except:
        pass  # Some lens configurations may cause numerical issues

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
    
    # Normalize images for better visualization
    gt = gt / np.max(gt) if np.max(gt) > 0 else gt
    lensed = lensed / np.max(lensed) if np.max(lensed) > 0 else lensed
    
    # Metadata for display
    metadata = {
        'lens_models': lens_types,
        'num_sources': num_sources,
        'lens_info': lens_info,
        'source_info': source_info
    }

    return gt, lensed, mag_map, metadata

def display_examples(num_examples=5, image_size=64):
    """
    Generate and display multiple examples of gravitational lensing simulations
    
    Parameters:
        num_examples: Number of examples to display
        image_size: Size of the simulation images
    """
    # Create a figure with 3 columns (source, magnification, lensed)
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
    
    for i in range(num_examples):
        # Generate a new simulation
        source, lensed, mag_map, metadata = generate_realistic_simulation(image_size)
        
        # Display source image
        im0 = axes[i, 0].imshow(source, origin='lower', cmap='inferno')
        axes[i, 0].set_title("Source Galaxy")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Display magnification map (log scale for better visualization)
        mag_map = np.abs(mag_map)  # Absolute value of magnification
        # Apply log scaling with clipping to avoid zeros/infinities
        mag_map = np.log10(np.clip(mag_map, 0.1, 1000))
        im1 = axes[i, 1].imshow(mag_map, origin='lower', cmap='viridis')
        axes[i, 1].set_title("Magnification (log)")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Display lensed image
        im2 = axes[i, 2].imshow(lensed, origin='lower', cmap='inferno')
        axes[i, 2].set_title("Lensed Image")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Add metadata text
        lens_models_str = ", ".join(metadata['lens_models'])
        info_text = f"Models: {lens_models_str}\n"
        info_text += "Lens properties:\n"
        for info in metadata['lens_info']:
            info_text += f"  • {info}\n"
        info_text += f"Sources: {metadata['num_sources']}\n"
        for info in metadata['source_info']:
            info_text += f"  • {info}\n"
            
        axes[i, 2].text(1.05, 0.5, info_text, transform=axes[i, 2].transAxes,
                      fontsize=9, verticalalignment='center', 
                      bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle("Gravitational Lensing Simulation Examples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.85)
    return fig

if __name__ == '__main__':
    # Just display 5 examples without saving any files
    print("Generating and displaying 5 gravitational lensing examples...")
    fig = display_examples(num_examples=5, image_size=64)
    plt.show()
