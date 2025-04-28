import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.observation_api import SingleBand
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import util
from astropy.cosmology import FlatLambdaCDM
import os

def generate_realistic_simulation(image_size=64, z_lens=0.5, z_source=1.5, noise_level='realistic'):
    """
    Generates a realistic gravitational lensing simulation with multiple lens components 
    and between 1 to 3 extended sources.
    
    Parameters:
        image_size (int): Size of the output image in pixels
        z_lens (float): Redshift of the lens galaxy
        z_source (float): Redshift of the source galaxy
        noise_level (str): 'none', 'low', or 'realistic'
    
    Returns:
        source_image (np.ndarray): The ground truth (unlensed) source image
        lensed_image (np.ndarray): The lensed image with observational effects
        lensed_noiseless (np.ndarray): The lensed image without noise (for training)
        metadata (dict): Additional information about the simulation
    """
    # 1. Set up cosmology for physical scaling
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    # Angular diameter distances for proper scaling
    D_s = cosmo.angular_diameter_distance(z_source).value  # source distance
    D_l = cosmo.angular_diameter_distance(z_lens).value    # lens distance
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value  # lens-source distance
    
    # Einstein radius scaling factor
    scale_factor = D_ls / (D_l * D_s)
    
    # 2. Define more realistic lens models with proper scaling
    lens_types = ['SIE', 'NFW', 'SHEAR_GAMMA_PSI']
    lens_model = LensModel(lens_types)
    
    # 3. Define source models - galaxies follow Sersic profiles
    num_sources = np.random.randint(1, 4)
    source_list = ['SERSIC_ELLIPSE'] * num_sources
    source_model = LightModel(source_list)
    
    # 4. Create a coordinate grid
    # Use smaller pixel scale for better resolution (0.05 arcsec typical for HST)
    pixel_scale = 0.05  # arcsec per pixel
    
    # Total field of view in arcsec
    fov = pixel_scale * image_size  
    grid_lin = np.linspace(-fov/2, fov/2, image_size)
    x_grid, y_grid = np.meshgrid(grid_lin, grid_lin)
    
    # 5. Set realistic lens parameters
    kwargs_lens = []
    
    # Primary lens: Singular Isothermal Ellipsoid (massive galaxy)
    # Einstein radius typically 0.5-2 arcsec for galaxy lenses
    theta_E_main = np.random.uniform(0.7, 1.8)
    kwargs_lens.append({
        'theta_E': theta_E_main,  # Einstein radius in arcsec
        'e1': np.random.uniform(-0.3, 0.3),  # ellipticity component
        'e2': np.random.uniform(-0.3, 0.3),  # ellipticity component
        'center_x': np.random.uniform(-0.1, 0.1),  # slight offset from center
        'center_y': np.random.uniform(-0.1, 0.1)
    })
    
    # Secondary lens: NFW profile (dark matter halo)
    # Rs is typically 10-30 times the Einstein radius for galaxy-scale lenses
    rs_scale = np.random.uniform(10, 30) * theta_E_main
    kwargs_lens.append({
        'Rs': rs_scale,  # scale radius
        # alpha_Rs properly scaled according to cosmology and mass
        'alpha_Rs': theta_E_main * np.random.uniform(0.05, 0.15),
        'center_x': kwargs_lens[0]['center_x'],  # centered with main galaxy
        'center_y': kwargs_lens[0]['center_y']   # centered with main galaxy
    })
    
    # External shear (from large scale structure or nearby galaxies)
    kwargs_lens.append({
        'gamma1': np.random.uniform(-0.05, 0.05),  # realistic external shear magnitude
        'gamma2': np.random.uniform(-0.05, 0.05),
        # shear center is at the lens position by definition
        'ra_0': kwargs_lens[0]['center_x'],
        'dec_0': kwargs_lens[0]['center_y']
    })
    
    # 6. Set realistic source parameters for each source galaxy
    kwargs_sources = []
    
    # Main source always close to optical axis for strong lensing
    main_source_x = np.random.uniform(-0.3, 0.3)
    main_source_y = np.random.uniform(-0.3, 0.3)
    
    # First source (usually the brightest)
    kwargs_sources.append({
        'amp': np.random.uniform(8, 15),  # brightness
        'R_sersic': np.random.uniform(0.1, 0.5),  # effective radius (arcsec)
        'n_sersic': np.random.uniform(0.8, 4.0),  # Sersic index: disk (n=1) to bulge (n=4)
        'e1': np.random.uniform(-0.5, 0.5),  # ellipticity
        'e2': np.random.uniform(-0.5, 0.5),  # ellipticity
        'center_x': main_source_x,
        'center_y': main_source_y
    })
    
    # Additional sources (if any)
    for i in range(1, num_sources):
        # Additional sources either clustered with main source or more distant
        if np.random.random() < 0.7:  # 70% chance of clustered source
            offset = np.random.uniform(0.1, 0.4)
            angle = np.random.uniform(0, 2*np.pi)
            dx, dy = offset * np.cos(angle), offset * np.sin(angle)
            source_x = main_source_x + dx
            source_y = main_source_y + dy
        else:  # 30% chance of more distant source
            source_x = np.random.uniform(-0.8, 0.8)
            source_y = np.random.uniform(-0.8, 0.8)
        
        # Sources are typically fainter than the main source
        kwargs_sources.append({
            'amp': np.random.uniform(3, 8),
            'R_sersic': np.random.uniform(0.1, 0.3),
            'n_sersic': np.random.uniform(0.8, 4.0),
            'e1': np.random.uniform(-0.5, 0.5),
            'e2': np.random.uniform(-0.5, 0.5),
            'center_x': source_x,
            'center_y': source_y
        })
    
    # 7. Generate the ground truth (unlensed) source image
    source_image = source_model.surface_brightness(x_grid, y_grid, kwargs_sources)
    
    # 8. Use lenstronomy's ray-shooting for more accurate lensing
    ra_grid, dec_grid = x_grid.flatten(), y_grid.flatten()
    beta_ra, beta_dec = lens_model.ray_shooting(ra_grid, dec_grid, kwargs_lens)
    
    # 9. Calculate magnification (determinant of the distortion matrix)
    mag_map = lens_model.magnification(ra_grid, dec_grid, kwargs_lens)
    mag_map = mag_map.reshape(image_size, image_size)
    
    # 10. Map source light to lens plane (more accurate method)
    lensed_noiseless = np.zeros_like(x_grid)
    for i in range(len(kwargs_sources)):
        kwargs_source = [kwargs_sources[i]]
        # Evaluate source brightness at the traced-back positions
        source_light = source_model.surface_brightness(beta_ra, beta_dec, kwargs_source)
        source_light = source_light.reshape(image_size, image_size)
        lensed_noiseless += source_light
    
    # 11. Add observational effects if requested
    if noise_level != 'none':
        # Define realistic PSF (seeing) and detector properties
        if noise_level == 'realistic':
            # Generate a realistic PSF (typical ground-based telescope)
            seeing = np.random.uniform(0.8, 1.2)  # seeing in arcsec
            psf_pixels = seeing / pixel_scale
            psf_kernel = util.gaussian_kernel(num_pix=21, delta_pix=pixel_scale, fwhm=seeing)
            psf = PSF(psf_kernel)
            
            # Observational parameters (exposure time, background, etc.)
            lsst_param = LSST()  # Using LSST parameters as example
            observation = SingleBand(**lsst_param)
            
            # Create the simulation API
            sim = SimAPI(numpix=image_size, kwargs_single_band=lsst_param)
            
            # Apply PSF and add noise
            lensed_image = sim.simulate(lensed_noiseless, 
                                       psf_class=psf, 
                                       add_poisson=True, 
                                       add_background=True)
        else:  # low noise case
            # Simple Gaussian PSF with minimal noise
            psf_pixels = 2  # 2-pixel FWHM PSF
            psf_kernel = util.gaussian_kernel(num_pix=11, delta_pix=pixel_scale, fwhm=psf_pixels*pixel_scale)
            psf = PSF(psf_kernel)
            
            # Convolve with PSF
            lensed_image = util.image_util.re_size_convolve(lensed_noiseless, psf_kernel)
            
            # Add minimal noise (high SNR)
            noise = np.random.normal(0, 0.01 * np.max(lensed_image), size=lensed_image.shape)
            lensed_image += noise
    else:
        lensed_image = lensed_noiseless
    
    # 12. Return normalized images
    source_image = source_image / np.max(source_image) if np.max(source_image) > 0 else source_image
    lensed_noiseless = lensed_noiseless / np.max(lensed_noiseless) if np.max(lensed_noiseless) > 0 else lensed_noiseless
    lensed_image = lensed_image / np.max(lensed_image) if np.max(lensed_image) > 0 else lensed_image
    
    # Collect metadata
    metadata = {
        'z_lens': z_lens,
        'z_source': z_source,
        'noise_level': noise_level,
        'num_sources': num_sources,
        'einstein_radius': theta_E_main,
        'lens_params': kwargs_lens,
        'source_params': kwargs_sources
    }
    
    return source_image.astype(np.float32), lensed_image.astype(np.float32), lensed_noiseless.astype(np.float32), metadata

def visualize_simulation(source, lensed, lensed_noiseless=None, metadata=None, title="Gravitational Lensing Simulation"):
    """
    Visualize a source and its lensed image side by side
    
    Parameters:
        source: Source galaxy image
        lensed: Lensed image with observational effects
        lensed_noiseless: Optional noiseless lensed image
        metadata: Optional dictionary with simulation parameters
        title: Figure title
    """
    if lensed_noiseless is None:
        # 2 panel display: source and lensed image
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Source image
        im0 = axes[0].imshow(source, origin='lower', cmap='viridis')
        axes[0].set_title("Source")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Lensed image
        im1 = axes[1].imshow(lensed, origin='lower', cmap='viridis')
        axes[1].set_title("Lensed Image")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        # 3 panel display: source, lensed without noise, and lensed with noise
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Source image
        im0 = axes[0].imshow(source, origin='lower', cmap='viridis')
        axes[0].set_title("Source")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Lensed image without noise
        im1 = axes[1].imshow(lensed_noiseless, origin='lower', cmap='viridis')
        axes[1].set_title("Lensed (No Noise)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Lensed image with noise
        im2 = axes[2].imshow(lensed, origin='lower', cmap='viridis')
        axes[2].set_title("Lensed (With Noise)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add metadata as text if available
    if metadata is not None:
        info_text = f"z_lens: {metadata['z_lens']:.2f}, z_source: {metadata['z_source']:.2f}\n"
        info_text += f"Einstein radius: {metadata['einstein_radius']:.2f} arcsec\n"
        info_text += f"Noise level: {metadata['noise_level']}, Sources: {metadata['num_sources']}"
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def display_examples(num_examples=5, image_size=64):
    """
    Generate and display a set of example simulations
    
    Parameters:
        num_examples: Number of examples to generate and display
        image_size: Size of the simulation images in pixels
    """
    # Create a figure with subplots arranged in rows
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
    
    # Generate different types of simulations
    noise_levels = ['none', 'low', 'realistic']
    z_lens_values = np.linspace(0.3, 0.9, num_examples)
    z_source_values = np.linspace(1.0, 2.5, num_examples)
    
    for i in range(num_examples):
        # Generate simulation with variety
        noise_level = noise_levels[i % len(noise_levels)]
        z_lens = z_lens_values[i]
        z_source = z_source_values[i]
        
        source, lensed, lensed_noiseless, metadata = generate_realistic_simulation(
            image_size=image_size,
            z_lens=z_lens,
            z_source=z_source,
            noise_level=noise_level
        )
        
        # Display source image
        im0 = axes[i, 0].imshow(source, origin='lower', cmap='viridis')
        axes[i, 0].set_title(f"Source (z={z_source:.2f})")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Display lensed image without noise
        im1 = axes[i, 1].imshow(lensed_noiseless, origin='lower', cmap='viridis')
        axes[i, 1].set_title(f"Lensed (No Noise)")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Display lensed image with noise
        im2 = axes[i, 2].imshow(lensed, origin='lower', cmap='viridis')
        axes[i, 2].set_title(f"Lensed ({noise_level.capitalize()} Noise)")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Add metadata text to last column
        info_text = f"z_lens: {metadata['z_lens']:.2f}, θE: {metadata['einstein_radius']:.2f} arcsec\n"
        info_text += f"Sources: {metadata['num_sources']}, e1: {metadata['lens_params'][0]['e1']:.2f}, e2: {metadata['lens_params'][0]['e2']:.2f}"
        axes[i, 2].text(1.05, 0.5, info_text, transform=axes[i, 2].transAxes, 
                      fontsize=9, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle("Gravitational Lensing Simulation Examples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig

if __name__ == '__main__':
    # Create output directory
    output_dir = r'C:\Users\mythi\.astropy\Code\Hopes_and_dreams\data_lv5'
    image_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Display 5 example simulations when running the script
    print("Generating example simulations...")
    example_fig = display_examples(num_examples=5, image_size=64)
    plt.savefig(os.path.join(image_dir, 'example_simulations.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Ask if user wants to proceed with generating the full dataset
    proceed = input("\nDo you want to proceed with generating the full dataset? (y/n): ")
    
    if proceed.lower() in ['y', 'yes']:
        # Generate simulations
        num_simulations = 20000  # Adjust as needed
        save_images = False      # Set to True to save visualizations
        
        # Add variety by varying redshifts and noise levels
        z_lens_options = np.random.uniform(0.2, 0.9, num_simulations)
        z_source_options = np.random.uniform(1.0, 3.0, num_simulations)
        noise_options = ['none', 'low', 'realistic']
        
        for i in range(num_simulations):
            # Pick random redshifts and noise level for variety
            z_lens = z_lens_options[i]
            z_source = z_source_options[i]
            noise_level = noise_options[i % len(noise_options)]
            
            # Generate simulation
            source, lensed, lensed_noiseless, metadata = generate_realistic_simulation(
                image_size=64, 
                z_lens=z_lens, 
                z_source=z_source,
                noise_level=noise_level
            )
            
            # Save data
            filename = f'simulation_{i:05d}.npz'
            np.savez(
                os.path.join(output_dir, filename), 
                gt=source, 
                lensed=lensed,
                lensed_noiseless=lensed_noiseless,
                metadata=metadata
            )
            
            # Optional: Save visualization
            if save_images and i % 100 == 0:  # Save every 100th image to avoid too many files
                fig = visualize_simulation(source, lensed, lensed_noiseless, metadata, 
                                         f"Simulation {i} (z_lens={z_lens:.2f}, z_source={z_source:.2f})")
                plt.savefig(os.path.join(image_dir, f'vis_{i:05d}.png'), dpi=150)
                plt.close(fig)
                
            if (i+1) % 100 == 0:
                print(f"Simulation {i+1} of {num_simulations} completed.")
    else:
        print("Dataset generation cancelled.")