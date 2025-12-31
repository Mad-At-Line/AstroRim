# AstroRIM
**Physics-Informed Inversion for Strong Gravitational Lensing (Conditional RIM + differentiable forward operator)**

AstroRIM is an end-to-end pipeline for **gravitational lens inversion**: recovering an **unlensed source-plane image** from a **lensed observation** using a **Recurrent Inference Machine (RIM)** conditioned on a **learned, differentiable, physics-informed forward lensing operator**.

This repository contains:
- Simulation generators (Lenstronomy-based) for synthetic lens/source pairs
- Inference/evaluation tooling (SSIM/MSE, FITS I/O, batch evaluation, residuals)
- Real-lens preprocessing utilities (normalization, centering, enhancement)
- Diagnostic/analysis tooling (mass profiles, summary figures)
- Example figures/images used in the accompanying paper/report

> **Author:** Jack Walsh  
> **Contact:** 20jwalsh@greystonescollege.ie  
> **School:** Greystones Community College
AstroRIM: Complete Usage Instructions
Overview
AstroRIM is a gravitational lens reconstruction system that recovers unlensed source galaxies from observed lensed images. This guide covers the complete workflow from data preparation to final mass analysis.

Prerequisites
Required Dependencies
bashpip install torch torchvision numpy scipy matplotlib astropy lenstronomy scikit-image
```

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~20GB disk space for training data
- ~100 CPU-hours for full dataset generation (or ~37 GPU-hours for training)

---

## Workflow Overview
```
Raw Lens Images → Normalization → Model Inference → Source Reconstruction → Mass Analysis

Step 1: Generate Synthetic Training Data (Optional)
If you want to train your own model from scratch, generate synthetic lens/source pairs:
Run Individual Simulation Scripts
bash# Generate ~50k pairs from each simulator variant
cd Code/SimulationScripts

python simgenv1.py  # Foundation baseline with strong arcs
python simgenv2.py  # Enhanced physics, moderate complexity
python simgenv3.py  # Multi-instrument diversity
python simgenv4.py  # Expanded parameter space
python simgenv5.py  # Weak-lensing emphasis
python simgenv6.py  # Maximum observational realism
Expected Output

Each script generates paired FITS files:

source_XXXXX.fits - Ground truth unlensed sources
lensed_XXXXX.fits - Simulated lensed observations


Default output: Code/Training/synthetic_data/
Total size: ~12GB for 500k pairs

Simulation Parameters
Each variant uses different ranges (see paper Section 3.2.2):

Einstein radii: 0.3″ to 2.0″
Ellipticities: 0.0 to 0.7
PSF types: Gaussian, Moffat, Airy
Noise regimes: Space-like to ground-based


Step 2: Train the Model (Optional)
If using pre-trained checkpoints, skip to Step 3.
Initial Training Phase
bashcd Code/Training

# Train for 30 epochs on 250k pairs (200k train / 50k validation)
python trainer.py \
  --data_dir synthetic_data/ \
  --output_dir ../Models/ \
  --epochs 30 \
  --batch_size 8 \
  --lr_rim 5e-5 \
  --lr_forward 5e-6
Finetuning Phase
bash# Generate additional 250k pairs with new random seeds
cd ../SimulationScripts
python simgenv1.py --output_suffix _finetune
# ... repeat for v2-v6 ...

cd ../Training

# Finetune for 7 epochs with reduced learning rates
python finetuning.py \
  --data_dir synthetic_data_finetune/ \
  --checkpoint ../Models/cond_rim_initial.pt \
  --output_dir ../Models/ \
  --epochs 7 \
  --batch_size 8 \
  --lr_rim 5e-6 \
  --lr_forward 5e-7
```

### Training Outputs
- `cond_forward_finetune_best.pt` - Forward operator checkpoint (~350k params)
- `cond_rim_finetune_best.pt` - RIM checkpoint (~300k params)
- Training logs with MSE and SSIM metrics

### Expected Performance
- Validation MSE: ~3.7 × 10⁻⁴
- Validation SSIM: ~0.951
- Training time: ~37 GPU-hours on RTX 2080 Ti

---

## Step 3: Prepare Real Lens Images

### Input Requirements

Your lens images must be:
- **Format**: FITS files with valid WCS headers
- **Calibration**: Linear flux scale (ADU or electrons)
- **Size**: 96×96 pixels (or will be resized)
- **Content**: Single-band observations (F814W, r-band, etc.)

### Directory Structure
```
your_lens_data/
├── RXJ1131-1231.fits
├── SDSSJ1004+4112.fits
├── B1608+656.fits
└── ...
Normalize Images
bashcd Code/EvaluationAndUsage

python Real_Data_Normalizer.py \
  --input_dir ../../your_lens_data/ \
  --output_dir ../../normalized_lenses/ \
  --target_size 96 \
  --method robust
Normalization Methods

robust (recommended): Uses median/MAD for outlier resistance
minmax: Linear rescaling to [0, 1]
percentile: Clips to 1st-99th percentile, then rescales

Critical Warnings
⚠️ DO NOT use PNG/JPEG images converted to FITS

Non-linear gamma compression breaks photometric assumptions
Unknown PSF and resampling introduce systematic errors
See failure case LRG-3-757 in paper Section 4.3.1

⚠️ Ensure valid pixel scale in FITS headers

Check for CDELT1/CDELT2 or CD1_1/CD2_2 keywords
Required for physical mass conversions
Typical values: 0.04″/pix (HST), 0.2″/pix (ground-based)

Example: Check FITS Header
pythonfrom astropy.io import fits

hdul = fits.open('RXJ1131-1231.fits')
header = hdul[0].header

print(f"Pixel scale: {header.get('CDELT1', 'MISSING')} arcsec/pix")
print(f"Instrument: {header.get('INSTRUME', 'UNKNOWN')}")
print(f"Filter: {header.get('FILTER', 'UNKNOWN')}")

Step 4: Run Source Reconstruction
Basic Inference
bashcd Code/EvaluationAndUsage

python Reconstruction_Generator.py \
  --input_dir ../../normalized_lenses/ \
  --output_dir ../../reconstructions/ \
  --forward_model ../Models/cond_forward_finetune_best.pt \
  --rim_model ../Models/cond_rim_finetune_best.pt \
  --num_iterations 10 \
  --device cuda
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dir` | Directory with normalized FITS images | Required |
| `--output_dir` | Where to save reconstructions | Required |
| `--forward_model` | Path to forward operator checkpoint | Required |
| `--rim_model` | Path to RIM checkpoint | Required |
| `--num_iterations` | RIM unrolling steps (5-20) | 10 |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--batch_size` | Process multiple images simultaneously | 1 |

### Output Files

For each input `lens_name.fits`, produces:
```
reconstructions/
├── lens_name_source.fits          # Reconstructed source image
├── lens_name_forward.fits         # Forward-projected lensed image
├── lens_name_residual.fits        # Observation - Forward projection
├── lens_name_parameters.json      # Inferred lens parameters
└── lens_name_diagnostic.png       # Quick-look visualization
Inference Performance

GPU: ~50-100 ms per image (batch size 1)
CPU: ~300 ms per image
Memory: <2GB VRAM per image


Step 5: Generate Mass Analysis
Run Mass Profiling
bashcd Code/EvaluationAndUsage

python Mass_Analysis.py \
  --reconstruction_dir ../../reconstructions/ \
  --output_dir ../../mass_profiles/ \
  --lens_metadata lens_redshifts.csv \
  --cosmology Planck15
Lens Metadata File
Create lens_redshifts.csv with required information:
csvlens_name,z_lens,z_source,pixel_scale_arcsec
RXJ1131-1231,0.295,0.658,0.2045
SDSSJ1004+4112,0.680,1.734,0.4112
B1608+656,0.630,1.390,0.206
Cosmology Options

Planck15: H₀=67.7, Ωₘ=0.307 (default)
Planck18: H₀=67.4, Ωₘ=0.315
WMAP9: H₀=69.3, Ωₘ=0.286
FlatLambdaCDM: Custom H₀ and Ωₘ

Output Diagnostics
For each lens, generates comprehensive multi-panel figures:
Panel 1: Forward Model Check

Original observation
Forward-projected reconstruction
Residual map (observation - forward)
χ² map

Panel 2: Convergence Field

2D convergence κ(x,y) map
Einstein ring overlay
Color scale: κ=0 (blue) to κ>2 (red)

Panel 3: Radial Profile

κ(r) vs radius
Critical density line (κ=1)
Einstein radius marker

Panel 4: Azimuthal Variation

κ vs position angle at r = θ_E
Shows ellipticity/shear effects

Panel 5: Enclosed Mass

M(<r) vs radius
Einstein mass M(<θ_E) marker

Panel 6: Parameter Table

Nine inferred lens parameters
Derived quantities (θ_E, σ_v, masses)
Goodness-of-fit metrics

Physical Quantities Computed
QuantityFormulaUnitsEinstein radiusθ_E = b/qarcsecVelocity dispersionσ_v from SIEkm/sCritical surface densityΣ_crit(z_l, z_s)kg/m²Einstein massM(<θ_E) = π θ²_E Σ_crit D²_lM_☉

Step 6: Interpret Results
Quality Indicators
✅ Good Reconstruction

RMS residual < 0.05
Smooth, coherent source morphology
Forward projection matches observation
κ(r) crosses κ=1 at reasonable θ_E

⚠️ Possible Issues

Structured residuals (rings, arcs)
Negative flux in source
Discontinuous κ field
χ² concentrated on Einstein ring

Common Failure Modes
SymptomLikely CauseSolutionRing-shaped residualsPSF mismatchProvide PSF model or use PSF-matched dataHigh χ² everywhereNormalization issueCheck flux scale and backgroundOffset sourceWCS/pixel scale errorVerify FITS header keywordsNoisy κ mapInsufficient S/NBin data or use smoother regularization
Forward Consistency Check
The most important diagnostic is forward consistency:
pythonimport numpy as np
from astropy.io import fits

obs = fits.getdata('observation.fits')
fwd = fits.getdata('lens_name_forward.fits')
residual = obs - fwd

print(f"RMS residual: {np.sqrt(np.mean(residual**2)):.4f}")
print(f"Max |residual|: {np.max(np.abs(residual)):.4f}")
If RMS < 0.05 and residuals are randomly distributed, the lens model is likely correct.

Advanced Usage
Batch Processing
bash# Process entire survey catalog
python Reconstruction_Generator.py \
  --input_dir /data/euclid_lenses/ \
  --output_dir /results/euclid/ \
  --forward_model ../Models/cond_forward_finetune_best.pt \
  --rim_model ../Models/cond_rim_finetune_best.pt \
  --batch_size 16 \
  --device cuda \
  --num_workers 4
Custom RIM Iterations
bash# Fewer iterations = faster but less refined (weak lenses)
python Reconstruction_Generator.py ... --num_iterations 5

# More iterations = slower but higher quality (complex systems)
python Reconstruction_Generator.py ... --num_iterations 20
CPU-Only Inference
bash# For machines without GPU
python Reconstruction_Generator.py ... --device cpu
Export for MCMC Refinement
python# Use AstroRIM output as initial guess for Lenstronomy
import json

with open('lens_name_parameters.json') as f:
    params = json.load(f)

# Pass to traditional modeling pipeline
lens_model = [
    {'type': 'SIE', 'theta_E': params['b'], 'q': params['q'], 
     'phi': params['phi'], 'center_x': params['x0'], 'center_y': params['y0']},
    {'type': 'SHEAR_GAMMA_PSI', 'gamma': params['gamma'], 'psi': params['gamma_phi']},
    {'type': 'NFW', 'Rs': params['rs'], 'kappa_s': params['kappa_s']}
]

Troubleshooting
Issue: "CUDA out of memory"
bash# Reduce batch size
python Reconstruction_Generator.py ... --batch_size 1

# Or use CPU
python Reconstruction_Generator.py ... --device cpu
Issue: "KeyError: CDELT1"
python# Your FITS lacks pixel scale - add manually
from astropy.io import fits
hdul = fits.open('lens.fits', mode='update')
hdul[0].header['CDELT1'] = -0.04  # arcsec/pix (negative for RA)
hdul[0].header['CDELT2'] = 0.04
hdul.flush()
Issue: "Reconstruction is all zeros"

Check normalization didn't clip signal to zero
Verify input FITS has non-zero data
Try --method percentile instead of robust

Issue: "Mass values seem unrealistic"

Verify redshifts are correct (not swapped)
Check pixel scale in FITS header
Ensure cosmology matches your analysis


Citation
If you use AstroRIM in your research, please cite:
bibtex@article{walsh2025astrorim,
  title={AstroRIM: Joint Training of a Conditional Forward Operator with a Recurrent Inference Machine for Source Object Reconstruction in Gravitational Lens Systems},
  author={Walsh, Jack and Brennan, John and Regan, John and O'Sullivan, Creidhe},
  journal={Preprint},
  year={2025}
}

Support

Code Repository: https://github.com/Mad-At-Line/AstroRim
Issues: Open a GitHub issue for bugs or questions
Email: 20jwalsh@greystonescollege.ie


Quick Start Example
Complete minimal workflow:
bash# 1. Normalize your lens image
python Real_Data_Normalizer.py \
  --input_dir ./my_lenses/ \
  --output_dir ./normalized/

# 2. Run reconstruction
python Reconstruction_Generator.py \
  --input_dir ./normalized/ \
  --output_dir ./results/ \
  --forward_model ../Models/cond_forward_finetune_best.pt \
  --rim_model ../Models/cond_rim_finetune_best.pt

# 3. Generate mass analysis
python Mass_Analysis.py \
  --reconstruction_dir ./results/ \
  --output_dir ./diagnostics/ \
  --lens_metadata redshifts.csv

# 4. View results
open diagnostics/my_lens_diagnostic.png
That's it! You now have source reconstructions and mass profiles for your gravitational lenses.
