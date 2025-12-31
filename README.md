# AstroRIM
**Physics-Informed Inversion for Strong Gravitational Lensing (RIM + differentiable forward operator)**

AstroRIM is an end-to-end pipeline for **gravitational lens inversion**: recovering an **unlensed source-plane image** from a **lensed observation** using a **Recurrent Inference Machine (RIM)** jointly trained with a **learned, differentiable, physics-informed forward lensing operator**.

This repository contains:
- Simulation generators (Lenstronomy-based) for creating synthetic lens/source pairs.
- Inference/evaluation tooling (SSIM/MSE, FITS I/O, batch evaluation, residuals).
- Real-lens preprocessing utilities (normalization, centering, enhancement).
- Diagnostic/analysis tooling for producing mass-profile plots and summary figures.
- Example images and artifacts used in the accompanying paper/report.

> **Author:** Jack Walsh  
> **Contact:** 20jwalsh@greystonescollege.ie  
> **School:** Greystones Community College

---

## Why AstroRIM?
Traditional lens inversion (especially outside rigid parametric models) is a difficult inverse problem: many source configurations can match the same observation. AstroRIM addresses this by:
- Learning iterative inference updates (RIM) *and*
- Learning/optimizing the forward operator (a differentiable lensing operator) **jointly**, so gradient flow is consistent end-to-end.

From the current project summary, AstroRIM was trained on large synthetic datasets with composite lens models (SIE + NFW + SIS + external shear) and extended Sérsic sources at 96×96 resolution, achieving strong SSIM/MSE performance on held-out synthetic data. *(See paper/report for full details.)*

---

## Repository layout

AstroRim/
├─ Code/ # scripts: simulation, evaluation, preprocessing, diagnostics/figures
├─ Models/ # trained PyTorch checkpoints (.pt) and/or model artifacts
├─ image_dump/ # sample figures/images used in the writeup
├─ LICENSE
└─ README.md

yaml
Copy code

> **Note:** scripts are research-focused and evolve quickly. If you’ve renamed local scripts, just adjust the paths in the commands below.

---

## Requirements

### Core dependencies
You’ll typically need:
- Python 3.9+ (3.10+ recommended)
- PyTorch (CUDA optional but recommended)
- NumPy, SciPy, Matplotlib
- Astropy (FITS + cosmology utilities)
- scikit-image (metrics + preprocessing)
- Lenstronomy (simulation / lens model building)

### Optional (used by some analysis tooling)
- tqdm (progress bars)
- pandas (CSV export / tables)
- h5py (HDF5 export)
- pyyaml (YAML config support)
- joblib (caching)
- scikit-learn (some utilities)

---

## Installation

### Option A — pip + venv (simple)
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

python -m pip install --upgrade pip

# Install core packages
pip install numpy scipy matplotlib astropy scikit-image lenstronomy

# Install PyTorch (choose the correct command for your OS/CUDA)
# See: https://pytorch.org/get-started/locally/

# Optional extras
pip install tqdm pandas h5py pyyaml joblib scikit-learn
Option B — conda (recommended if you hit binary issues)
bash
Copy code
conda create -n astrorim python=3.10 -y
conda activate astrorim

conda install -c conda-forge numpy scipy matplotlib astropy scikit-image -y
pip install lenstronomy

# PyTorch via conda (pick CUDA build as appropriate)
# https://pytorch.org/get-started/locally/
Quickstart (typical workflow)
1) Generate simulations (synthetic dataset)
Simulation scripts live in Code/ and generally write FITS files containing:

a ground truth source (GT)

a lensed/observed image (LENSED/OBS)

metadata in FITS headers (lens/source params, etc.)

Most sim generators are configured near the top of the file (e.g., OUTPUT_DIR, number of realizations, PSF/noise settings). Example:

bash
Copy code
python Code/simgen_v2.py
If you have multiple sim generators (e.g., simgenv3.py, simgenv6.py), treat them as “profiles” for different regimes (strong/weak lensing, noise models, PSFs, etc.).

2) Run inference + evaluation (SSIM/MSE, recon dumps)
The evaluation script expects either a directory of FITS files or a single FITS file, plus your trained checkpoints.

Example:

bash
Copy code
python Code/Simulation_tests.py \
  --input-dir /path/to/fits_dir \
  --out-dir   /path/to/results \
  --model-path   Models/cond_rim_finetune_best.pt \
  --forward-path Models/cond_forward_finetune_best.pt \
  --device cuda \
  --n-iter 10 \
  --kernel-size 21
Helpful toggles:

--no-png disables per-file PNG outputs

--no-rescale disables output rescaling

--use-subhalos / --n-subhalos if your forward operator was trained with substructure

Outputs typically include:

reconstructed sources

metrics per file (SSIM/MSE)

optional per-case diagnostic figures

3) Lens/mass diagnostics + summary figures (optional)
There is an analysis/figure script designed to:

load reconstructions / convergence maps

compute radial mass profiles (cosmology-aware)

export figures, tables, and summary PDFs

Example usage pattern:

bash
Copy code
python Code/integrated_mastertest.py \
  --forward-model Models/cond_forward_finetune_best.pt \
  --input /path/to/recon_fits_dir \
  --output-dir /path/to/analysis_out \
  --z-lens 0.68 \
  --z-source 1.73 \
  --pixscale-method psf_fwhm \
  --assumed-seeing 0.8 \
  --summary-pdf
If your FITS reconstructions include a source-plane HDU, the analyzer can look for extensions like RECON, SRC, SOURCE, SOURCE_PLANE.

4) Preprocess real-lens FITS (normalization/centering/enhancement)
For survey/cutout FITS, preprocessing can help reduce background clutter and standardize inputs.

Example:

bash
Copy code
python Code/mass_normalizer.py \
  --input-dir /path/to/real_fits \
  --output-dir /path/to/processed \
  --target-size 96 \
  --denoise combined \
  --enhance adaptive_ring \
  --centering brightness_weighted
FITS format expectations (important)
The evaluation tooling is designed to be tolerant of “simulator version drift”, but best results come from consistent conventions:

Lensed/observed image stored in an HDU named: LENSED or OBS / OBSERVED

Ground truth source stored in an HDU named: GT

If names aren’t present, some scripts fall back to “Primary HDU is GT, next HDU is LENSED” patterns.

If something loads incorrectly, enable any available “debug FITS structure” flags in the scripts to print HDU names/shapes.

Reproducibility notes
Fix random seeds in sim generation if you want deterministic datasets.

Keep your simulation distribution consistent with training (lens types, PSF/noise, pixel scale), or you’ll amplify the synthetic-to-real gap.

If you’re evaluating on real lenses, document:

pixel scale assumptions

PSF estimates (FWHM/beta if Moffat)

any FITS conversion steps (PNG→FITS pipelines can introduce artifacts)

Citing
If you use this code in academic work, please cite the accompanying AstroRIM paper/report.

BibTeX (template):

bibtex
Copy code
@misc{walsh_astrorim_2025,
  title        = {AstroRIM: Physics-Informed Inversion for Strong Gravitational Lensing},
  author       = {Walsh, Jack},
  year         = {2025},
  howpublished = {\url{https://github.com/Mad-At-Line/AstroRim}},
  note         = {GitHub repository}
}
License
Apache-2.0 (see LICENSE).

Contributing / Issues
If you find a bug, reproducibility issue, or want to propose a feature:

Open an Issue with:

the command you ran

your input data description (shapes, HDUs, pixel scale)

error logs / screenshots

PRs are welcome, especially for:

pinned dependency versions

small example datasets

cleaner config-driven runs (instead of editing constants in scripts)

Acknowledgements
Built with major reliance on the open-source scientific Python ecosystem, especially:

PyTorch

Astropy

Lenstronomy

SciPy / NumPy / scikit-image
