AstroRIM
Physics-Informed Inversion for Strong Gravitational Lensing (Conditional RIM + differentiable forward operator)

AstroRIM is an end-to-end pipeline for gravitational lens inversion: recovering an unlensed source-plane image from a lensed observation using a Recurrent Inference Machine (RIM) conditioned on a learned, differentiable, physics-informed forward lensing operator.

Author: Jack Walsh
Contact: 20jwalsh@greystonescollege.ie
School: Greystones Community College

Table of contents
Requirements

Installation

Quickstart

1) Generate simulations

2) Run inference + evaluation

3) Run diagnostics / mass-profile figures

4) Preprocess real-lens FITS

FITS format expectations

Reproducibility notes

Citing

License

Contributing / Issues

Acknowledgements

Requirements
Core dependencies
Python 3.9+ (3.10+ recommended)

PyTorch (CUDA optional but recommended)

NumPy, SciPy, Matplotlib

Astropy (FITS + cosmology utilities)

scikit-image (metrics + preprocessing)

Lenstronomy (simulation / lens modeling)

Optional (used by some analysis tooling)
tqdm (progress bars)

pandas (CSV export / tables)

h5py (HDF5 export)

pyyaml (YAML config support)

joblib (caching)

scikit-learn (some utilities)

Installation
Option A — pip + venv (simple)
bash
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate

python -m pip install --upgrade pip

# Core packages
pip install numpy scipy matplotlib astropy scikit-image lenstronomy

# Install PyTorch (choose the correct command for your OS/CUDA)
# https://pytorch.org/get-started/locally/

# Optional extras
pip install tqdm pandas h5py pyyaml joblib scikit-learn
Option B — conda (recommended if you hit binary issues)
bash
conda create -n astrorim python=3.10 -y
conda activate astrorim

conda install -c conda-forge numpy scipy matplotlib astropy scikit-image -y
pip install lenstronomy

# PyTorch via conda (pick CUDA build as appropriate)
# https://pytorch.org/get-started/locally/
Quickstart
1) Generate simulations
Simulation scripts live in Code/ and write FITS files containing:

GT: ground-truth source-plane image

LENSED / OBS: simulated observed (lensed) image

metadata in FITS headers (lens/source params, etc.)

Most sim generators are configured near the top of the file (e.g., OUTPUT_DIR, number of realizations, PSF/noise settings).

Example:

bash
python Code/simgen_v2.py
If you have multiple sim generators (e.g., simgenv3.py, simgenv6.py), treat them as “profiles” for different regimes (noise, PSF, lens distributions, etc.).

2) Run inference + evaluation
The evaluation script expects either a directory of FITS files or a single FITS file, plus your trained checkpoints.

Example:

bash
python Code/Simulation_tests.py \
  --input-dir /path/to/fits_dir \
  --out-dir   /path/to/results \
  --model-path   Models/cond_rim_finetune_best.pt \
  --forward-path Models/cond_forward_finetune_best.pt \
  --device cuda \
  --n-iter 10 \
  --kernel-size 21
Helpful toggles (if supported by your script version):

--no-png disables per-file PNG outputs

--no-rescale disables output rescaling

--use-subhalos / --n-subhalos if your forward operator was trained with substructure

Outputs typically include:

reconstructed source-plane FITS

per-file metrics (SSIM/MSE)

optional per-case diagnostic figures

3) Run diagnostics / mass-profile figures
A diagnostics / figure script can:

load reconstructions / κ maps (if present)

compute radial profiles and cosmology-aware mass conversions

export figures, tables, and summary PDFs

Example usage pattern:

bash
python Code/integrated_mastertest.py \
  --forward-model Models/cond_forward_finetune_best.pt \
  --input /path/to/recon_fits_dir \
  --output-dir /path/to/analysis_out \
  --z-lens 0.68 \
  --z-source 1.73 \
  --pixscale-method psf_fwhm \
  --assumed-seeing 0.8 \
  --summary-pdf
If your FITS reconstructions include a source-plane HDU, the analyzer may look for extensions like RECON, SRC, SOURCE, or SOURCE_PLANE.

4) Preprocess real-lens FITS
For survey/cutout FITS, preprocessing helps reduce background clutter and standardize inputs.

Example:

bash
python Code/mass_normalizer.py \
  --input-dir /path/to/real_fits \
  --output-dir /path/to/processed \
  --target-size 96 \
  --denoise combined \
  --enhance adaptive_ring \
  --centering brightness_weighted
FITS format expectations
The evaluation tooling is designed to be tolerant of “simulator version drift”, but best results come from consistent conventions:

Observed/lensed image stored in an HDU named: LENSED or OBS / OBSERVED

Ground truth source stored in an HDU named: GT

If names aren’t present, some scripts fall back to common patterns such as:

Primary HDU = GT, next HDU = LENSED

If something loads incorrectly, enable any available “debug FITS structure” flags in the scripts to print HDU names and shapes.

Reproducibility notes
Fix random seeds in sim generation if you want deterministic datasets.

Keep your simulation distribution consistent with training (lens types, PSF/noise, pixel scale), or you’ll amplify the synthetic-to-real gap.

For real lenses, document:

pixel scale assumptions

PSF estimates (FWHM/β if Moffat)

any FITS conversion steps (PNG→FITS pipelines can introduce artifacts)

Citing
If you use this code in academic work, please cite the accompanying AstroRIM paper/report.

BibTeX (template):

bibtex
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
Issues and PRs are welcome. If you open an Issue, please include:

the exact command you ran

your input data description (HDU names, shapes, pixel scale if known)

error logs and/or screenshots

Acknowledgements
Built with major reliance on the scientific Python ecosystem, especially:

PyTorch

Astropy

Lenstronomy

NumPy / SciPy / scikit-image
