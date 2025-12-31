# AstroRIM

**Physics-Informed Inversion for Strong Gravitational Lensing (RIM + differentiable forward operator)**

AstroRIM is an end-to-end pipeline for **gravitational lens inversion** and **gravitational lens mass profiling**: recovering an **unlensed source-plane image** from a **lensed observation** using a **Recurrent Inference Machine (RIM)** jointly trained with a **learned, conditional, and differentiable, physics-paraneterized forward lensing operator**.

This repository contains:

* Simulation generators (Lenstronomy-based) for creating synthetic lens/source pairs.
* Inference/evaluation tooling (SSIM/MSE, FITS I/O, batch evaluation, residuals).
* Real-lens preprocessing utilities (normalization, centering, enhancement).
* Diagnostic/analysis tooling for producing mass-profile plots and summary figures.
* Example images and artifacts used in the accompanying paper/report.

> **Author:** Jack Walsh
> **Contact:** [20jwalsh@greystonescollege.ie](mailto:20jwalsh@greystonescollege.ie)
> **School:** Greystones Community College

---

## Why AstroRIM?

Traditional lens inversion (especially those who operate outside rigid parametric models) is a difficult inverse problem: many source configurations can match the same observation, and these models are often trained specific to one lens. AstroRIM addresses this by:

* Learning iterative inference updates (RIM) *and*
* Learning/optimizing the forward operator (a conditional and differentiable lensing operator) **jointly**, so gradient flow is consistent end-to-end.

AstroRIM was trained on large synthetic datasets with composite lens models (SIE + NFW + SIS + external shear) and extended Sérsic sources at 96×96 resolution, achieving strong SSIM/MSE performance on held-out synthetic data. *(See paper/report for full details.)*

---

> **Note:** scripts are research-focused and evolve quickly. If you’ve renamed local scripts, just adjust the paths in the commands below.

---

## Requirements

### Core dependencies

You’ll typically need:

* Python 3.9+ (3.10+ recommended)
* PyTorch (CUDA optional but recommended)
* NumPy, SciPy, Matplotlib
* Astropy (FITS + cosmology utilities)
* scikit-image (metrics + preprocessing)
* Lenstronomy (simulation / lens model building)

### Optional (used by some analysis tooling)

* tqdm (progress bars)
* pandas (CSV export / tables)
* h5py (HDF5 export)
* pyyaml (YAML config support)
* joblib (caching)
* scikit-learn (some utilities)

---

## Installation

### Option A - pip + venv

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
```

### Option B - Conda

```bash
conda create -n astrorim python=3.10 -y
conda activate astrorim

conda install -c conda-forge numpy scipy matplotlib astropy scikit-image -y
pip install lenstronomy

# PyTorch via conda (pick CUDA build as appropriate)
# https://pytorch.org/get-started/locally/
```

---

## Quickstart (typical workflow)

### 1) Generate simulations (synthetic dataset)

Simulation scripts live in `Code/` and generally write FITS files containing:

* a ground truth source (`GT`)
* a lensed/observed image (`LENSED`/`OBS`)
* metadata in FITS headers (lens/source params, etc.)

Most sim generators are configured near the top of the file (e.g., `OUTPUT_DIR`, number of realizations, PSF/noise settings). Example:

```bash
python Code/simgen_v2.py
```

If you have multiple sim generators (e.g., `simgenv3.py`, `simgenv6.py`), treat them as “profiles” for different regimes (strong/weak lensing, noise models, PSFs, etc.).

---

### 2) Run reconstruction inference + evaluation

Run the RIM inference over a directory of FITS files (or a single FITS). This step typically produces reconstructed source-plane outputs and per-file metrics (SSIM/MSE).

Example:

```bash
python Code/Simulation_tests.py \
  --input-dir /path/to/fits_dir \
  --out-dir /path/to/results \
  --model-path Models/cond_rim_finetune_best.pt \
  --forward-path Models/cond_forward_finetune_best.pt \
  --device cuda \
  --n-iter 10
```

Common optional flags (depending on your script version):

* `--no-png` : disable per-file PNG outputs
* `--no-rescale` : disable output rescaling
* `--use-subhalos` / `--n-subhalos` : include substructure if trained for it

Outputs typically include:

* reconstructed FITS (with a `RECON` HDU)
* SSIM/MSE summary logs and/or CSV
* optional diagnostic figures

---

### 3) Run mass-profile diagnostics + summary figures (optional)

Generate physics-motivated diagnostics from reconstructions (e.g., radial κ profiles, mass estimates, per-case summary panels).

Example:

```bash
python Code/integrated_mastertest.py \
  --forward-model Models/cond_forward_finetune_best.pt \
  --input /path/to/recon_fits_dir \
  --output-dir /path/to/analysis_out \
  --z-lens 0.68 \
  --z-source 1.73 \
  --pixscale-method psf_fwhm \
  --assumed-seeing 0.8 \
  --summary-pdf
```

---

### 4) Preprocess real-lens FITS (optional)

Preprocess survey/cutout FITS images to improve background suppression, centering, and consistency with the training distribution.

Example:

```bash
python Code/mass_normalizer.py \
  --input-dir /path/to/real_fits \
  --output-dir /path/to/processed \
  --target-size 96 \
  --denoise combined \
  --enhance adaptive_ring \
  --centering brightness_weighted
```

---

## FITS format expectations

The pipeline is tolerant of variation, but best results are obtained when:

* The lensed/observed image is stored in an HDU named `LENSED` or `OBS`
* The ground-truth source is stored in an HDU named `GT`

If HDU names are missing, some scripts fall back to common patterns (e.g., Primary HDU = GT, next HDU = LENSED).

---

## Reproducibility notes

* Fix random seeds in simulation generation for deterministic datasets.
* Keep simulation distributions consistent with training (lens population, noise/PSF, pixel scale) to reduce domain shift.
* For real lenses, document pixel scale assumptions, PSF estimates, and any FITS conversion steps.

---

## Citing

If you use this code in academic work, please cite the accompanying AstroRIM paper/report.

```bibtex
@misc{walsh_astrorim_software_2025,
  title        = {AstroRIM},
  author       = {Walsh, Jack and Brennan, John and Regan, John and {O'Sullivan}, Creidhe},
  year         = {2025},
  month        = dec,
  howpublished = {GitHub repository},
  url          = {https://github.com/Mad-At-Line/AstroRim/tree/af51ab4},
  note         = {Version: main @ af51ab4; accessed 2025-12-31}
}
```

---

## License

Apache-2.0 (see `LICENSE`).

---

## Contributing / Issues

Issues and PRs are welcome. If you open an issue, please include:

* the exact command you ran
* input data details (HDU names, shapes, pixel scale, and any other relevant information)
* error logs / screenshots

---

## Acknowledgements

Built with major reliance on the scientific Python ecosystem, especially:

* PyTorch
* Astropy
* Lenstronomy
* NumPy / SciPy / scikit-image
