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
