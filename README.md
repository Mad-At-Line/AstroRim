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

---

## Table of contents
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [1) Generate simulations](#1-generate-simulations)
  - [2) Run inference + evaluation](#2-run-inference--evaluation)
  - [3) Run diagnostics / mass-profile figures](#3-run-diagnostics--mass-profile-figures)
  - [4) Preprocess real-lens FITS](#4-preprocess-real-lens-fits)
- [FITS format expectations](#fits-format-expectations)
- [Reproducibility notes](#reproducibility-notes)
- [Citing](#citing)
- [License](#license)
- [Contributing / Issues](#contributing--issues)
- [Acknowledgements](#acknowledgements)

---

---

## Requirements

### Core dependencies
- Python 3.9+ (3.10+ recommended)
- PyTorch (CUDA optional but recommended)
- NumPy, SciPy, Matplotlib
- Astropy (FITS + cosmology utilities)
- scikit-image (metrics + preprocessing)
- Lenstronomy (simulation / lens modeling)

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

# Core packages
pip install numpy scipy matplotlib astropy scikit-image lenstronomy

# Install PyTorch (choose the correct command for your OS/CUDA)
# https://pytorch.org/get-started/locally/

# Optional extras
pip install tqdm pandas h5py pyyaml joblib scikit-learn
