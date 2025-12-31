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
