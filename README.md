# MACE Multi-Model NEB Comparison for Pt Dissolution Pathways

This repository contains a Python script designed to perform a multi-model NEB (Nudged Elastic Band) comparison for the dissolution pathways of platinum (Pt). The script leverages various models and DFT (Density Functional Theory) calculations to analyze and compare different configurations.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Options](#options)
- [Directory Structure](#directory-structure)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Overview

The script performs the following tasks:
1. **Validation**: Checks the initial and final structures for each configuration.
2. **NEB Calculation**: Runs NEB calculations using multiple models.
3. **Comparison**: Compares the results from different models.
4. **Pathology Detection**: Identifies any issues or anomalies in the NEB pathways.
5. **DFT Refinement**: Optionally refines the NEB pathways using DFT calculations.
6. **Leaderboard Generation**: Generates a leaderboard summarizing model performance.

## Prerequisites

To run this script, you need:
- Python 3.x
- Required Python packages: `ase`, `numpy`, `pandas`, `matplotlib`, `scipy`, `torch`
- Access to computational resources for running NEB and DFT calculations (e.g., VASP or CP2K)

You can install the required packages using pip:

```bash
pip install ase numpy pandas matplotlib scipy torch
```

## Usage

### Running the Script

To run the script, use the following command:

```bash
python mace_neb_comparison.py --csv path/to/config.csv [options]
```

### Options

- `--csv`: Path to the configuration CSV file (default: `PATH_CSV`).
- `--validate-only`: Only run structure validation, skip NEB calculations.
- `--no-neb`: Skip endpoint relaxation and NEB optimization; instead reload previously-saved NEB results from disk.
- `--dft-only`: Implies `--no-neb` and runs straight through to DFT refinement + leaderboard.
- `--force-dft`: Re-run VASP NEB polish even for configs that already have a cached refined result on disk.
- `--dft-type`: DFT backend for the ground-truth refinement step (`vasp` or `cp2k`, default: `vasp`).
- `--cp2k-run-type`: CP2K run type (`sp` or `geo_opt`, default: `sp`).
- `--cp2k-geo-opt-steps`: Maximum iterations for CP2K geometry optimization (default: 100).
- `--write-dft-inputs`: Write DFT input files without launching a DFT job.
- `--read-dft-results`: Skip launching DFT entirely and read results from disk.
- `--no-dft`: Skip the DFT / VASP pathway refinement step entirely.
- `--endpoint-relax-mode`: Mode for endpoint relaxation (`per-model` or `shared`, default: `shared`).
- `--output-dir`: Root output directory (default: `OUTPUT_ROOT`).
- `--reference-model`: Model key to use for endpoint pre-relaxation and as the DFT-polish initial guess (default: `REFERENCE_MODEL_KEY`).
- `--only-models`: Comma-separated subset of model keys to run.
- `--skip-models`: Comma-separated model keys to exclude from this run.
- `--ignore-images`: Exclude specific NEB image indices from plots/RMSE/barrier scoring.

## Directory Structure

The script generates the following directory structure in the output directory:

```
OUTPUT_ROOT/
├── {model}/                ← NEB files for each model
│   ├── {config}/
│       └── {config}_neb.extxyz
├── comparison_plots/        ← overlay energy profiles + barrier bar charts
├── pathology_plots/         ← flagged image plots
├── dft_refinement/          ← VASP NEB polish per config
│   ├── {config}/
│       └── image_XX/
│           └── OUTCAR (or .out for CP2K)
└── reports/
    ├── neb_comparison_report.json
    ├── model_leaderboard.json
    ├── model_leaderboard.csv
    ├── pairwise_model_dft_comparison.csv
    ├── pairwise_model_dft_matrix.csv
    ├── per_image_breakdown/
    │   └── {config}_per_image_breakdown.csv
    └── timings.json
```

## Output

The script generates various reports and plots in the `reports` directory:

- **neb_comparison_report.json**: A machine-readable JSON report containing all results.
- **model_leaderboard.json**: A JSON file summarizing model performance.
- **model_leaderboard.csv**: A CSV file summarizing model performance.
- **pairwise_model_dft_comparison.csv**: A CSV file showing per-config method-vs-method energy RMSE/Δbarrier.
- **pairwise_model_dft_matrix.csv**: An aggregated mean RMSE matrix (all methods incl. DFT).
- **per_image_breakdown/**: Absolute-energy CSV files for each configuration and image.
- **timings.json**: Wall-clock time per phase.

