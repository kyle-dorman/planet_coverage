# Planet Coverage Analysis

This repository contains code for analyzing PlanetScope Dove and SkySat satellite coverage along coastal regions.

## Overview

The codebase supports:
- Querying Planet imagery via API
- Aggregating coverage statistics across space and time
- Analyzing revisit frequency and sampling patterns
- Visualizing results using notebooks and summary plots

## Requirements

- Conda (Miniconda recommended)
- Git

## Setup

### 1. Install Conda
Install Miniconda (recommended) or Anaconda:

- Miniconda: https://docs.anaconda.com/miniconda/install/
- Anaconda: https://docs.anaconda.com/anaconda/install/

On macOS (optional via Homebrew):
```bash
brew install --cask miniconda
```

### 2. Install Git
Check if Git is installed:
```bash
which git      # macOS/Linux
where git      # Windows
```

If not installed:
```bash
# macOS
brew install git

# Windows
conda install git
```

### 3. Clone the Repository
```bash
git clone git@github.com:kyledorman/planet_coverage.git
cd planet_coverage
```

### 4. Create Environment
```bash
conda env create -f environment.yml
conda activate planet_coverage
```

## Usage

### Run Analysis Notebook
Launch Jupyter Lab:
```bash
conda activate planet_coverage
jupyter lab --notebook-dir=notebooks --port=8893
```

Open and run:
```
notebooks/analysis.ipynb
```

This notebook queries processed data using DuckDB and generates figures used in the study.

## Code Formatting
```bash
conda activate planet_coverage
./lint.sh
```

## Updating Dependencies
```bash
conda env update --file environment.yml --prune
conda activate planet_coverage
```

## Notes

- Data processing relies on columnar storage (Parquet) and analytical query engines (DuckDB/Polars) for scalability.
- Spatial operations are performed using Python geospatial libraries.

## Disclaimer

This repository contains research code provided as-is for reproducibility and reference. It is not optimized as a production software package.
