# Burst Research Project

This repository contains research code and experiments for burst detection in time series data. The project implements various algorithms for detecting and analyzing seasonal bursts in temporal data.

## Features

- Seasonal burst detection using multiple algorithms:
  - Exhaustive search for seasonal bursts
  - Heuristic search with skipping 
  - Spatial burst detection with naive parallelization
  - Online detection with Welford's method

- Utility functions for:
  - Burst post-processing
  - Connected components analysis
  - Statistical wavelet transform
  - Visualization tools

## Installation

### Using Conda (Recommended)

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate seasonal-burst
```

2. Install the package in development mode:
```bash
pip install -e .
```

For development, install additional dependencies:
```bash
pip install -e ".[dev]"
```

### Manual Installation

If you prefer not to use conda, you can install directly with pip:

```bash
python -m pip install .
```

## Project Structure

```
.
├── core/                     # Core implementation
│   ├── src/                  # Source code
│   │   ├── seasonal_burst/   # Burst detection algorithms
│   │   └── utils/            # Utility functions
│   └── test/                 # Test suite
├── docs/                     # Documentation
├── examples/                 # Example notebooks and scripts
├── grl-case-study/           # GRL paper case study
├── icdm-paper/               # ICDM paper materials
└── sigmod-demo/              # SIGMOD demonstration
```

## Development

The project uses modern Python packaging with `pyproject.toml`. The core package is located in `core/src/` and includes:

- `seasonal_burst/`: Main implementation of burst detection algorithms
- `utils/`: Utility functions and helper modules
- `tests/`: Tests of the main algorithms

### Running Tests

```bash
pytest core/test
```

### Code Style

The project uses Black for code formatting and isort for import sorting. To format the code:

```bash
black core/src
isort core/src
```