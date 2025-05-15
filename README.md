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

- Case studies 
  - Mediterranean Sea Surface Temperature
  - Paris heat stress index (WBGT) 
  - Link to temperature or stress bursts to mortality 
  - National electricity consumption

- Performance evaluations
  - Scaling of all algorithms
  - Tradeoff between exhaustive and heuristic search
  - Numerical precision

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
├── Data/                     # Data files
│   ├── ERA5_land/            # ERA5 land data
│   ├── power/                # Power data
│   ├── SST/                  # Sea Surface Temperature data
├── docs/                     # Documentation
├── examples/                 # Example notebooks and scripts
├── experiments/              # Experiment scripts for performance evaluation
├── case-studies/             # Case studies
│   ├── mediterranean/        # Mediterranean Sea Surface Temperature
│   ├── paris/                # Paris heat stress index (WBGT)
│   ├── mortality/            # Link to temperature or stress bursts to mortality
│   └── electricity/          # National electricity consumption
├── environment.yml           # Conda environment file
├── pyproject.toml            # Python packaging configuration
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


## Data
The data used in the case studies is available in the `Data/` directory. The data files are organized into subdirectories based on the type of data.
- `ERA5_land/`: Contains ERA5 land data files, downloaded from the [Copernicus Climate Change Service](https://climate.copernicus.eu/).
- `SST/`: Contains Sea Surface Temperature data files, downloaded from the [Copernicus Climate Change Service](https://climate.copernicus.eu/).
- `WBGT/`: Contains WBGT data files, downloaded from the [WBGT website](https://www.wbgt.fr/).
- `mortality/`: Contains mortality data files, downloaded from the [INSEE website](https://www.insee.fr/en/accueil).
- `power/`: Contains electricity consumption data files, downloaded from the [RTE website](https://data.rte-france.com/).