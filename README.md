# Burst Research Project

This repository contains research code and experiments for the seasonal burst detection in time series data. The project implements efficients algorithms for detecting and analyzing seasonal bursts in temporal data.

## Features

- Seasonal burst detection using multiple algorithms:
  - Exhaustive search for seasonal bursts with:
    - a generic aggregation function
    - fast computation exclusive to the sum
  - Heuristic search with skipping 
  - Spatial burst detection with naive parallelization
  - Online detection with incremental one pass algorithm


- Utility functions for:
  - Burst post-processing
  - Connected components analysis on gridded data
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
│   ├── Paris/                # Paris heat stress index (WBGT) and temperature data
│   └── mortality/            # Mortality data
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
- `ERA5_land/`: Contains ERA5 land data files, downloaded from the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).
- `SST/`: Contains Sea Surface Temperature data files, downloaded from the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).
- `WBGT/`: Contains WBGT data files, downloaded from the [Copernicus Climate Change Service](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).
- `mortality/`: Contains mortality data files, downloaded from the [Data gouv website](https://www.data.gouv.fr/fr/datasets/agregation-des-fichiers-des-personnes-decedees/).
- `power/`: Contains electricity consumption data files, downloaded from the [Data gouv website](https://www.data.gouv.fr/fr/datasets/consommation-quotidienne-brute/).


## Experiments

To run the experiments, run the performance evaluation script:

```bash
python run_experiments.py
```

This will execute the performance evaluation scripts and generate the results.
The results will be saved in the `experiments/results/` directory.
The results include:
- Scaling of all algorithms
- Tradeoff between exhaustive and heuristic search
- Numerical precision
- Performance of the online detection algorithm

Figures and tables can be generated with the notebook `experiments/paper_figures.ipynb`.

To run a specific experiment, run the correspoding command:
```bash 
python experiments/experiments/run_all.py --experiments <experiment_name>
```
where `<experiment_name>` is the name of the experiment you want to run.
The available experiments are:
  -  `all`: Run all experiments
  - `exhaustive_algo_scaling`: Scaling of the exhaustive algorithm
  - `numerical`: Numerical precision of the exhaustive algorithm
  - `skipping_tradeoff`: Tradeoff between exhaustive and heuristic search
  - `grid_skipping_tradeoff`: Tradeoff between exhaustive and heuristic search with grid skipping
  - `sum_computation`: Performance of the computation of the sum
  - `online_scaling`: Scaling of the online detection algorithm


## Case Studies

The case studies are located in the `case-studies/` directory. Each case study includes Jupyter notebooksto analyze the data and apply the burst detection algorithms. The case studies include:
- `mediterranean/`: Analysis of the Mediterranean Sea Surface Temperature data. The notebook includes:
  - `averaged_sst.ipynb`: Analysis of the averaged SST data
  - `grid_sst.ipynb`: Analysis of the gridded SST data
  - `cluster_grid.ipynb`: Visualization of the clusters' contours
- `heat_stress/`: Analysis of the Paris heat stress index (WBGT) data
- `mortality/`: Analysis of the link between temperature or stress bursts and mortality
- `power/`: Analysis of the national electricity consumption data


## Citation

If you use this work in your research, please cite:

- Algorithm paper
    > Guillaume Coulaud, Audrey Brouillet, Reza Akbarinia, Florent Masseglia, Dennis Shasha. Efficient Detection of Seasonal Bursts: Applications to Climate Time Series. DEXA 2026 - 37th International Conference on Database and Expert Systems Applications, Aug 2026, Graz, Austria. pp.129-143. https://inria.hal.science/hal-05723497v1.
    ```bibtex
    @article{brouillet2025climburst,
      title={ClimBurst: A novel method to detect climatological anomalies over time and space},
      author={Brouillet, Audrey and Coulaud, Guillaume and Shasha, Dennis and Akbarinia, Reza and Masseglia, Florent},
      journal={Geophysical Research Letters},
      volume={52},
      number={19},
      pages={e2025GL117095},
      year={2025},
      publisher={Wiley Online Library}
    }
    ```
- Athmospheric application paper
    > Brouillet, A., Coulaud, G., Shasha, D., Akbarinia, R., & Masseglia, F. (2025). ClimBurst: A novel method to detect climatological anomalies over time and space. Geophysical Research Letters, 52(19), e2025GL117095. https://inria.hal.science/hal-05379952v1.
    ```bibtex
    @article{brouillet2025climburst,
      title={ClimBurst: A novel method to detect climatological anomalies over time and space},
      author={Brouillet, Audrey and Coulaud, Guillaume and Shasha, Dennis and Akbarinia, Reza and Masseglia, Florent},
      journal={Geophysical Research Letters},
      volume={52},
      number={19},
      pages={e2025GL117095},
      year={2025},
      publisher={Wiley Online Library}
    }
    ```
---

## License

This project is licensed under the [CeCILL Free Software License Agreement](LICENSE) (compatible with GNU GPL).
