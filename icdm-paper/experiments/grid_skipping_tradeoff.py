"""Experiment to evaluate the tradeoff between skipping and accuracy on grid data."""

import logging
from pathlib import Path
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xarray as xr

from skipping_tradeoff import (
    run_exhaustive_search,
    evaluate_skipping_strategy,
    bursts_to_days,
    precision,
    hit_precision
)

logger = logging.getLogger(__name__)


def run_grid_experiment(ts: np.ndarray, lat_idx: int, lon_idx: int, config: dict) -> Dict:
    """Run experiment for a single grid point.

    Parameters:
        ts: Time series data for the grid point
        lat_idx: Latitude index
        lon_idx: Longitude index
        config: Configuration dictionary

    Returns:
        Dictionary containing experiment results
    """
    # Extract parameters
    period_length = config['parameters']['period_length']
    n_period = len(ts) // period_length
    min_length = config['parameters']['min_length']
    max_length = config['parameters']['max_length']
    threshold = config['parameters']['threshold']
    skip_factors = config['parameters']['skip_factors']
    intra_period = config['data'].get('intra_period', False)
    skip_method = config['parameters']['skip_method']

    # Get baseline results
    baseline_bursts, baseline_time = run_exhaustive_search(
        ts, n_period, period_length, min_length, max_length, threshold, intra_period
    )

    if len(baseline_bursts) == 0:
        return None  # Skip grid points with no baseline bursts

    baseline_days = bursts_to_days(baseline_bursts)

    results = []

    # Test different skip factors
    for skip_factor in skip_factors:

        bursts, exec_time = evaluate_skipping_strategy(
            ts, skip_factor, n_period, period_length,
            min_length, max_length, threshold, intra_period,
            skip_method=skip_method
        )

        if len(bursts) != 0:
            # Compute metrics
            detected_days = bursts_to_days(bursts)

            results.append({
                'lat': lat_idx,
                'lon': lon_idx,
                'skip_factor': skip_factor,
                'execution_time': exec_time,
                'baseline_time': baseline_time,
                'speedup': baseline_time / exec_time,
                'n_bursts': len(bursts),
                'detected_days': len(detected_days),
                'baseline_days': len(baseline_days),
                'day_precision': precision(baseline_days, detected_days),
                'hit_precision_25': hit_precision(baseline_bursts, bursts, 25),
                'hit_precision_50': hit_precision(baseline_bursts, bursts, 50),
                'hit_precision_75': hit_precision(baseline_bursts, bursts, 75),
                'hit_precision_95': hit_precision(baseline_bursts, bursts, 95)
            })
        else:
            results.append({
                'lat': lat_idx,
                'lon': lon_idx,
                'skip_factor': skip_factor,
                'execution_time': exec_time,
                'baseline_time': baseline_time,
                'speedup': baseline_time / exec_time,
                'n_bursts': 0,
                'detected_days': 0,
                'baseline_days': len(baseline_days),
                'day_precision': 0,
                'hit_precision_25': 0,
                'hit_precision_50': 0,
                'hit_precision_75': 0,
                'hit_precision_95': 0
            })

    return results


def plot_grid_results(results_df: pd.DataFrame, output_dir: Path, skip_method: str):
    """Create visualization plots for grid experiment results."""
    # Create figure directory
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    print(results_df.head(1))
    skip_factors = pd.to_numeric(
        results_df['skip_factor'], errors='coerce').dropna().astype(int).unique()

    info_per_skip_factor = {}
    for skip_factor, group in results_df.groupby('skip_factor'):
        total_baseline_days = pd.to_numeric(
            group['baseline_days'], errors='coerce').fillna(0).sum()
        total_baseline_time = pd.to_numeric(
            group['baseline_time'], errors='coerce').fillna(0).sum()
        total_execution_time = pd.to_numeric(
            group['execution_time'], errors='coerce').fillna(0).sum()
        detected_days = pd.to_numeric(
            group['detected_days'], errors='coerce').fillna(0).sum()

        if total_execution_time > 0 and total_baseline_days > 0:  # Avoid division by zero
            speedup = total_baseline_time / total_execution_time
            day_precision = detected_days / total_baseline_days
        else:
            speedup, day_precision = 0, 0  # Handle division errors gracefully

        info_per_skip_factor[skip_factor] = {
            "speedup": speedup,
            "day_precision": day_precision
        }
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    x = np.arange(len(skip_factors))
    width = 0.35

    day_precision_values = [info_per_skip_factor[k]
                            ["day_precision"] for k in skip_factors]
    speedup_values = [info_per_skip_factor[k]["speedup"] for k in skip_factors]

    print(day_precision_values)
    print(speedup_values)

    rects1 = ax1.bar(x - width/2, day_precision_values, width,
                     label='Day Precision', color='lightblue')
    rects2 = ax2.bar(x + width/2, speedup_values, width,
                     label='Speedup', color='darkblue')

    ax1.set_xlabel('Skip factor')
    ax1.set_ylabel('Precision', color='lightblue')
    ax2.set_ylabel('Speedup', color='darkblue')
    ax1.set_title('Date Skipping: Precision and Speedup')
    ax1.set_xticks(x)
    ax1.set_xticklabels(skip_factors)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / "figures/grid_tradeoff.png", bbox_inches='tight')
    plt.close()


def run(config: dict) -> None:
    """Run grid-based skipping tradeoff experiments."""
    logger.info("Starting grid skipping tradeoff experiments")

    # Load data
    ds = xr.open_dataset(config['data']['path'])
    var_name = config['data']['variable']
    data = ds[var_name].values.astype(np.float64)

    # Create output directory
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get grid dimensions
    n_time, n_lat, n_lon = data.shape

    name = "results/" + config['output']['metrics_filename']

    if not Path(output_dir / name).exists():

        # Find valid grid points (non-NaN)
        valid_points = []
        tdata = data[0, :, :]
        if var_name == "tp":
            valid_points = [(i, j) for j in range(n_lon)
                            for i in range(n_lat) if not np.all(data[:, i, j] == 0)]
        else:
            valid_points = [(i, j) for j in range(n_lon) for i in range(
                n_lat) if not np.all(np.isnan(tdata[i, j]))]

        all_results = []

        # Run experiments for each grid point
        for lat_idx, lon_idx in tqdm(valid_points, desc="Processing grid points"):
            ts = data[:, lat_idx, lon_idx]

            # Run experiment for current grid point
            results = run_grid_experiment(ts, lat_idx, lon_idx, config)

            if results is not None:
                all_results.extend(results)

        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / name, index=False)

    else:
        results_df = pd.read_csv(output_dir / name)

    # Create plots
    plot_grid_results(results_df, output_dir,
                      config['parameters']['skip_method'])

    logger.info("Completed grid skipping tradeoff experiments")


if __name__ == '__main__':
    # Example configuration
    test_config = {
        'parameters': {
            'min_length': 1,
            'max_length': 240,
            'period_length': 365,
            'threshold': 3.0,
            'skip_factors': [1, 2, 5, 10, 20],
            'skip_method': 'date'  # or 'length'
        },
        'data': {
            'path': "Data/SST/merged_data_processed.nc",
            'variable': 'sst',
            'intra_period': False
        },
        'output': {
            'results_dir': 'results/grid_tradeoff',
            'metrics_filename': 'grid_tradeoff_metrics.csv'
        }
    }
    run(test_config)
