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
        
        # if len(bursts) == 0:
        #     continue
            
        # Compute metrics
        detected_days = bursts_to_days(bursts)
        
        results.append({
            'lat': lat_idx,
            'lon': lon_idx,
            'skip_factor': skip_factor,
            'execution_time': exec_time,
            'speedup': baseline_time / exec_time,
            'n_bursts': len(bursts),
            'day_precision': precision(baseline_days, detected_days),
            'hit_precision_25': hit_precision(baseline_bursts, bursts, 25),
            'hit_precision_50': hit_precision(baseline_bursts, bursts, 50),
            'hit_precision_75': hit_precision(baseline_bursts, bursts, 75),
            'hit_precision_95': hit_precision(baseline_bursts, bursts, 95)
        })
    
    return results


def plot_grid_results(results_df: pd.DataFrame, output_dir: Path, skip_method: str):
    """Create visualization plots for grid experiment results."""
    # Create figure directory
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Plot precision distribution for each skip factor
    plt.figure(figsize=(12, 8))
    skip_factors = sorted(results_df['skip_factor'].unique())
    
    # Create boxplot for day precision
    plt.subplot(2, 1, 1)
    box_data = [results_df[results_df['skip_factor'] == sf]['day_precision'] 
                for sf in skip_factors]
    plt.boxplot(box_data, labels=skip_factors)
    plt.axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
    plt.xlabel('Skip Factor')
    plt.ylabel('Day Precision')
    plt.title(f'{skip_method} Skipping: Day Precision Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create boxplot for hit precision (50%)
    plt.subplot(2, 1, 2)
    box_data = [results_df[results_df['skip_factor'] == sf]['hit_precision_50'] 
                for sf in skip_factors]
    plt.boxplot(box_data, labels=skip_factors)
    plt.axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
    plt.xlabel('Skip Factor')
    plt.ylabel('Hit Precision (50%)')
    plt.title(f'{skip_method} Skipping: Hit Precision Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"{skip_method}_precision_distribution.pdf")
    plt.close()
    
    # Plot speedup distribution
    plt.figure(figsize=(10, 6))
    box_data = [results_df[results_df['skip_factor'] == sf]['speedup'] 
                for sf in skip_factors]
    plt.boxplot(box_data, labels=skip_factors)
    plt.xlabel('Skip Factor')
    plt.ylabel('Speedup')
    plt.title(f'{skip_method} Skipping: Speedup Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_dir / f"{skip_method}_speedup_distribution.pdf")
    plt.close()
    
    # Create heatmaps for different metrics
    metrics = ['day_precision', 'hit_precision_50', 'speedup']
    for metric in metrics:
        for skip_factor in skip_factors:
            metric_data = results_df[results_df['skip_factor'] == skip_factor]
            grid_data = np.full((max(metric_data['lat']) + 1, 
                               max(metric_data['lon']) + 1), np.nan)
            
            for _, row in metric_data.iterrows():
                grid_data[int(row['lat']), int(row['lon'])] = row[metric]
            
            plt.figure(figsize=(10, 8))
            plt.imshow(grid_data, cmap='viridis', aspect='auto')
            plt.colorbar(label=metric)
            plt.title(f'{skip_method} Skipping: {metric} (Skip Factor {skip_factor})')
            plt.xlabel('Longitude Index')
            plt.ylabel('Latitude Index')
            plt.savefig(fig_dir / f"{skip_method}_{metric}_sf{skip_factor}_heatmap.pdf")
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
    
    # Find valid grid points (non-NaN)
    valid_points = []
    tdata = data[0,:,:]
    for i in range(n_lat):
        for j in range(n_lon):
            if not np.isnan(tdata[i, j]):
                valid_points.append((i, j))
    
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
    
    # Save results
    results_df.to_csv(output_dir / config['output']['metrics_filename'], index=False)
    
    # Create plots
    plot_grid_results(results_df, output_dir, config['parameters']['skip_method'])
    
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
