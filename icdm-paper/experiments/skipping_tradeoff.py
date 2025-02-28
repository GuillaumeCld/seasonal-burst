"""Experiment to evaluate the tradeoff between skipping and accuracy."""

import logging
from pathlib import Path
import time
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from seasonal_burst import exhaustive_search, skipping_date, skipping_length
from utils.post_process import merge_bursts
import utils.operators as op

logger = logging.getLogger(__name__)


def timeit(func, *args, **kwargs):
    """Measure execution time of a function."""
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


def bursts_to_days(bursts: List[Tuple[int, int, float]]) -> Set[int]:
    """Convert burst intervals to set of days."""
    days = set()
    for start, end, _ in bursts:
        days.update(range(start, end + 1))
    return days


@njit
def precision(true_days: Set[int], detected_days: Set[int]) -> float:
    """Compute precision between true and detected days."""
    if len(detected_days) == 0:
        return 0
    
    # print(len(detected_days - true_days), "In detected but not in true")
    # print(len(true_days - detected_days), "In true but not in detected")
    return len(detected_days.intersection(true_days)) / len(true_days)


@njit
def hit_precision(true_merged_bursts: List[Tuple[int, int, float]],
                  detected_merged_bursts: List[Tuple[int, int, float]],
                  threshold: int = 50) -> float:
    """
    Compute hit-based precision between merged bursts.
    A hit is counted if at least threshold% of a detected burst overlaps with a true burst.
    """
    if len(true_merged_bursts) == 0:
        return 0

    hits = 0
    for true_burst in true_merged_bursts:
        true_length = true_burst[1] - true_burst[0] + 1
        overlap_length = true_length * threshold // 100
        for detected_burst in detected_merged_bursts:
            if op.overlapping(true_burst, detected_burst, min_overlap=overlap_length) :
                hits += 1
                break

    return hits / len(true_merged_bursts)


def run_exhaustive_search(ts: np.ndarray,
                          n_period: int, period_length: int,
                          min_length: int, max_length: int,
                          threshold: float, intra_period: bool = False) -> Tuple[List, float]:
    """
    Evaluate a skipping strategy on the time series.

    Parameters:
        ts: Input time series
        n_period: Number of periods
        period_length: Length of each period
        min_length: Minimum burst length
        max_length: Maximum burst length
        threshold: Detection threshold
        intra_period: Whether to use intra-period detection

    Returns:
        Tuple of (detected_bursts, execution_time)
    """

    # Run detection
    if intra_period:
        bursts, execution_time = timeit(exhaustive_search.exhaustive_search_intra_period,
                                        ts, n_period, threshold,
                                        period_length, min_length, max_length)
    else:
        bursts, execution_time = timeit(exhaustive_search.exhaustive_search,
                                        ts, n_period, threshold,
                                        period_length, min_length, max_length)

    return bursts, execution_time


def evaluate_skipping_strategy(ts: np.ndarray, skip_factor: int,
                               n_period: int, period_length: int,
                               min_length: int, max_length: int,
                               threshold: float, intra_period: bool = False,
                               skip_method: str = 'date') -> Tuple[List, float]:
    """Evaluate a skipping strategy on the time series."""



    if skip_method == 'date':
        bursts, execution_time = timeit(skipping_date.burst_compute,
                                        ts, n_period, threshold,
                                        period_length, min_length, max_length, skip_factor)
    elif skip_method == 'length':
        bursts, execution_time = timeit(skipping_length.burst_compute,
                                        ts, n_period, threshold,
                                        period_length, min_length, max_length, skip_factor)
    else:
        raise ValueError(f"Invalid skip method: {skip_method}")

    return bursts, execution_time


def run_skipping_experiment(config: dict, skip_method: str) -> pd.DataFrame:
    """Run date-based skipping experiments."""
    logger.info("Running date skipping experiments")

    # Load data
    ts = np.load(config['data']['path'])
    period_length = config['parameters']['period_length']
    n_period = len(ts) // period_length
    min_length = config['parameters']['min_length']
    max_length = config['parameters']['max_length']
    threshold = config['parameters']['threshold']
    skip_factors = config['parameters']['skip_factors']
    intra_period = config['data'].get('intra_period', False)

    results = []

    # Get baseline results (no skipping)
    baseline_bursts, baseline_time = run_exhaustive_search(
        ts, n_period, period_length, min_length, max_length, threshold, intra_period)
    baseline_days = bursts_to_days(baseline_bursts)
    baseline_merged = merge_bursts(baseline_bursts)

    # Test different skip factors
    for skip_factor in skip_factors:

        bursts, exec_time = evaluate_skipping_strategy(
            ts, skip_factor, n_period, period_length,
            min_length, max_length, threshold, intra_period,
            skip_method=skip_method
        )

        # Compute metrics
        detected_days = bursts_to_days(bursts)
        merged_bursts = merge_bursts(bursts)

        results.append({
            'skip_factor': skip_factor,
            'execution_time': exec_time,
            'speedup': baseline_time / exec_time,
            'n_bursts': len(bursts),
            'n_merged_bursts': len(merged_bursts),
            'day_precision': precision(baseline_days, detected_days),
            'hit_precision_25': hit_precision(baseline_merged, merged_bursts, 25),
            'hit_precision_50': hit_precision(baseline_merged, merged_bursts, 50),
            'hit_precision_75': hit_precision(baseline_merged, merged_bursts, 75),
            'hit_precision_90': hit_precision(baseline_merged, merged_bursts, 90),
        })

    return pd.DataFrame(results)


def plot_results(date_results: pd.DataFrame, length_results: pd.DataFrame,
                 output_dir: Path, plot_filename: str):
    """Create comparison plots for both skipping strategies."""
    plt.figure(figsize=(15, 10))

    # Create figure for date skipping
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    x = np.arange(len(date_results['skip_factor']))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, date_results['day_precision'], width, label='Day Precision', color='lightblue')
    rects2 = ax2.bar(x + width/2, date_results['speedup'], width, label='Speedup', color='darkblue')
    
    ax1.set_xlabel('Skip factor')
    ax1.set_ylabel('Precision')
    ax2.set_ylabel('Speedup')
    ax1.set_title('Date Skipping: Precision and Speedup')
    ax1.set_xticks(x)
    ax1.set_xticklabels(date_results['skip_factor'])
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"figures/date_{plot_filename}")
    plt.close()
    
    # Create figure for length skipping
    fig2, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    rects1 = ax1.bar(x - width/2, length_results['day_precision'], width, label='Day Precision', color='lightgreen')
    rects2 = ax2.bar(x + width/2, length_results['speedup'], width, label='Speedup', color='darkgreen')
    
    ax1.set_xlabel('Skip factor')
    ax1.set_ylabel('Precision')
    ax2.set_ylabel('Speedup')
    ax1.set_title('Length Skipping: Precision and Speedup')
    ax1.set_xticks(x)
    ax1.set_xticklabels(length_results['skip_factor'])
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"figures/length_{plot_filename}")
    plt.close()


def run(config: dict) -> None:
    """Run skipping tradeoff experiments."""
    logger.info("Starting skipping tradeoff experiments")

    # Create output directory
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    if "date" in config['parameters']['skip_methods']:
        date_results = run_skipping_experiment(config, skip_method='date')
        date_results.to_csv(
            output_dir / "results/date_skipping_results.csv", index=False)

    if "length" in config['parameters']['skip_methods']:
        length_results = run_skipping_experiment(config, skip_method='length')
        length_results.to_csv(
            output_dir / "results/length_skipping_results.csv", index=False)

    # Create plots if both methods were run
    if len(config['parameters']['skip_methods']) == 2:
        plot_results(date_results, length_results,
                     output_dir, config['output']['plot_filename'])


    logger.info("Completed skipping tradeoff experiments")


if __name__ == '__main__':
    # Example configuration for testing
    test_config = {
        'parameters': {
            'min_length': 1,
            'max_length': 240,
            'period_length': 365,
            'threshold': 3.0,
            'skip_factors': [1, 2, 5, 10, 20],
            'skip_methods': ['date', 'length']
        },
        'data': {
            'path': "Data/SST/mediterranean_average_processed.npy",
            'intra_period': False
        },
        'output': {
            'results_dir': 'results/tradeoff',
            'plot_filename': 'tradeoff_results.pdf',
            'metrics_filename': 'tradeoff_metrics.csv'
        }
    }
    run(test_config)
