"""Experiment to evaluate the scaling behavior of the exhaustive search algorithm."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seasonal_burst import exhaustive_search

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_period: int, period_length: int) -> np.ndarray:
    """Generate synthetic time series data."""
    n = n_period * period_length
    ts = np.random.normal(0, 1, n)
    return ts


def run_scaling_experiment(ts: np.ndarray, n_period: int, period_length: int, max_length: int, threshold: float) -> float:
    """Run a single scaling experiment and return execution time."""
    start_time = time.time()
    _ = exhaustive_search.exhaustive_search(
        ts, n_period, threshold, period_length, 1, max_length)
    end_time = time.time()
    return end_time - start_time


def run(config: dict) -> None:
    """Run the exhaustive algorithm scaling experiments.

    This experiment evaluates how the algorithm scales with:
    1. Period length (fixing number of periods)
    2. Number of periods (fixing period length)
    3. Maximum burst length
    """
    logger.info("Starting exhaustive algorithm scaling experiments")

    # Extract configuration parameters
    max_lengths = [int(e) for e in config['common']['max_lengths']]
    threshold = float(config['common']['threshold'])
    n_runs = int(config['execution']['n_runs'])
    np.random.seed(config['execution']['random_seed'])

    # Period length scaling
    period_lengths = np.arange(
        int(config['period_length_scaling']['min_period_length']),
        int(config['period_length_scaling']['max_period_length']),
        int(config['period_length_scaling']['step_period_length'])
    )
    fixed_n_period = int(config['period_length_scaling']['n_period'])

    # Number of periods scaling
    n_periods = np.arange(
        int(config['n_period_scaling']['min_n_period']),
        int(config['n_period_scaling']['max_n_period']),
        int(config['n_period_scaling']['step_n_period'])
    )
    fixed_period_length = int(config['n_period_scaling']['period_length'])

    results = []

    # Experiment 1: Period length scaling
    logger.info("Running period length scaling experiment")
    for period_length in period_lengths:
        for max_length in max_lengths:
            times = []
            for _ in range(n_runs):
                ts = generate_synthetic_data(fixed_n_period, period_length)
                execution_time = run_scaling_experiment(
                    ts, fixed_n_period, period_length, max_length, threshold)
                times.append(execution_time)

            results.append({
                'experiment': 'period_length_scaling',
                'period_length': period_length,
                'n_period': fixed_n_period,
                'max_length': max_length,
                'time_mean': np.mean(times),
                'time_std': np.std(times)
            })

    # Experiment 2: Number of periods scaling
    logger.info("Running number of periods scaling experiment")
    for n_period in n_periods:
        for max_length in max_lengths:
            times = []
            for _ in range(n_runs):
                ts = generate_synthetic_data(n_period, fixed_period_length)
                execution_time = run_scaling_experiment(
                    ts, n_period, fixed_period_length, max_length, threshold)
                times.append(execution_time)

            results.append({
                'experiment': 'n_period_scaling',
                'period_length': fixed_period_length,
                'n_period': n_period,
                'max_length': max_length,
                'time_mean': np.mean(times),
                'time_std': np.std(times)
            })

    # Save results
    df = pd.DataFrame(results)
    output_dir = Path(config['output']['results_dir'])
    name = "results/" + config['output']['metrics_filename']
    df.to_csv(output_dir / name, index=False)

    # Plot results
    # Period length scaling plot
    name = "figures/" + "period_length_" + config['output']['plot_filename']
    plt.figure(figsize=(15, 10))
    for max_length in max_lengths:
        mask = (df['experiment'] == 'period_length_scaling') & (
            df['max_length'] == max_length)
        plt.plot(df[mask]['period_length'], df[mask]['time_mean'],
                 label=f'max_length={max_length}')
    plt.xlabel('Period Length')
    plt.ylabel('Execution Time (s)')
    plt.title('Scaling with Period Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / name, bbox_inches='tight')
    plt.close()


    # Number of periods scaling plot
    name = "figures/" + "n_period_" + config['output']['plot_filename']
    plt.figure(figsize=(15, 10))
    for max_length in max_lengths:
        mask = (df['experiment'] == 'n_period_scaling') & (
            df['max_length'] == max_length)
        plt.plot(df[mask]['n_period'], df[mask]['time_mean'],
                 label=f'max_length={max_length}')
    plt.xlabel('Number of Periods')
    plt.ylabel('Execution Time (s)')
    plt.title('Scaling with Number of Periods')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / name, bbox_inches='tight')
    plt.close()

    logger.info("Completed exhaustive algorithm scaling experiments")


if __name__ == "__main__":
    pass
