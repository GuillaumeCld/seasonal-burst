"""Experiment to evaluate the scaling behavior of the online algorithm."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seasonal_burst.online_algorithm import WelfordBurstDetector

logger = logging.getLogger(__name__)


def measure_processing_time(obs: float, index: int, detector: WelfordBurstDetector) -> float:
    """Measure the processing time for online detection.
    
    Parameters:
        obs: Observation to process
        index: Index of the observation
        detector: Detector to use
        
    Returns:
        Processing time in microseconds
    """
    # Measure time for processing the last observation in microseconds
    start_time = time.perf_counter()
    detector.process(obs, index)
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1e6  # Convert to microseconds


def run_scaling_experiment(config: dict, min_length: int, max_length: int) -> list:
    """Run scaling experiment for a specific window type."""
    # Extract parameters
    period_length = int(config['parameters']['period_length'])
    threshold = float(config['parameters']['threshold'])
    n_runs = int(config['execution']['n_runs'])
    init_years = int(config['parameters']['init_years'])
    processing_years = int(config['parameters']['processing_years'])
    results = []
    ts = np.load(config['data']['path'])       
    ts_init = ts[:init_years * period_length]
    ts_processing = ts[-processing_years * period_length:]

    processing_time_sliding   = []
    processing_time_expanding = []
    for _ in range(n_runs):
        detector_sliding = WelfordBurstDetector(
            ts=ts_init,
            n_year=init_years,
            len_year=period_length,
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
            moving=True
        )
        detector_expanding = WelfordBurstDetector(
            ts=ts_init,
            n_year=init_years,
            len_year=period_length,
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
            moving=False
        )
        for index, obs in enumerate(ts_processing):
            time_sliding = measure_processing_time(obs, index, detector_sliding)
            time_expanding = measure_processing_time(obs, index, detector_expanding)
            processing_time_sliding.append(time_sliding)
            processing_time_expanding.append(time_expanding)

    return [max_length, np.mean(processing_time_sliding), np.mean(processing_time_expanding)]


def create_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots for scaling results."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Processing time vs number of observations
    plt.figure(figsize=(12, 6))
    plt.plot(df['max_length'], df['processing_time_sliding'], label='Sliding')
    plt.plot(df['max_length'], df['processing_time_expanding'], label='Expanding')
    plt.xlabel('Maximum Length')
    plt.ylabel('Processing Time (μs)') 
    plt.title('Online Processing Time vs Maximum Length')
    plt.legend()
    plt.grid(True, alpha=0.3)  
    plt.savefig(fig_dir / "processing_time.pdf", bbox_inches='tight')
    plt.close()
    

def run(config: dict) -> None:
    """Run online scaling experiments."""
    logger.info("Starting online scaling experiments")
    
    # Create output directory
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments for each window type
    min_length = int(config['parameters']['min_length'])
    max_length = int(config['parameters']['max_length'])
    step_length = int(config['parameters']['step_length'])
    max_lengths = []
    time_sliding = []
    time_expanding = []
    lengths = [step_length * i for i in range(1, max_length // step_length + 1)]
    lengths.insert(0, min_length)
    lengths.insert(0, min_length)
    for length in lengths:
        results = run_scaling_experiment(config, min_length, length)
        max_lengths.append(results[0])
        time_sliding.append(results[1])
        time_expanding.append(results[2])

    # remove first element because to prevent random high numba cost in
    results = {
        'max_length': max_lengths[1:],
        'processing_time_sliding': time_sliding[1:],  
        'processing_time_expanding': time_expanding[1:]
    }

    df = pd.DataFrame(results)
    name = "results/" + config['output']['metrics_filename'] 
    df.to_csv(output_dir / name, index=False)
    # Create plots
    create_plots(df, output_dir)
    
    logger.info("Completed online scaling experiments")


if __name__ == "__main__":
    # Example configuration for testing
    test_config = {
        'parameters': {
            'min_length': 1,
            'max_length': 365,
            'period_length': 365,
            'threshold': 3.0,
            'init_years': 20,
            'processing_years': 30,
            'step_length': 10
        },
        'data': {
            'path': "Data/SST/mediterranean_average_processed.npy"
        },
        'output': {
            'results_dir': 'results/online_scaling',
            'plot_filename': 'online_scaling_results.pdf',
            'metrics_filename': 'online_scaling_metrics.csv'
        },
        'execution': {
            'n_runs': 5,
            'random_seed': 42
        }
    }
    run(test_config) 