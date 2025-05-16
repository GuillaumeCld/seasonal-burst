"""Experiment to evaluate different sum computation strategies."""

import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, float64, int64

logger = logging.getLogger(__name__)

def timeit(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start

@njit((float64[:], int64, int64, int64))
def explicit_sum(ts, max_length, n_period, period_length):
    sums = np.zeros((n_period, period_length), dtype=np.float64)
    for length in range(2, max_length+1):
        for i in range(n_period):
            for j in range(period_length):

                start = i * period_length + j
                end = start + length - 1
                if end >= len(ts):
                    break
                sums[i, j] = np.sum(ts[start:end + 1])

    return sums

@njit((float64[:], int64, int64, int64))
def incremental_window(ts, max_length, n_period, period_length):
    sums = np.zeros((n_period, period_length), dtype=np.float64)
    n = len(ts)

    for length in range(2, max_length+1):
        end_index = n - length + 1
        end_year = end_index // period_length

        sums[0, 0] = np.sum(ts[:length])
    
        for j in range(1, period_length):
            sums[0, j] = sums[0, j-1] + ts[j+length-1] - ts[j-1]

        for year in range(1, end_year):
            start = year * period_length
            sums[year, 0] = np.sum(ts[start:start+length])
            for j in range(1, period_length):
                sums[year, j] = sums[year, j-1] + ts[start+j+length-1] - ts[start+j-1]

        max_sequence = period_length - length + 1
        start = end_year * period_length

        sums[end_year, 0] = np.sum(ts[start:start+length])
        for j in range(1, max_sequence):
            sums[end_year, j] = sums[end_year, j-1] + \
                ts[start+j+length-1] - ts[start+j-1]
        
    return sums

@njit((float64[:], int64, int64, int64))
def incremental_vectorized(ts, max_length, n_period, period_length):
    sums = np.zeros((n_period, period_length), dtype=np.float64)
    n = len(ts)

    for length in range(2, max_length+1):
        end_index = n - length + 1
        e_span = end_index % period_length
        end_year = end_index // period_length
        start = length - 1
        end = start + period_length
       
        sums[0,:] += ts[start:end]
        for year in range(1, end_year):
            start = year * period_length + length - 1
            end = start + period_length
            sums[year, :] = sums[year, :] + ts[start:end]
    
        # Last year special case to handle interval out of bounds
        start = end_year * period_length + length - 1
        end = start + e_span
        sums[end_year, :e_span] += ts[start:end]

    return sums



def run(config: dict) -> None:
    """Run sum computation experiments comparing different strategies."""
    logger.info("Starting sum computation experiments")
    
    # Extract configuration
    n_runs = config['execution']['n_runs']
    np.random.seed(int(config['execution']['random_seed']))
    
    # Parameters
    period_length = int(config['data']['period_length'])
    n_period = int(config['data']['n_periods'])
    min_length = int(config['data']['min_length'])
    max_length = int(config['data']['max_length'])
    step_length = int(config['data']['step_length'])
    
    
        
    ts = np.random.normal(0, 1, n_period * period_length)



    times_explicit = []
    times_incremental = []
    times_vectorized = []

    lengths = np.arange(min_length, max_length + 1, step_length)

    for length in lengths:
        times_explicit.append(timeit(explicit_sum, ts, length, n_period, period_length))
        times_incremental.append(timeit(incremental_window, ts, length, n_period, period_length))
        times_vectorized.append(timeit(incremental_vectorized, ts, length, n_period, period_length))


    incremental_slope = np.polyfit(lengths, times_incremental, 1)[0]
    vectorized_slope = np.polyfit(lengths, times_vectorized, 1)[0]
    ratio = incremental_slope / vectorized_slope

    print(f'Incremental slope: {incremental_slope}')
    print(f'Vectorized slope: {vectorized_slope}')
    print(f'Ratio: {ratio}')


    output_dir = Path(config['output']['results_dir'])

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, times_explicit, label='Explicit')
    plt.plot(lengths, times_incremental, label='Incremental - sliding')
    plt.plot(lengths, times_vectorized, label='Incremental - vectorized')
    plt.legend()
    plt.xlabel('Maximal length')
    plt.ylabel('Time (s)')
    plt.savefig(output_dir / 'figures/sum_compute.pdf', bbox_inches='tight')
    plt.close()

        

    
    # Save results
    result = {
        'explicit': times_explicit,
        'sliding': times_incremental,
        'vectorized': times_vectorized}
    df = pd.DataFrame(result)
    df.to_csv(output_dir / config['output']['metrics_filename'], index=False)
    
   
    
    logger.info("Completed sum computation experiments")

if __name__ == '__main__':
    run()