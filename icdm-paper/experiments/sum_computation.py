import numpy as np
from numba import njit, float64, int64
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def timeit(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start

@njit((float64[:], int64, int64, int64))
def explicit_sum_compute(ts, max_length, n_year, len_year):
    sums = np.zeros((n_year, len_year), dtype=np.float64)
    for length in range(2, max_length+1):
        for i in range(n_year):
            for j in range(len_year):

                start = i * len_year + j
                end = start + length - 1
                if end >= len(ts):
                    break
                sums[i, j] = np.sum(ts[start:end + 1])

    return sums

@njit((float64[:], int64, int64, int64))
def incremental_window(ts, max_length, n_year, len_year):
    sums = np.zeros((n_year, len_year), dtype=np.float64)
    n = len(ts)

    for length in range(2, max_length+1):
        end_index = n - length + 1
        end_year = end_index // len_year

        sums[0, 0] = np.sum(ts[:length])
    
        for j in range(1, len_year):
            sums[0, j] = sums[0, j-1] + ts[j+length-1] - ts[j-1]

        for year in range(1, end_year):
            start = year * len_year
            sums[year, 0] = np.sum(ts[start:start+length])
            for j in range(1, len_year):
                sums[year, j] = sums[year, j-1] + ts[start+j+length-1] - ts[start+j-1]

        max_sequence = len_year - length + 1
        start = end_year * len_year

        sums[end_year, 0] = np.sum(ts[start:start+length])
        for j in range(1, max_sequence):
            sums[end_year, j] = sums[end_year, j-1] + \
                ts[start+j+length-1] - ts[start+j-1]
        
    return sums

@njit((float64[:], int64, int64, int64))
def incremental_vectorized(ts, max_length, n_year, len_year):
    sums = np.zeros((n_year, len_year), dtype=np.float64)
    n = len(ts)

    for length in range(2, max_length+1):
        end_index = n - length + 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        start = length - 1
        end = start + len_year
       
        sums[0,:] += ts[start:end]
        for year in range(1, end_year):
            start = year * len_year + length - 1
            end = start + len_year
            sums[year, :] = sums[year, :] + ts[start:end]
    
        # Last year special case to handle interval out of bounds
        start = end_year * len_year + length - 1
        end = start + e_span
        sums[end_year, :e_span] += ts[start:end]

    return sums

def main():

    n_year = 50
    len_year = 365 * 24

    ts = np.random.rand(n_year * len_year)
    max_length = 240
    step = 10

    times_explicit = []
    times_incremental = []
    times_vectorized = []

    lengths = list(range(step, max_length+1, step))

    for length in tqdm(lengths):

        times_explicit.append(timeit(explicit_sum_compute, ts, length, n_year, len_year))
        times_incremental.append(timeit(incremental_window, ts, length, n_year, len_year))
        times_vectorized.append(timeit(incremental_vectorized, ts, length, n_year, len_year))


    print(f"{times_explicit[-1]:.3f}")
    print(f"{times_incremental[-1]:.3f}")
    print(f"{times_vectorized[-1]:.3f}")


    incremental_slope = np.polyfit(lengths, times_incremental, 1)[0]
    vectorized_slope = np.polyfit(lengths, times_vectorized, 1)[0]
    ratio = incremental_slope / vectorized_slope

    print(f'Incremental slope: {incremental_slope}')
    print(f'Vectorized slope: {vectorized_slope}')
    print(f'Ratio: {ratio}')

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, times_explicit, label='Explicit')
    plt.plot(lengths, times_incremental, label='Incremental - sliding')
    plt.plot(lengths, times_vectorized, label='Incremental - vectorized')
    plt.legend()
    plt.xlabel('Maximal length')
    plt.ylabel('Time (s)')
    plt.savefig('sum_compute.pdf', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()