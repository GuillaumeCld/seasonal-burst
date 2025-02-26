import numpy as np
import time as time
import src.burst_detection.numba.exhaustive_search as nexhaustive
import matplotlib.pyplot as plt
from tqdm import tqdm



def timeit(func, *args, **kwargs):
    start = time.time()
    output = func(*args, **kwargs)
    return output, time.time() - start

def var_len_year(n_year, max_lengths, min_len_year, max_len_year, step_len_year):

    threshold = 3
    min_length = 1

    all_times = []
    np.random.seed(42)

    for length in max_lengths:
        print(f'Length {length}')
        times = []
        for len_year in tqdm(range(min_len_year, max_len_year+1, step_len_year)):
            ts = np.random.normal(0, 1, (n_year * len_year)).astype(np.float64)
            _, time_exhaustive = timeit(nexhaustive.compute_all_bursts, ts, n_year, threshold, len_year, min_length, length)

            times.append(time_exhaustive)
        all_times.append(times)

    plt.figure(figsize=(10, 5))
    for i, length in enumerate(max_lengths):
        plt.plot(range(min_len_year, max_len_year+1, step_len_year), all_times[i], label=f'{length}')

    plt.xlabel('Period length')
    plt.ylabel('Time (s)')
    plt.legend(title='Maximal length')
    plt.savefig('src/experiments/figures/len_year.png')
    plt.show()


def var_n_year(len_year, max_lengths, min_n_year, max_n_year, step_n_year):

    threshold = 3
    min_length = 1

    all_times = []

    np.random.seed(42)
    for length in max_lengths:
        times = []
        for n_year in tqdm(range(min_n_year, max_n_year+1, step_n_year)):
            ts = np.random.normal(0, 1, (n_year * len_year)).astype(np.float64)

            _, time_exhaustive = timeit(nexhaustive.compute_all_bursts, ts, n_year, threshold, len_year, min_length, length)

            times.append(time_exhaustive)
        all_times.append(times)

    plt.figure(figsize=(10, 5))
    for i, length in enumerate(max_lengths):
        plt.plot(range(min_n_year, max_n_year+1, step_n_year), all_times[i], label=f'{length}')

    plt.xlabel('Number of periods')
    plt.ylabel('Time (s)')
    plt.legend(title='Maximal length')
    plt.savefig('src/experiments/figures/n_year.png')
    plt.show()

def var_n_length(n_year, len_year, min_length, max_length, step_length):
    threshold = 3

    all_times = []

    np.random.seed(42)

    for length in tqdm(range(min_length, max_length+1, step_n_year)):
        ts = np.random.normal(0, 1, (n_year * len_year)).astype(np.float64)

        _, time_exhaustive = timeit(nexhaustive.compute_all_bursts, ts, n_year, threshold, len_year, min_length, length)

        all_times.append(time_exhaustive)

    plt.figure(figsize=(10, 5))
    plt.plot(range(min_length, max_length+1, step_length), all_times)
    plt.xlabel('Maximal length')
    plt.ylabel('Time (s)')
    plt.savefig('src/experiments/figures/length.png')

    plt.show()



if __name__ == "__main__":
    
    n_year = 50
    max_lengths = [24, 3*24, 10*24]
    min_len_year = 50*24
    max_len_year = 365*24
    step_len_year = 100
    var_len_year(n_year, max_lengths, min_len_year, max_len_year, step_len_year)

    len_year = 365*24
    max_lengths = [24, 3*24, 10*24]
    min_n_year = 20
    max_n_year = 100
    step_n_year = 10

    var_n_year(len_year, max_lengths, min_n_year, max_n_year, step_n_year)

    # var_n_length(n_year, len_year, 1, 501, 50)
