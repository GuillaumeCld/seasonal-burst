import numpy as np
from numba import njit, float64, int64, float32
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def explicit_sum_compute(ts, length, n_year, len_year):

    sums = np.zeros((n_year, len_year), dtype=np.float64)
    for i in range(n_year):
        for j in range(len_year):

            start = i * len_year + j
            end = start + length - 1
            if end >= len(ts):
                break
            sums[i, j] = np.sum(ts[start:end+1])
    return sums


def explicit_z_score_compute(ts, length, n_year, len_year):

    sums = np.zeros((n_year, len_year), dtype=np.float128)
    for i in range(n_year):
        for j in range(len_year):

            start = i * len_year + j
            end = start + length - 1
            if end >= len(ts):
                break
            sums[i, j] = np.sum(ts[start:end+1])


    means = np.zeros(len_year, dtype=np.float64)
    stds = np.zeros(len_year, dtype=np.float64)
    for date in range(len_year):
        
        end_index = len(ts) - length + 1
        end_year = end_index // len_year
        e_span = end_index % len_year

        if date < e_span:
            column = sums[:end_year+1, date]
        else:
            column = sums[:end_year, date]
        means[date] = np.mean(column)
        stds[date] = np.std(column)

    z_score = (sums - means) / stds

    return sums, means, stds, z_score

@njit((float64[:], int64, int64, int64))
def incremental_vectorized_double(ts, max_length, n_year, len_year):
    sums = np.copy(ts).reshape(n_year, len_year)
    n = len(ts)

    for length in range(2, max_length+1):
        end_index = n - length + 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        start = length - 1
        end = start + len_year
       
        for year in range(end_year):
            start = year * len_year + length - 1
            end = start + len_year
            sums[year, :] = sums[year, :] + ts[start:end]
    
        # Last year special case to handle interval out of bounds
        start = end_year * len_year + length - 1
        end = start + e_span
        sums[end_year, :e_span] += ts[start:end]

    return sums


@njit((float32[:], int64, int64, int64))
def incremental_vectorized_single(ts, max_length, n_year, len_year):
    sums = np.copy(ts).reshape(n_year, len_year)
    n = len(ts)


    for length in range(2, max_length+1):
        end_index = n - length + 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        start = length - 1
        end = start + len_year
       
        for year in range(end_year):
            start = year * len_year + length - 1
            end = start + len_year
            sums[year, :] += ts[start:end]
    
        # Last year special case to handle interval out of bounds
        start = end_year * len_year + length - 1
        end = start + e_span
        sums[end_year, :e_span] += ts[start:end]

    return sums


@njit((float64[:], int64, float64, int64, int64, int64))
def z_score_vectorized_double(ts, n_year, threshold, len_year, min_length, max_length):
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    end_index = n
    mean_v = np.zeros(len_year, dtype=np.float64)
    for size in range(2, max_length+1):
        end_index -= 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        start = size - 1
        end = start + len_year
        v[0,:] += ts[start:end]
        mean_v = np.copy(v[0, :]) 
        std_v = np.zeros(len_year, dtype=np.float64)
        for year in range(1, end_year):
            inv = 1/(year+1)
            start = year * len_year + size - 1
            end = start + len_year
            v[year, :] += ts[start:end]
            tmp = v[year, :] - mean_v
            std_v += year*inv *  tmp*tmp            
            mean_v += tmp * inv
        start = end_year * len_year + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        inv = 1/(end_year+1)
        tmp = v[end_year, :e_span] - mean_v[:e_span]
        std_v[:e_span] += end_year*inv * tmp*tmp
        mean_v[:e_span] += tmp * inv  
        std_v[:e_span] = std_v[:e_span]*inv
        std_v[e_span:] = std_v[e_span:] / end_year
        std_v = np.sqrt(std_v)  
 
        
        z_score = np.copy(v)
        for year in range(end_year):
            z_score[year,:] -= mean_v
            z_score[year,:] /= std_v

        z_score[end_year, :e_span] -= mean_v[:e_span]
        z_score[end_year, :e_span] /= std_v[:e_span]
    return  v, mean_v, std_v, z_score

@njit((float32[:], int64, float64, int64, int64, int64))
def z_score_vectorized_single(ts, n_year, threshold, len_year, min_length, max_length):
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year).astype(np.float32)

    end_index = n

    mean_v = np.zeros(len_year, dtype=np.float32)
    for size in range(2, max_length+1):

        end_index -= 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        start = size - 1
        end = start + len_year
        v[0,:] += ts[start:end]
        mean_v = np.copy(v[0, :]) 
        std_v = np.zeros(len_year, dtype=np.float32)
        for year in range(1, end_year):
            inv = 1/(year+1)
            start = year * len_year + size - 1
            end = start + len_year
            v[year, :] += ts[start:end]
            tmp = v[year, :] - mean_v
            std_v += year*inv *  tmp*tmp            
            mean_v += tmp * inv
        start = end_year * len_year + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        inv = 1/(end_year+1)
        tmp = v[end_year, :e_span] - mean_v[:e_span]
        std_v[:e_span] += end_year*inv * tmp*tmp
        mean_v[:e_span] += tmp * inv  
        std_v[:e_span] = std_v[:e_span]*inv
        std_v[e_span:] = std_v[e_span:] / end_year
        std_v = np.sqrt(std_v)  


        z_score = np.copy(v)
        for year in range(end_year):
            z_score[year,:] -= mean_v
            z_score[year,:] /= std_v

        z_score[end_year, :e_span] -= mean_v[:e_span]
        z_score[end_year, :e_span] /= std_v[:e_span]

    return  v, mean_v, std_v, z_score


def run(config):

    data_path = config['data']['path']
    len_year = int(config['data']['len_year'])
    max_length = int(config['data']['max_length'])
    step = int(config['data']['step_length'])
    lengths = range(step, max_length+1, step)

    ts_double = np.load(data_path)
    ts_single = np.copy(ts_double).astype(np.float32)
    n_year = len(ts_single) // len_year


    sums_error_single = []
    sums_error_double = []
    mean_error_single = []
    mean_error_double = []
    std_error_single = []
    std_error_double = []
    z_score_error_single = []
    z_score_error_double = []

    for length in lengths:
        n_seq = len(ts_single) - length + 1
        sums, mean, std, z_score = explicit_z_score_compute(ts_double, length, n_year, len_year)
        sums_single, mean_single, std_single, z_score_single = z_score_vectorized_single(ts_single, n_year, 1, len_year, 1, length)
        sums_double, mean_double, std_double, z_score_double = z_score_vectorized_double(ts_double, n_year, 1, len_year, 1, length)


        sums = sums.flatten()[:n_seq]
        sums_single = sums_single.flatten()[:n_seq]
        sums_double = sums_double.flatten()[:n_seq]
        z_score = z_score.flatten()[:n_seq]
        z_score_single = z_score_single.flatten()[:n_seq]
        z_score_double = z_score_double.flatten()[:n_seq]

        sums = sums.astype(np.float32)
        sums_single = sums_single.astype(np.float32)
        sums_double = sums_double.astype(np.float32)

        mean = mean.astype(np.float32)
        mean_single = mean_single.astype(np.float32)
        mean_double = mean_double.astype(np.float32)

        std = std.astype(np.float32)
        std_single = std_single.astype(np.float32)
        std_double = std_double.astype(np.float32)

        z_score = z_score.astype(np.float32)
        z_score_single = z_score_single.astype(np.float32)
        z_score_double = z_score_double.astype(np.float32)


        sums_error_single.append(np.linalg.norm(sums - sums_single) / np.linalg.norm(sums))
        sums_error_double.append(np.linalg.norm(sums - sums_double) / np.linalg.norm(sums))


        mean_error_single.append(np.linalg.norm(mean - mean_single) / np.linalg.norm(mean))
        std_error_single.append(np.linalg.norm(std - std_single) / np.linalg.norm(std))
        z_score_error_single.append(np.linalg.norm(z_score - z_score_single) / np.linalg.norm(z_score))

        mean_error_double.append(np.linalg.norm(mean - mean_double) / np.linalg.norm(mean))
        std_error_double.append(np.linalg.norm(std - std_double) / np.linalg.norm(std))
        z_score_error_double.append(np.linalg.norm(z_score - z_score_double) / np.linalg.norm(z_score))


    results = {
        'lengths': lengths,
        'sums_error_single': sums_error_single,
        'sums_error_double': sums_error_double,
        'mean_error_single': mean_error_single,
        'mean_error_double': mean_error_double,
        'std_error_single': std_error_single,
        'std_error_double': std_error_double,
        'z-score_error_single': z_score_error_single,
        'z-score_error_double': z_score_error_double
    }
    df = pd.DataFrame(results)
    output_dir = Path(config['output']['results_dir'])
    name = "results/" + config['output']['metrics_filename']
    df.to_csv(output_dir / name, index=False)


    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

    axs[0,0].plot(range(step, max_length+1, step), sums_error_single, label='Single precision')
    axs[0,0].plot(range(step, max_length+1, step), sums_error_double, label='Double precision')
    axs[0,0].set_xlabel('Length of the sequence')
    axs[0,0].set_ylabel('Relative error')
    axs[0,0].set_title('Sum Error')

    axs[0,1].plot(range(step, max_length+1, step), mean_error_single)
    axs[0,1].plot(range(step, max_length+1, step), mean_error_double)
    axs[0,1].set_xlabel('Length of the sequence')
    axs[0,1].set_ylabel('Relative error')
    axs[0,1].set_title('Mean Error')

    axs[1,0].plot(range(step, max_length+1, step), std_error_single)
    axs[1,0].plot(range(step, max_length+1, step), std_error_double)
    axs[1,0].set_xlabel('Length of the sequence')
    axs[1,0].set_ylabel('Relative error')
    axs[1,0].set_title('Standard Deviation Error')

    axs[1,1].plot(range(step, max_length+1, step), z_score_error_single)
    axs[1,1].plot(range(step, max_length+1, step), z_score_error_double)
    axs[1,1].set_xlabel('Length of the sequence')
    axs[1,1].set_ylabel('Relative error')
    axs[1,1].set_title('Z-score Error')
    axs[1,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    lines, labels = [], []
    for row in axs:
        for ax in row:
            ax_lines, ax_labels = ax.get_legend_handles_labels()
            lines.extend(ax_lines)
            labels.extend(ax_labels)

    fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.025))
    plt.tight_layout()

    plt.savefig(output_dir / 'figures/numerical_precision.pdf', format='pdf', bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == '__main__':
    config = {
        'data': {
            'path': "Data/SST/mediterranean_average_processed.npy",
            'len_year': 365,
            'max_length': 365,
            'step': 50
        }
    }
    run(config)