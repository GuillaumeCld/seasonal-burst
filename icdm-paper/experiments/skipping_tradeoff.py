import numpy as np
import time as time

import src.burst_detection.numba.exhaustive_search as nexhaustive
import src.burst_detection.numba.exhaustive_search_skip_lengths as nexhaustive_skip_lengths
import src.burst_detection.numba.exhaustive_search_skip_both as nexhaustive_skip_both
import src.burst_detection.numba.exhaustive_search_skip_dates as nexhaustive_skip_dates


import src.utils.post_process as pp
import src.utils.operators as op
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import plotly.graph_objects as go
import pandas as pd

from numba import njit




def timeit(func, *args, **kwargs):
    start = time.time()
    output = func(*args, **kwargs)
    return output, time.time() - start

def bursts_to_days(bursts):
    merged_bursts = pp.merge_sequences(bursts) 
    list_days = []
    for burst in merged_bursts:
        s, e, _ = burst
        list_days.extend(range(s, e+1))
    
    return set(list_days)

@njit
def precision(true_days, detected_days):
    days_precision = len(detected_days.intersection(true_days)) / len(true_days)


    return  days_precision

@njit
def hit_precision(true_merged_bursts, detected_merged_bursts, threshold=50):
    if len(true_merged_bursts) == 0:
        return 0
    count = 0
    for true_burst in true_merged_bursts:
        true_length = true_burst[1] - true_burst[0] + 1
        overlap_length = true_length * threshold // 100
        for detected_burst in detected_merged_bursts:
            # if detected_burst has 50% overlap with true_burst
            if op.overlapping(true_burst, detected_burst, min_overlap=overlap_length) :
                count += 1
                break

    return count / len(true_merged_bursts)


def date_skipping_scaling():
    threshold = 3
    min_length = 3
    max_length = 240
    len_year = 92 * 24

    ds = xr.open_dataset("../ClimBurst/data/wgbt_era5_Paris_1974-2023jja.nc")
    ts = ds.wbgt.values.ravel().astype(np.float64)
    # ds = xr.open_dataset("Data/SST/merged_data_processed.nc")
    # ts = ds.sst.values[:, 0, 0].astype(np.float64)
    n_year = len(ts) // len_year

    print(f"Number of years: {n_year}")
    burst_exhaustive, time_exhaustive = timeit(nexhaustive.compute_all_bursts_numba_abs_incomplete_years, ts, n_year, threshold, len_year, min_length, max_length)
    days_exhaustive = bursts_to_days(burst_exhaustive)


    # steps = np.arange(50, 0, -5)
    # steps = np.concatenate((steps, [1]))
    steps = np.array([1,2,5,10,20])

    times = []
    speedups = []
    precisions = [] 

    for s in steps:
        bursts, time = timeit(nexhaustive_skip_dates.compute_all_bursts_numba_abs_incomplete_years, ts, n_year, threshold, len_year, min_length, max_length, s)
        detected_days = bursts_to_days(bursts)
        precision_score = precision(days_exhaustive, detected_days)
        precisions.append(precision_score)
        times.append(time)   
        speedups.append(time_exhaustive / time)

        print(f"Skip: {s} - Time: {time:.3f} - Speedup: {time_exhaustive / time:.1f} - Precision: {precision_score:.3f}")
        not_detected =  np.array(list(days_exhaustive.difference(detected_days)))
        not_detected.sort()
        print(not_detected // len_year)

    times = [t*1e3 for t in times]



   # Create evenly spaced indices for bars
    x_positions = np.arange(len(steps))  # Maps steps to linear spacing

    plt.figure(figsize=(14, 10))
    ax1 = plt.gca()

    # Bar width
    bar_width = 0.4

    # Plot Precision Bars
    ax1.bar(x_positions - bar_width/2, precisions, color='#1f77b4', width=bar_width, label="Precision")
    ax1.set_xlabel("Skip Value", fontsize=26)
    ax1.set_ylabel("Precision", fontsize=26,)
    ax1.tick_params(axis='y', labelsize=22)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Set x-ticks and correct labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(steps, fontsize=22)  # Keep original step values as labels

    # Add Value Labels for Precision
    for i, v in enumerate(precisions):
        ax1.text(x_positions[i] - bar_width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=20, color='#1f77b4')

    # Secondary Y-axis for Speedup
    ax2 = ax1.twinx()
    ax2.bar(x_positions + bar_width/2, speedups, color='#ff7f0e', width=bar_width, label="Speedup")
    ax2.set_ylabel("Speedup", fontsize=26)
    ax2.tick_params(axis='y', labelsize=22)

    # Add Value Labels for Speedup
    for i, v in enumerate(speedups):
        ax2.text(x_positions[i] + bar_width/2, v + 0.1, f"{v:.1f}", ha='center', fontsize=20, color='#ff7f0e')

    # Improved Legend Placement (Above with Two Columns)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc="upper center", fontsize=22, ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    plt.tight_layout()
    plt.savefig("src/experiments/figures/date_skip_scaling.pdf", format='pdf', bbox_inches='tight')
    plt.show()


    # plt.figure(figsize=(14, 10))

    # ax1 = plt.gca()
    # line1 = ax1.plot(steps, times, marker='o', label="Time (ms)", color='blue')
    # ax1.set_xlabel("Skip Value", fontsize=26)
    # ax1.set_ylabel("Time (ms)", fontsize=26)

    # ax2 = ax1.twinx()
    # line2 = ax2.plot(steps, speedups, marker='x', label="Speedup", color='red')
    # ax2.set_ylabel("Speedup", fontsize=26)

    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper center', fontsize=24)

    # ax1.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("src/experiments/figures/date_skip_scaling.pdf", format='pdf', bbox_inches='tight')

def length_skipping_scaling():
    threshold = 3
    min_length = 1
    max_length = 240
    len_year = 92 * 24

    ds = xr.open_dataset("../ClimBurst/data/wgbt_era5_Paris_1974-2023jja.nc")
    ts = ds.wbgt.values.ravel().astype(np.float64)
    n_year = len(ts) // len_year

    burst_exhaustive, time_exhaustive = timeit(nexhaustive.compute_all_bursts_numba_abs_incomplete_years, ts, n_year, threshold, len_year, min_length, max_length)
    merge_exhaustive = pp.merge_sequences(burst_exhaustive)
    days_exhaustive = bursts_to_days(burst_exhaustive)
    # steps = np.arange(50, 0, -5)
    # steps = np.concatenate((steps, [1]))
    
    steps = np.array([1,2,5,10,20])

    times = []
    speedups = []
    precisions = [] 

    for s in steps:
        bursts, time = timeit(nexhaustive_skip_lengths.compute_all_bursts_numba_abs_incomplete_years, ts, n_year, threshold, len_year, min_length, max_length, s)
        detected_days = bursts_to_days(bursts)
        precision_score = precision(days_exhaustive, detected_days)

        precisions.append(precision_score)
        
        times.append(time)   
        speedups.append(time_exhaustive / time)



        print(f"Skip: {s} - Time: {time:.3f} - Speedup: {time_exhaustive / time:.1f} - Precision: {precision_score:.3f}")

        not_detected =  np.array(list(days_exhaustive.difference(detected_days)))
        not_detected.sort()
        print(not_detected // len_year)


    times = [t*1e3 for t in times]

    # Create evenly spaced indices for bars
    x_positions = np.arange(len(steps))  # Maps steps to linear spacing

    plt.figure(figsize=(14, 10))
    ax1 = plt.gca()

    # Bar width
    bar_width = 0.4

    # Plot Precision Bars
    ax1.bar(x_positions - bar_width/2, precisions, color='#1f77b4', width=bar_width, label="Precision")
    ax1.set_xlabel("Skip Value", fontsize=26)
    ax1.set_ylabel("Precision", fontsize=26,)
    ax1.tick_params(axis='y', labelsize=22)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Set x-ticks and correct labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(steps, fontsize=22)  # Keep original step values as labels

    # Add Value Labels for Precision
    for i, v in enumerate(precisions):
        ax1.text(x_positions[i] - bar_width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=20, color='#1f77b4')

    # Secondary Y-axis for Speedup
    ax2 = ax1.twinx()
    ax2.bar(x_positions + bar_width/2, speedups, color='#ff7f0e', width=bar_width, label="Speedup")
    ax2.set_ylabel("Speedup", fontsize=26)
    ax2.tick_params(axis='y', labelsize=22)

    # Add Value Labels for Speedup
    for i, v in enumerate(speedups):
        ax2.text(x_positions[i] + bar_width/2, v + 0.1, f"{v:.1f}", ha='center', fontsize=20, color='#ff7f0e')

    # Improved Legend Placement (Above with Two Columns)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc="upper center", fontsize=22, ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    plt.tight_layout()
    plt.savefig("src/experiments/figures/length_skip_scaling.pdf", format='pdf', bbox_inches='tight')
    plt.show()






    # plt.figure(figsize=(14, 10))

    # ax1 = plt.gca()
    # line1 = ax1.plot(steps, times, marker='o', label="Time (ms)", color='blue')
    # ax1.set_xlabel("Skip Value", fontsize=26)
    # ax1.set_ylabel("Time (ms)", fontsize=26)

    # ax2 = ax1.twinx()
    # line2 = ax2.plot(steps, speedups, marker='x', label="Speedup", color='red')
    # ax2.set_ylabel("Speedup", fontsize=26)
    # ax2.set_yscale('log')
    # custom_ticks = [1.5, 2, 5, 10, int(max(speedups)//1)]
    # ax2.set_yticks(custom_ticks)
    # ax2.set_yticklabels([f'{tick:g}' for tick in custom_ticks])  
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper center', fontsize=24)

    # ax1.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("src/experiments/figures/length_skip_scaling.pdf", format='pdf', bbox_inches='tight')

def both_skipping_scaling():
    threshold = 3
    min_length = 1
    max_length = 240
    len_year = 92 * 24

    ds = xr.open_dataset("../ClimBurst/data/wgbt_era5_Paris_1974-2023jja.nc")
    ts = ds.wbgt.values.ravel().astype(np.float64)
    n_year = len(ts) // len_year

    burst_exhaustive, time_exhaustive = timeit(nexhaustive.compute_all_bursts, ts, n_year, threshold, len_year, min_length, max_length)
    merge_exhaustive = pp.merge_sequences(burst_exhaustive)
    days_exhaustive = bursts_to_days(burst_exhaustive)
    
    steps = np.arange(50, 0, -5)
    steps = np.concatenate((steps, [1]))
    steps = [1,2,5,10,20]
    times = []
    speedups = []
    precisions = [] 
    for s in steps:
        bursts, time = timeit(nexhaustive_skip_both.compute_all_bursts, ts, n_year, threshold, len_year, s, min_length, max_length)
        # merge_bursts = pp.merge_sequences(bursts)
        detected_days = bursts_to_days(bursts)
        precision_score = precision(days_exhaustive, detected_days)
        precisions.append(precision_score)
        
        times.append(time)   
        speedups.append(time_exhaustive / time)

        print(f"Skip: {s} - Time: {time:.3f} - Speedup: {time_exhaustive / time:.1f} - Precision: {precision_score:.3f}")
    times = [t*1e3 for t in times]

   # Create evenly spaced indices for bars
    x_positions = np.arange(len(steps))  # Maps steps to linear spacing

    plt.figure(figsize=(14, 10))
    ax1 = plt.gca()

    # Bar width
    bar_width = 0.4

    # Plot Precision Bars
    ax1.bar(x_positions - bar_width/2, precisions, color='#1f77b4', width=bar_width, label="Precision")
    ax1.set_xlabel("Skip Value", fontsize=26)
    ax1.set_ylabel("Precision", fontsize=26,)
    ax1.tick_params(axis='y', labelsize=22)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Set x-ticks and correct labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(steps, fontsize=22)  # Keep original step values as labels

    # Add Value Labels for Precision
    for i, v in enumerate(precisions):
        ax1.text(x_positions[i] - bar_width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=20, color='#1f77b4')

    # Secondary Y-axis for Speedup
    ax2 = ax1.twinx()
    ax2.bar(x_positions + bar_width/2, speedups, color='#ff7f0e', width=bar_width, label="Speedup")
    ax2.set_ylabel("Speedup", fontsize=26)
    ax2.set_ylim(0, np.max(speedups) + 1.1)
    ax2.tick_params(axis='y', labelsize=22)

    # Add Value Labels for Speedup
    for i, v in enumerate(speedups):
        ax2.text(x_positions[i] + bar_width/2, v + 0.1, f"{v:.1f}", ha='center', fontsize=20, color='#ff7f0e')

    # Improved Legend Placement (Above with Two Columns)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc="upper center", fontsize=22, ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    plt.tight_layout()
    plt.savefig("src/experiments/figures/both_skip_scaling.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # plt.figure(figsize=(14, 10))

    # ax1 = plt.gca()
    # line1 = ax1.plot(steps, times, marker='o', label="Time (ms)", color='blue')
    # ax1.set_xlabel("Skip Value", fontsize=26)
    # ax1.set_ylabel("Time (ms)", fontsize=26)

    # ax2 = ax1.twinx()
    # line2 = ax2.plot(steps, speedups, marker='x', label="Speedup", color='red')
    # ax2.set_ylabel("Speedup", fontsize=26)
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper center', fontsize=24)
    # ax2.set_yscale('log')
    # custom_ticks = [1.5, 2, 5, 10, int(max(speedups)//1)]
    # ax2.set_yticks(custom_ticks)
    # ax2.set_yticklabels([f'{tick:g}' for tick in custom_ticks])  

    # ax1.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("src/experiments/figures/both_skip_scaling.pdf", format='pdf', bbox_inches='tight')


def all_grid_medi():

    # Load the SST dataset
    ds = xr.open_dataset("Data/SST/merged_data_processed.nc")
    ds = ds.sst.values.astype(np.float64)

    # Extract grid dimensions
    n_lat, n_lon = ds.shape[1:]


    indices = []
    tmp = ds[1, :, :]  
    for i in range(n_lat):
        for j in range(n_lon):
            if not np.isnan(tmp[i, j]):
                indices.append((i, j))

    # Parameters
    threshold = 3
    len_year = 365
    min_length = 1
    max_length = 365

    results = []
    steps = [20, 10, 5, 1] 
    exhaustive_days = 0
    exhaustive_time = 0.0
    skip_time = [0.0 for _ in steps]

    skip_days = [0 for _ in steps]
    for (i, j) in tqdm(indices):
        ts = ds[:, i, j] 
        n_year = len(ts) // len_year

        burst_exhaustive, time_exhaustive = timeit(
            nexhaustive.compute_all_bursts,
            ts,
            n_year,
            threshold,
            len_year,
            min_length,
            max_length
        )
        exhaustive_time += time_exhaustive

        if len(burst_exhaustive) != 0:

            days_exhaustive = bursts_to_days(burst_exhaustive)
            merge_exhaustive = pp.merge_sequences(burst_exhaustive)
            exhaustive_days += len(days_exhaustive)

        precisions = []
        hit_precisions_25 = []
        hit_precisions_50 = []
        hit_precisions_75 = []
        hit_precisions_95 = []
        for s in steps:
            bursts, time = timeit(
                nexhaustive_skip_lengths.compute_all_bursts,
                ts,
                n_year,
                threshold,
                len_year,
                s,
                min_length,
                max_length
            )
            skip_time[steps.index(s)] += time
            if len(bursts) == 0:
                continue
            merge_bursts = pp.merge_sequences(bursts)
            detected_days = bursts_to_days(bursts)
            precision_score = precision(days_exhaustive, detected_days)
            hit_prec_50 = hit_precision(merge_exhaustive, merge_bursts, 50)
            hit_prec_25 = hit_precision(merge_exhaustive, merge_bursts, 25)
            hit_prec_75 = hit_precision(merge_exhaustive, merge_bursts, 75)
            hit_prec_95 = hit_precision(merge_exhaustive, merge_bursts, 95)

            precisions.append(precision_score)
            hit_precisions_25.append(hit_prec_25)
            hit_precisions_50.append(hit_prec_50)
            hit_precisions_75.append(hit_prec_75)
            hit_precisions_95.append(hit_prec_95)

            skip_days[steps.index(s)] += len(detected_days)

        results.append({
            "lat": i,
            "lon": j,
            "precisions": precisions,
            "hit_precisions_25": hit_precisions_25,
            "hit_precisions_50": hit_precisions_50,
            "hit_precisions_75": hit_precisions_75,
            "hit_precisions_95": hit_precisions_95
        })

    print(f"Exhaustive days: {exhaustive_days}")
    for s, d in zip(steps, skip_days):
        speedup = exhaustive_time / skip_time[steps.index(s)]
        print(f"Skip {s} days: {d}, precision: {d/exhaustive_days:.3f}, speedup: {speedup:.1f}")
        
    for i, step in enumerate(steps):
        overall_hit_precisions_25 =  []
        overall_hit_precisions_50 =  []
        overall_hit_precisions_75 =  []
        overall_hit_precisions_95 =  []
        for r in results:
            overall_hit_precisions_25.append(r["hit_precisions_25"][i])
            overall_hit_precisions_50.append(r["hit_precisions_50"][i])
            overall_hit_precisions_75.append(r["hit_precisions_75"][i])
            overall_hit_precisions_95.append(r["hit_precisions_95"][i])

        overall_hit_precisions_25 = np.mean(overall_hit_precisions_25)
        overall_hit_precisions_50 = np.mean(overall_hit_precisions_50)
        overall_hit_precisions_75 = np.mean(overall_hit_precisions_75)
        overall_hit_precisions_95 = np.mean(overall_hit_precisions_95)

        print(f"{step} - Hit Precision 25: {overall_hit_precisions_25:.3f} - Hit Precision 50: {overall_hit_precisions_50:.3f} - Hit Precision 75: {overall_hit_precisions_75:.3f} - Hit Precision 95: {overall_hit_precisions_95:.3f}")

    # Display results for all grid points as boxplots
    precisions = np.array([r["precisions"] for r in results])

    positions = np.linspace(1, 1.3, len(steps))[::-1]
    plt.figure(figsize=(14, 8))  
    plt.boxplot(precisions, positions=positions, widths=0.07, patch_artist=True, 
                boxprops=dict(facecolor='lightgray', color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))


    plt.axhline(0.95, color='darkred', linestyle='--', linewidth=2, label='Target: 95\%')
    plt.xlim(0.95,1.55)
    plt.xlabel("Skip value", fontsize=24)
    plt.ylabel("Precision", fontsize=24)
    plt.xticks(positions, labels=steps, fontsize=22)
    plt.yticks(fontsize=22)

    plt.grid(True, axis="y", which='both', linestyle='--', linewidth=1, alpha=0.7)
    plt.legend(fontsize=22, loc='lower right')
    plt.tight_layout()
    plt.show()
    
    # # Display results for all grid points as boxplots
    # hit_precisions = np.array([r["hit_precisions"] for r in results])
    # plt.figure(figsize=(14, 10))
    # plt.boxplot(hit_precisions, positions=steps)
    # plt.hlines(0.95, steps[0], steps[-1], colors='r', linestyles='dashed', label='95%')
    # plt.title("Hit Precision at All Grid Points")
    # plt.ylabel("Hit Precision")
    # plt.xlabel("Constant value for linear skip")
    # plt.legend()
    # plt.show()


def all_grid_eu():

    # Load the SST dataset
    ds = xr.open_dataset("Data/ERA5_land/EUW/t2m_daily_avg_1950_2023.nc")
    ds = ds.t2m.values.astype(np.float64)

    # Extract grid dimensions
    n_lat, n_lon = ds.shape[1:]


    indices = []
    tmp = ds[1, :, :]  
    for i in range(n_lat):
        for j in range(n_lon):
            if not np.isnan(tmp[i, j]):
                indices.append((i, j))

    # Parameters
    threshold = 3
    len_year = 365
    min_length = 1
    max_length = 365

    results = []
    steps = [20, 10, 5, 1] 
    exhaustive_days = 0
    exhaustive_time = 0.0
    skip_time = [0.0 for _ in steps]

    skip_days = [0 for _ in steps]
    for (i, j) in tqdm(indices):
        ts = ds[:, i, j] 
        n_year = len(ts) // len_year

        burst_exhaustive, time_exhaustive = timeit(
            nexhaustive.compute_all_bursts,
            ts,
            n_year,
            threshold,
            len_year,
            min_length,
            max_length
        )
        exhaustive_time += time_exhaustive

        if len(burst_exhaustive) != 0:

            days_exhaustive = bursts_to_days(burst_exhaustive)
            merge_exhaustive = pp.merge_sequences(burst_exhaustive)
            exhaustive_days += len(days_exhaustive)

        precisions = []
        hit_precisions = []
        for s in steps:
            bursts, time = timeit(
                nexhaustive_skip_lengths.compute_all_bursts,
                ts,
                n_year,
                threshold,
                len_year,
                s,
                min_length,
                max_length
            )
            skip_time[steps.index(s)] += time
            if len(bursts) == 0:
                continue
            merge_bursts = pp.merge_sequences(bursts)
            detected_days = bursts_to_days(bursts)
            precision_score = precision(burst_exhaustive, days_exhaustive, bursts, detected_days)
            hit_prec = hit_precision(merge_exhaustive, merge_bursts)

            precisions.append(precision_score)
            hit_precisions.append(hit_prec)

            skip_days[steps.index(s)] += len(detected_days)

        results.append({
            "lat": i,
            "lon": j,
            "precisions": precisions,
            "hit_precisions": hit_precisions
        })

    print(f"Exhaustive days: {exhaustive_days}")
    for s, d in zip(steps, skip_days):
        speedup = exhaustive_time / skip_time[steps.index(s)]
        print(f"Skip {s} days: {d}, precision: {d/exhaustive_days:.3f}, speedup: {speedup:.1f}")
        


    # Display results for all grid points as boxplots
    precisions = np.array([r["precisions"] for r in results])
    plt.figure(figsize=(14, 10))
    plt.boxplot(precisions, positions=steps)
    plt.hlines(0.95, steps[0], steps[-1], colors='r', linestyles='dashed', label='95%') 
    plt.title("Precision at All Grid Points")
    plt.ylabel("Precision")
    plt.xlabel("Constant value for linear skip")
    plt.legend()
    plt.show()

    
    # Display results for all grid points as boxplots
    hit_precisions = np.array([r["hit_precisions"] for r in results])
    plt.figure(figsize=(14, 10))
    plt.boxplot(hit_precisions, positions=steps)
    plt.hlines(0.95, steps[0], steps[-1], colors='r', linestyles='dashed', label='95%')
    plt.title("Hit Precision at All Grid Points")
    plt.ylabel("Hit Precision")
    plt.xlabel("Constant value for linear skip")
    plt.legend()
    plt.show()

def all_grid_eu_preci():

    # Load the SST dataset
    ds = xr.open_dataset("Data/ERA5_land/EUW/tp_daily_sum_1950_2023.nc")
    ds = ds.tp.values.astype(np.float64)

    # Extract grid dimensions
    n_lat, n_lon = ds.shape[1:]


    indices = []
    tmp = ds[1, :, :]  
    indices = [(i,j) for j in range(n_lon) for i in range(n_lat) if  not np.all(ds[:,i,j]==0)]

    # Parameters
    threshold = 4
    len_year = 365
    min_length = 14
    max_length = 365

    results = []
    steps = [20, 5] 
    exhaustive_days = 0
    exhaustive_time = 0.0
    skip_time = [0.0 for _ in steps]


    nb_burst_per_length = {l: 0 for l in range(min_length, max_length+1)}
    nb_burst_per_length_skip = [{l: 0 for l in range(min_length, max_length+1)} for _ in steps]

    skip_days = [0 for _ in steps]
    for (i, j) in tqdm(indices):
        ts = ds[:, i, j] 
        n_year = len(ts) // len_year

        burst_exhaustive, time_exhaustive = timeit(
            nexhaustive.compute_all_bursts,
            ts,
            n_year,
            threshold,
            len_year,
            min_length,
            max_length
        )
        exhaustive_time += time_exhaustive

        if len(burst_exhaustive) == 0:
            continue
        else:
            days_exhaustive = bursts_to_days(burst_exhaustive)
            merge_exhaustive = pp.merge_sequences(burst_exhaustive)
            exhaustive_days += len(days_exhaustive)

            for burst in burst_exhaustive:
                nb_burst_per_length[burst[1]-burst[0]+1] += 1

            precisions = []
            hit_precisions = []
            for i, s in enumerate(steps):
                bursts, time = timeit(
                    nexhaustive_skip_lengths.compute_all_bursts,
                    ts,
                    n_year,
                    threshold,
                    len_year,
                    s,
                    min_length,
                    max_length
                )
                skip_time[steps.index(s)] += time

                if len(bursts) == 0:
                    precisions.append(0)
                    hit_precisions.append(0)
                    continue
                else:
                    merge_bursts = pp.merge_sequences(bursts)
                    detected_days = bursts_to_days(bursts)
                    precision_score = precision(days_exhaustive, detected_days)
                    hit_prec = hit_precision(merge_exhaustive, merge_bursts)

                    precisions.append(precision_score)
                    hit_precisions.append(hit_prec)

                    skip_days[steps.index(s)] += len(detected_days)

                    for burst in bursts:
                        nb_burst_per_length_skip[i][burst[1]-burst[0]+1] += 1

        results.append({
            "lat": i,
            "lon": j,
            "precisions": precisions,
            "hit_precisions": hit_precisions
        })



    # plot the number of bursts per length
    plt.figure(figsize=(14, 10))
    plt.bar(nb_burst_per_length.keys(), nb_burst_per_length.values(), label="Exhaustive")
    plt.bar(nb_burst_per_length_skip[0].keys(), nb_burst_per_length_skip[0].values(), label="Skip")
    plt.title("Number of Bursts per Length")
    plt.xlabel("Length")
    plt.ylabel("Number of Bursts")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 10))
    plt.bar(nb_burst_per_length.keys(), nb_burst_per_length.values(), label="Exhaustive")
    plt.bar(nb_burst_per_length_skip[1].keys(), nb_burst_per_length_skip[1].values(), label="Skip")
    plt.title("Number of Bursts per Length")
    plt.xlabel("Length")
    plt.ylabel("Number of Bursts")
    plt.legend()
    plt.show()

    print(f"Exhaustive days: {exhaustive_days}")
    for s, d in zip(steps, skip_days):
        speedup = exhaustive_time / skip_time[steps.index(s)]
        print(f"Skip {s} days: {d}, precision: {d/exhaustive_days:.3f}, speedup: {speedup:.1f}")
        


    # Display results for all grid points as boxplots
    precisions = np.array([r["precisions"] for r in results])
    plt.figure(figsize=(14, 10))
    plt.boxplot(precisions, positions=steps)
    plt.hlines(0.95, steps[0], steps[-1], colors='r', linestyles='dashed', label='95%') 
    plt.title("Precision at All Grid Points")
    plt.ylabel("Precision")
    plt.xlabel("Constant value for linear skip")
    plt.legend()
    plt.show()

    
    # Display results for all grid points as boxplots
    hit_precisions = np.array([r["hit_precisions"] for r in results])
    plt.figure(figsize=(14, 10))
    plt.boxplot(hit_precisions, positions=steps)
    plt.hlines(0.95, steps[0], steps[-1], colors='r', linestyles='dashed', label='95%')
    plt.title("Hit Precision at All Grid Points")
    plt.ylabel("Hit Precision")
    plt.xlabel("Constant value for linear skip")
    plt.legend()
    plt.show()


def all_grid_eu_dates():

    # Load the SST dataset
    ds = xr.open_dataset("Data/ERA5_land/EUW/t2m_daily_avg_1950_2023.nc")
    ds = ds.t2m.values.astype(np.float64)

    # Extract grid dimensions
    n_lat, n_lon = ds.shape[1:]


    indices = []
    tmp = ds[1, :, :]  
    for i in range(n_lat):
        for j in range(n_lon):
            if not np.isnan(tmp[i, j]):
                indices.append((i, j))

    # Parameters
    threshold = 3
    len_year = 365
    min_length = 1
    max_length = 365

    results = []
    steps = [10, 1] 
    exhaustive_days = 0
    exhaustive_time = 0.0
    skip_time = [0.0 for _ in steps]

    skip_days = [0 for _ in steps]
    for (i, j) in tqdm(indices[::10]):
        ts = ds[:, i, j] 
        n_year = len(ts) // len_year

        burst_exhaustive, time_exhaustive = timeit(
            nexhaustive.compute_all_bursts,
            ts,
            n_year,
            threshold,
            len_year,
            min_length,
            max_length
        )
        exhaustive_time += time_exhaustive

        if len(burst_exhaustive) != 0:

            days_exhaustive = bursts_to_days(burst_exhaustive)
            merge_exhaustive = pp.merge_sequences(burst_exhaustive)
            exhaustive_days += len(days_exhaustive)

        precisions = []
        hit_precisions = []
        for s in steps:
            bursts, time = timeit(
                nexhaustive_skip_dates.compute_all_bursts,
                ts,
                n_year,
                threshold,
                len_year,
                s,
                min_length,
                max_length
            )
            skip_time[steps.index(s)] += time
            if len(bursts) == 0:
                continue
            merge_bursts = pp.merge_sequences(bursts)
            detected_days = bursts_to_days(bursts)
            precision_score = precision(days_exhaustive, detected_days)
            hit_prec = hit_precision(merge_exhaustive, merge_bursts)

            precisions.append(precision_score)
            hit_precisions.append(hit_prec)

            skip_days[steps.index(s)] += len(detected_days)

        results.append({
            "lat": i,
            "lon": j,
            "precisions": precisions,
            "hit_precisions": hit_precisions
        })

    print(f"Exhaustive days: {exhaustive_days}")
    for s, d in zip(steps, skip_days):
        speedup = exhaustive_time / skip_time[steps.index(s)]
        print(f"Skip {s} days: {d}, precision: {d/exhaustive_days:.3f}, speedup: {speedup:.1f}")
        


    # Display results for all grid points as boxplots
    precisions = np.array([r["precisions"] for r in results])
    plt.figure(figsize=(14, 10))
    plt.boxplot(precisions, positions=steps)
    plt.hlines(0.95, steps[0], steps[-1], colors='r', linestyles='dashed', label='95%') 
    plt.title("Precision at All Grid Points")
    plt.ylabel("Precision")
    plt.xlabel("Constant value for linear skip")
    plt.legend()
    plt.show()

    
    # Display results for all grid points as boxplots
    hit_precisions = np.array([r["hit_precisions"] for r in results])
    plt.figure(figsize=(14, 10))
    plt.boxplot(hit_precisions, positions=steps)
    plt.hlines(0.95, steps[0], steps[-1], colors='r', linestyles='dashed', label='95%')
    plt.title("Hit Precision at All Grid Points")
    plt.ylabel("Hit Precision")
    plt.xlabel("Constant value for linear skip")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    date_skipping_scaling()
    length_skipping_scaling()
    both_skipping_scaling()

    # all_grid_medi()
    # all_grid_eu()
    # all_grid_eu_preci()
    # all_grid_eu_dates()