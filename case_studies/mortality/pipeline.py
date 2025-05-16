"""
Usage:  python case_studies/mortality/pipeline.py
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from utils.post_process import merge_bursts
from seasonal_burst.exhaustive_search import exhaustive_search_intra_period

import matplotlib.pyplot as plt
import itertools

import random


def has_k_consecutive_above_threshold(arr, threshold, k):
    """
    Check if there are k consecutive values in the array that are above the threshold.

    Parameters:
    arr (np.ndarray): The input array.
    threshold (float): The threshold value.
    k (int): The number of consecutive values to check for.

    Returns:
    bool: True if there are k consecutive values above the threshold, False otherwise.
    """

    above_threshold = arr > threshold
    return np.any(np.convolve(above_threshold, np.ones(3, dtype=int), mode='valid') == k)


def sig_test(burst_set, ts_date_days, ts_mort, start_year=1974, end_year=2023, N=10000, lag=14, excludes=[1997]):
    """
    Perform a significance test to determine if the observed mortality increase after burst dates is significant.

    Parameters:
    burst_set (pd.DataFrame): DataFrame containing burst dates and their corresponding mortality data.
    ts_date_days (pd.Series): Series containing the time series dates.
    ts_mort (pd.Series): Series containing the time series mortality data.
    start_year (int): The starting year for the simulation.
    end_year (int): The ending year for the simulation.
    N (int): The number of simulations to perform.
    lag (int): The number of days to consider for the mortality increase.
    excludes (list): List of years to exclude from the simulation.

    Returns:
    tuple: A tuple containing the count of simulations with mortality increase greater than or equal to the observed value,
           and the proportion of such simulations--the empirical p-value.
    """

    ts_dates_normalized = ts_date_days.to_numpy().astype('datetime64[D]')
    date_to_index = {date: idx for idx, date in enumerate(ts_dates_normalized)}

    origin_sum = np.add.reduce(burst_set[f"mortality_{lag}"])

    burst_dates = burst_set['date'].dt.floor(
        'D').to_numpy().astype('datetime64[D]')
    indices = np.array([date_to_index.get(date) for date in burst_dates])
    # Remove invalid indices (if any)
    valid_mask = indices != None
    valid_indices = indices[valid_mask]
    # Prepare for simulation
    valid_burst_dates = burst_dates[valid_mask]
    day_month_pairs = pd.to_datetime(
        valid_burst_dates).strftime('%m-%d').tolist()
    years = np.arange(start_year, end_year+1)

    for year in excludes:
        if year in years:
            years = np.delete(years, np.where(years == year)[0][0])

    count = 0
    for i in range(N):
        rsum = 0.0
        for dm in day_month_pairs:
            ryear = random.choice(years)
            rdate = np.datetime64(f"{ryear}-{dm}", 'D')
            ind = date_to_index.get(rdate)
            if ind is not None:
                rsum += np.add.reduce(ts_mort[ind:ind+lag])
        if rsum >= origin_sum:
            count += 1

    return count, count / N


def bootstrap_diff_confidence_interval(burst_set, ts_date_days, ts_mort, num_bootstrap, ci, start_year, end_year, lag):
    """
    Calculate the bootstrap confidence interval for the difference in mortality rates.
    This function computes the difference in mortality rates between the observed burst dates and the average rates
    for the same dates in other years, and then bootstraps to find the confidence interval.
    Parameters:
    burst_set (pd.DataFrame): DataFrame containing burst dates and their corresponding mortality data.
    ts_date_days (pd.Series): Series containing the time series dates.
    ts_mort (pd.Series): Series containing the time series mortality data.
    num_bootstrap (int): Number of bootstrap samples to generate.
    ci (float): Confidence interval percentage.
    start_year (int): The starting year for the simulation.
    end_year (int): The ending year for the simulation.     
    lag (int): The number of days to consider for the mortality increase.

    Returns:
    dict: A dictionary containing the bootstrap means, the mean difference, the confidence interval,
          and the differences for each burst date.
    """

    # Normalize date array
    ts_dates_normalized = ts_date_days.to_numpy().astype('datetime64[D]')
    date_to_index = {date: idx for idx, date in enumerate(ts_dates_normalized)}

    burst_dates = burst_set['date'].dt.floor(
        'D').to_numpy().astype('datetime64[D]')
    years = np.arange(start_year, end_year+1)
    if start_year <= 1997 and end_year >= 1997:
        years = np.delete(years, np.where(years == 1997)[0][0])
    n_years = len(years)
    N = len(burst_dates)
    diffs = np.zeros(N)

    for i, bdate in enumerate(burst_dates):
        ind = date_to_index[bdate]
        # 14-day rate at burst year
        rate_y = np.add.reduce(ts_mort[ind:ind+lag])

        # Extract day and month
        month = pd.to_datetime(bdate).month
        day = pd.to_datetime(bdate).day

        rates_all_years = 0
        for y in years:
            comp_date = np.datetime64(f"{y}-{month:02}-{day:02}", 'D')
            ind_y = date_to_index.get(comp_date)
            rates_all_years += np.add.reduce(ts_mort[ind_y:ind_y+lag])

        avg_rate = rates_all_years / n_years
        diffs[i] = rate_y - avg_rate

    # Bootstrap: 10,000 means
    boot_means = np.array([
        np.mean(np.random.choice(diffs, size=N, replace=True))
        for _ in range(num_bootstrap)
    ])

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)


    return {
        "boot_means": boot_means,
        "mean_diff": np.mean(diffs),
        f"{ci}%_CI": (lower, upper),
        "diffs": diffs.tolist()
    }


def bootstrap_diff_confidence_interval2(burst_set, ts_date_days, ts_mort_4, num_bootstrap, ci, start_year, end_year, lag):
    # Normalize date array
    ts_dates_normalized = ts_date_days.to_numpy().astype('datetime64[D]')
    date_to_index = {date: idx for idx, date in enumerate(ts_dates_normalized)}


    burst_dates = burst_set['date'].dt.floor('D').to_numpy().astype('datetime64[D]')
    years = np.arange(start_year, end_year + 1)
    if start_year <= 1997 and end_year >= 1997:
        years = np.delete(years, np.where(years == 1997)[0][0])
    n_years = len(years)
    N = len(burst_dates)
    diffs = np.zeros(N)
    percent_increases = np.zeros(N)

    for i, bdate in enumerate(burst_dates):
        ind = date_to_index[bdate]

        rate_y = np.add.reduce(ts_mort_4[ind:ind + lag])  # 14-day rate at burst year

        # Extract day and month
        month = pd.to_datetime(bdate).month
        day = pd.to_datetime(bdate).day


        rates_all_years = 0
        for y in years:
            comp_date = np.datetime64(f"{y}-{month:02}-{day:02}", 'D')
            ind_y = date_to_index.get(comp_date)

            rates_all_years += np.add.reduce(ts_mort_4[ind_y:ind_y + lag])

        avg_rate = rates_all_years / n_years
        diffs[i] = rate_y - avg_rate
        percent_increases[i] = ((rate_y - avg_rate) / avg_rate) * 100
        # diffs[i] = percent_increases[i]

    # print(diffs)

    # Bootstrap: 10,000 means
    boot_means = np.array([
        np.mean(np.random.choice(diffs, size=N, replace=True))
        for _ in range(num_bootstrap)
    ])

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)

    avg_percent_increase = np.average(percent_increases)
    med_percent_increase = np.median(percent_increases)

    return {
        "boot_means": boot_means,
        "mean_diff": np.mean(diffs),
        f"{ci}%_CI": (lower, upper),
        "diffs": diffs.tolist(),
        "avg_percent_increase": avg_percent_increase,
        "med_percent_increase": med_percent_increase,
    }


def compute_lag_mortality(burst_set, ts_mort_jjas, lag, mort_start_year):
    """
    Compute the lagged mortality (sum over the 'lag' days after the date) for a given burst set and time series mortality data.

    Parameters:
    burst_set (pd.DataFrame): DataFrame containing burst dates and their corresponding mortality data.
    ts_mort_jjas (pd.Series): Series containing the time series mortality data for June to September.
    lag (int): The number of days to consider for the mortality increase.
    mort_start_year (int): The starting year for the mortality data.

        Returns:
    np.ndarray: An array containing the lagged mortality values for each burst date.
    """

    ts_date_days = pd.date_range(
        f'01-Jan-{mort_start_year}', '31-Dec-2023', freq='D')
    ts_date_days = ts_date_days[ts_date_days.month.isin([6, 7, 8, 9])]
    ts_dates_normalized = ts_date_days.to_numpy().astype('datetime64[D]')
    date_to_index = {date: idx for idx, date in enumerate(ts_dates_normalized)}
    # Normalize burst dates
    burst_dates = burst_set['date'].dt.floor(
        'D').to_numpy().astype('datetime64[D]')
    indices = np.array([date_to_index.get(date) for date in burst_dates])


    # Remove invalid indices (if any)
    valid_mask = indices != None
    valid_indices = indices[valid_mask]
    mortality = np.array([np.add.reduce(ts_mort_jjas[i:i+lag])
                         for i in valid_indices])
    

    return mortality



def build_data_frame(df, ts_mort_jja, ts_mort_jjas, lags, start_year, end_year):

    df_daily = df.resample("D", on="date").sum().reset_index().dropna()
    df_daily = df_daily[df_daily['date'].dt.month.isin([6, 7, 8])]



    df_daily = df_daily[df_daily['date'].dt.year >= start_year]
    df_daily = df_daily[df_daily['date'].dt.year <= end_year]


    df_daily['rate'] = ts_mort_jja



    df = df_daily[df_daily['events'] >= 1].reset_index()

    for lag in lags:
        df[f'mortality_{lag}'] = compute_lag_mortality(
            df, ts_mort_jjas, lag, start_year)

    return df


def exp_burst(ts_climate, ts_date, lag, start_year, threshold, minimum_length, maximum_length, exclude_years=[1997]):

    end_year = 2023
    n_year = end_year - start_year + 1

    len_year = 92 * 24

    

    ts_mort_jja = np.load("Data/mortality/death_rate_jja.npy").astype(np.float64)[(start_year-1975)*92:] * 10000
    ts_mort_jjas = np.load("Data/mortality/death_rate_jjas.npy").astype(np.float64)[(start_year-1975)*122:] * 10000

    mask = (ts_date.year >= start_year) & (ts_date.year <= end_year)
    ts_climate = ts_climate[mask]
    ts_date = ts_date[mask]


    ts_date_days = pd.date_range(f'01-Jan-{start_year}', f'31-Dec-{end_year}', freq='D')
    ts_date_days = ts_date_days[ts_date_days.month.isin([6, 7, 8, 9])]



    bursts = exhaustive_search_intra_period(ts_climate, n_year, threshold, len_year, minimum_length, maximum_length)
    bursts_merged = merge_bursts(bursts)
    bursts = [b for b in bursts_merged if b[2] >0]

    print(f"Number of bursts: {len(bursts)}")
    # print(bursts)

    # l = []
    # for (s,e,z) in bursts:
    #     if has_k_consecutive_above_threshold(ts_climate[s:e+1], 30, 1):
    #         l.append((s,e,z))
    # bursts = l
    # print(f"Number of bursts with 1 day above 30C: {len(bursts)}")

    ts_events = np.zeros_like(ts_climate)
    for (s, e, z) in bursts:
        ts_events[s:e+1] = 1


    df = pd.DataFrame({
        'date': ts_date,
        'events': ts_events,
    })

    
    df_bursts = build_data_frame(df, ts_mort_jja, ts_mort_jjas, [lag], start_year, end_year) 
    for year in exclude_years:
        df_bursts = df_bursts[df_bursts['date'].dt.year != year]

    # print(df_bursts)


    # Significance test
    count, p_value = sig_test(df_bursts, ts_date_days, ts_mort_jjas, start_year, end_year, N=10000, lag=lag)
    print(f"Significance test: {count} out of 10000 simulations had a mortality increase greater than or equal to the observed value. p-value: {p_value:.4f}")
    # Bootstrap confidence interval
    bootstrap_results = bootstrap_diff_confidence_interval2(df_bursts, ts_date_days, ts_mort_jjas, num_bootstrap=10000, ci=95, start_year=start_year, end_year=end_year, lag=lag)
    print(f"Bootstrap confidence interval: {bootstrap_results[f'95%_CI'][0]:.4f} - {bootstrap_results[f'95%_CI'][1]:.4f}")
    print(f"Mean difference: {bootstrap_results['mean_diff']:.4f}")
    print(f"Bootstrap mean: {np.mean(bootstrap_results['boot_means']):.4f}")
    print(f"Median percent increase: {bootstrap_results['med_percent_increase']:.4f}%")
    print(f"Mean percent increase: {bootstrap_results['avg_percent_increase']:.4f}%")


def absolute_exp(ts_climate, ts_date, threshold, lag):

    start_year = 1975
    end_year = 2023

    ts_mort_jja = np.load("Data/mortality/death_rate_jja.npy").astype(np.float64)[(start_year-1975)*92:] * 10000
    ts_mort_jjas = np.load("Data/mortality/death_rate_jjas.npy").astype(np.float64) * 10000

    mask = (ts_date.year >= start_year) & (ts_date.year <= end_year)
    ts_climate = ts_climate[mask]
    ts_date = ts_date[mask]


    ts_date_days = pd.date_range(f'01-Jan-{start_year}', f'31-Dec-{end_year}', freq='D')
    ts_date_days = ts_date_days[ts_date_days.month.isin([6, 7, 8, 9])]


    ts_events = ( ts_climate >= threshold).astype(int)

    df = pd.DataFrame({
        'date': ts_date,
        'events': ts_events,
    })

    lags = [lag]
    df_bursts = build_data_frame(df, ts_mort_jja, ts_mort_jjas, lags, start_year, end_year)

    # Significance test
    count, p_value = sig_test(df_bursts, ts_date_days, ts_mort_jjas, start_year, end_year, N=10000, lag=lag)
    print(f"Significance test: {count} out of 10000 simulations had a mortality increase greater than or equal to the observed value. p-value: {p_value:.4f}")
    # Bootstrap confidence interval
    # bootstrap_results = bootstrap_diff_confidence_interval(df_bursts, ts_date_days, ts_mort_jjas, num_bootstrap=10000, ci=95, start_year=start_year, end_year=end_year, lag=lag)
    bootstrap_results = bootstrap_diff_confidence_interval2(df_bursts, ts_date_days, ts_mort_jjas, num_bootstrap=10000, ci=95, start_year=start_year, end_year=end_year, lag=lag)
    print(f"Bootstrap confidence interval: {bootstrap_results[f'95%_CI'][0]:.4f} - {bootstrap_results[f'95%_CI'][1]:.4f}")
    print(f"Mean difference: {bootstrap_results['mean_diff']:.4f}")
    print(f"Bootstrap mean: {np.mean(bootstrap_results['boot_means']):.4f}")
    print(f"Median percent increase: {bootstrap_results['med_percent_increase']:.4f}%")
    print(f"Mean percent increase: {bootstrap_results['avg_percent_increase']:.4f}%")


def exp_burst_grid_search(ts_climate, ts_date, lag_range, threshold_range, min_len_range, max_len_range, exclude_years=[1997]):
    start_year = 1998
    end_year = 2023
    n_year = end_year - start_year + 1
    len_year = 92 * 24

    ts_mort_jja = np.load("death_rate_jja.npy").astype(np.float64)[(start_year-1975)*92:] * 10000
    ts_mort_jjas = np.load("death_rate_jjas.npy").astype(np.float64)[(start_year-1975)*122:] * 10000

    mask = (ts_date.year >= start_year) & (ts_date.year <= end_year)
    ts_climate = ts_climate[mask]
    ts_date = ts_date[mask]

    ts_date_days = pd.date_range(f'01-Jan-{start_year}', f'31-Dec-{end_year}', freq='D')
    ts_date_days = ts_date_days[ts_date_days.month.isin([6, 7, 8, 9])]

    best_result = None
    best_combo = None

    for lag, threshold, min_len, max_len in itertools.product(lag_range, threshold_range, min_len_range, max_len_range):
        print(f"Testing combination: lag={lag}, threshold={threshold}, min_len={min_len}, max_len={max_len}")
        bursts = exhaustive_search_intra_period(ts_climate, n_year, threshold, len_year, min_len, max_len)
        bursts_merged = merge_bursts(bursts)
        bursts = [b for b in bursts_merged if b[2] > 0]

        # l = []
        # for (s, e, z) in bursts:
        #     if has_k_consecutive_above_threshold(ts_climate[s:e+1], 30, 1):
        #         l.append((s, e, z))
        # bursts = l

        ts_events = np.zeros_like(ts_climate)
        for (s, e, z) in bursts:
            ts_events[s:e+1] = 1

        df = pd.DataFrame({
            'date': ts_date,
            'events': ts_events,
        })

        try:
            df_bursts = build_data_frame(df, ts_mort_jja, ts_mort_jjas, [lag], start_year, end_year)

            for year in exclude_years:
                df_bursts = df_bursts[df_bursts['date'].dt.year != year]
            count, p_value = sig_test(df_bursts, ts_date_days, ts_mort_jjas, start_year, end_year, N=10000, lag=lag)

            if p_value > 0.05:
                print("Skipping this combination due to high p-value.")
                continue
            bootstrap_results = bootstrap_diff_confidence_interval2(df_bursts, ts_date_days, ts_mort_jjas, num_bootstrap=10000, ci=95, start_year=start_year, end_year=end_year, lag=lag)
            score = bootstrap_results['med_percent_increase']  # Objective: minimize p-value (you could change this)
            print(f"Score: {score:.4f}, p-value: {p_value:.4f}")
            if best_result is None or score > best_result['score']:
                best_result = {
                    'score': score,
                    'lag': lag,
                    'threshold': threshold,
                    'min_len': min_len,
                    'max_len': max_len,
                    'p_value': p_value,
                    'ci': bootstrap_results['95%_CI'],
                    'mean_diff': bootstrap_results['mean_diff'],
                    'percent_increase': bootstrap_results['med_percent_increase']
                }
                best_combo = (lag, threshold, min_len, max_len)
        except Exception as e:
            print(f"Failed on combination (lag={lag}, threshold={threshold}, min_len={min_len}, max_len={max_len}): {e}")

    print("\nBest combination found:")
    print(best_result, best_combo)
    return best_result

if __name__ == "__main__":
    # Set the random seed for reproducibility
    # np.random.seed(42)
    # random.seed(42)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Run the main function

    ds = xr.open_dataset("Data/Paris/wgbt_era5_Paris_1974-2023jja.nc")
    ts_date = pd.DatetimeIndex(ds.time.values)


    ts_wbgt = ds.wbgt.values.ravel().astype(np.float64)

    ts_tas = xr.open_dataset("Data/Paris/ta_era5_Paris_1974-2023jja.nc").t2m.values.ravel().astype(np.float64) - 273.15



    lag_range = [7]
    threshold_range = [3]
    min_len_range = [2, 4, 12]
    max_len_range = [24, 48, 72, 96]

    # best = exp_burst_grid_search(ts_tas, ts_date, lag_range, threshold_range, min_len_range, max_len_range)

    # best = exp_burst_grid_search(ts_tas, ts_date, lag_range, threshold_range, min_len_range, max_len_range, exclude_years=[1997, 2003])

    # best = exp_burst_grid_search(ts_wbgt, ts_date, lag_range, threshold_range, min_len_range, max_len_range, exclude_years=[1997, 2003])
    
    # print("Temperature Experiment")
    # exp_burst(ts_tas, ts_date, 5)

    exp_burst(ts_tas, ts_date, 7, 1998, 3, 3, 72, exclude_years=[1997])
    exp_burst(ts_tas, ts_date, 7, 1998, 3, 3, 72, exclude_years=[1997, 2003])
    # print("")
    # exp_burst(ts_tas, ts_date, 14)
    # print("")
    # print("WBGT Experiment")
    # exp_burst(ts_wbgt, ts_date, 7)
    # print("")
    # exp_burst(ts_wbgt, ts_date, 14)
    # print("")


    # print("Absolute Temperature Experiment")
    # absolute_exp(ts_tas, ts_date, 34, 14)
    # print("")
    # print("Absolute WBGT Experiment")
    # absolute_exp(ts_wbgt, ts_date, 28, 14)
    # print("Done")

