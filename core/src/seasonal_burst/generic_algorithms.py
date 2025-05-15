import numpy as np


def exhaustive_search(
    ts, n_year, threshold, len_year, agg_fn, min_length=1, max_length=365
):
    """ """
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = []

    # Finding burst of length 1 (day) only the z-score are needed

    for length in range(min_length, max_length + 1):
        for year in range(n_year):
            for obs in range(len_year):
                start = year * len_year + obs
                end = start + length
                if end > n:
                    break
                v[year, obs] = agg_fn(ts[start:end])

        max_sequence = len_year - length + 1
        n_valid_years = n_year
        for interval in range(max_sequence):
            column = v[:n_valid_years, interval]
            mean = np.add.reduce(column) / n_valid_years
            squared_mean = np.add.reduce(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean * mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append((start, start + length - 1, z_score[y]))

        # Second loop for the condition: interval >= max_sequence
        # for those intervals the last valid year is excluded
        n_valid_years -= 1
        for interval in range(max_sequence, len_year):
            column = v[:n_valid_years, interval]
            mean = np.add.reduce(column) / n_valid_years
            squared_mean = np.add.reduce(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean * mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append((start, start + length - 1, z_score[y]))

    return all_bursts



def exhaustive_search_percentile(
    ts, n_year, percentile_threshold, len_year, agg_fn, min_length=1, max_length=365
):
    """
    Perform exhaustive search using a percentile-based threshold instead of z-score.

    Args:
        ts: Time series as a 1D numpy array.
        n_year: Number of years in the dataset.
        percentile_threshold: The percentile threshold (e.g. 95 for 95th percentile).
        len_year: Number of observations per year.
        agg_fn: Aggregation function for each sequence (e.g., np.sum, np.mean).
        min_length: Minimum length of sequences to evaluate.
        max_length: Maximum length of sequences to evaluate.
    """
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = []

    for length in range(min_length, max_length + 1):
        for year in range(n_year):
            for obs in range(len_year):
                start = year * len_year + obs
                end = start + length
                if end > n:
                    break
                v[year, obs] = agg_fn(ts[start:end])

        max_sequence = len_year - length + 1
        n_valid_years = n_year
        for interval in range(max_sequence):
            column = v[:n_valid_years, interval]
            thresh = np.percentile(column, percentile_threshold)
            flagged_years = np.where(column >= thresh)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append((start, start + length - 1, column[y]))

        n_valid_years -= 1
        for interval in range(max_sequence, len_year):
            column = v[:n_valid_years, interval]
            thresh = np.percentile(column, percentile_threshold)
            flagged_years = np.where(column >= thresh)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append((start, start + length - 1, column[y]))

    return all_bursts


def skipping_length(
    ts, n_year, threshold, len_year, agg_fn, skip_factor, min_length=1, max_length=365
):
    """ """
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = []

    # Finding burst of length 1 (day) only the z-score are needed

    length = min_length
    while length <= max_length:
        for year in range(n_year):
            for obs in range(len_year):
                start = year * len_year + obs
                end = start + length
                if end > n:
                    break
                v[year, obs] = agg_fn(ts[start:end])

        max_sequence = len_year - length + 1
        n_valid_years = n_year
        for interval in range(max_sequence):
            column = v[:n_valid_years, interval]
            mean = np.add.reduce(column) / n_valid_years
            squared_mean = np.add.reduce(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean * mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append((start, start + length - 1, z_score[y]))

        # Second loop for the condition: interval >= max_sequence
        # for those intervals the last valid year is excluded
        n_valid_years -= 1
        for interval in range(max_sequence, len_year):
            column = v[:n_valid_years, interval]
            mean = np.add.reduce(column) / n_valid_years
            squared_mean = np.add.reduce(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean * mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append((start, start + length - 1, z_score[y]))

        length += max(1, length // skip_factor)

    return all_bursts


def skipping_date(
    ts, n_year, threshold, len_year, agg_fn, skip_factor, min_length=1, max_length=365
):
    """ """
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = []

    # Finding burst of length 1 (day) only the z-score are needed

    for length in range(min_length, max_length + 1):

        step = max(1, length // skip_factor)
        for year in range(n_year):
            for obs in range(0, len_year, step):
                start = year * len_year + obs
                end = start + length
                if end > n:
                    break
                v[year, obs] = agg_fn(ts[start:end])

        max_sequence = len_year - length + 1
        for interval in range(0, len_year, step):

            if interval < max_sequence:
                n_valid_years = n_year
            else:
                n_valid_years = n_year - 1

            column = v[:n_valid_years, interval]
            mean = np.add.reduce(column) / n_valid_years
            squared_mean = np.add.reduce(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean * mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append((start, start + length - 1, z_score[y]))

    return all_bursts
