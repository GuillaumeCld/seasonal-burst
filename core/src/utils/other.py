import numpy as np
from numba import float64, int64, njit


@njit((float64[:], int64, int64, int64, int64))
def compute_daily_zscore(ts, n_year, len_year, min_length, max_length):
    """
    Compute the z-score for each day in the time series.
    """

    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    z_per_day = np.zeros((len(ts)))

    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        for i in range(len_year):
            column = v[:n_year, i]
            mean = np.sum(column) / n_year
            squared_mean = np.sum(np.square(column)) / n_year
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            for y, z in enumerate(z_score):
                start = i + y*len_year
                z_per_day[start] = z
    else:
        size = min_length
        end_index = n - size + 1
        e_span = len_year - size + 1
        end_year = end_index // len_year

        for year in range(end_year):
            start = year * len_year
            v[year, 0] = np.sum(ts[start:start+size])
            for j in range(1, len_year):
                v[year, j] = v[year, j-1] + ts[start+j+size-1] - ts[start+j-1]

        start = end_year * len_year
        v[end_year, 0] = np.sum(ts[start:start+size])
        for j in range(1, e_span):
            v[end_year, j] = v[end_year, j-1] + \
                ts[start+j+size-1] - ts[start+j-1]

        n_valid_years = end_year + 1
        for i in range(e_span):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            for y, z in enumerate(z_score):
                start = i + y*len_year
                for day in range(size):
                    abs_val = np.abs(z_per_day[start+day])
                    if abs_val < np.abs(z):
                        z_per_day[start+day] = z
        n_valid_years = end_year
        for i in range(e_span, len_year):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            for y, z in enumerate(z_score):
                start = i + y*len_year
                for day in range(size):
                    abs_val = np.abs(z_per_day[start+day])
                    if abs_val < np.abs(z):
                        z_per_day[start+day] = z

    end_index = n - size + 1
    # Finding burst of length 2 <= l <= 256
    for size in range(min_length, max_length + 1):
        end_index -= 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        # Compute aggreagates, vectorized by year
        for j in range(end_year):
            start = j * len_year + size - 1
            end = start + len_year
            v[j, :len_year] += ts[start:end]
        # Last year special case to handle interval out of bounds
        start = end_year * len_year + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        # Z-score computation
        # First loop for the condition: interval < e_span
        # for those intervals the last valid year is included
        n_valid_years = end_year + 1
        for i in range(e_span):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            for y, z in enumerate(z_score):
                start = i + y*len_year
                for day in range(size):
                    abs_val = np.abs(z_per_day[start+day])
                    if abs_val < np.abs(z):
                        z_per_day[start+day] = z
        # Second loop for the condition: interval >= e_span
        # for those intervals the last valid year is excluded
        n_valid_years = end_year
        for i in range(e_span, len_year):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            for y, z in enumerate(z_score):
                start = i + y*len_year
                for day in range(size):
                    abs_val = np.abs(z_per_day[start+day])
                    if abs_val < np.abs(z):
                        z_per_day[start+day] = z
    return z_per_day
