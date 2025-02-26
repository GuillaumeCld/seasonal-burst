import copy

import numpy as np
from numba import float64, int32, njit


@njit((float64[:], int32, int32, int32))
def SWT_per_year(ts: np.ndarray, n_year: int, len_year: int, n_level=9):
    """
    Calculate the Shifted Wavelet Tree (SWT) for each year of a time series.

    Parameters:
    - ts The time series data.
    - n_year The number of years in the time series.
    - len_year The length of each year in the time series.

    Returns:
    list: A list of SWTs for each year, where each SWT is a list of levels.
    """
    
    levels_matrices = []

    brick_sizes = np.empty(n_level, dtype=np.int32)
    ns_bricks = np.empty(n_level, dtype=np.int32)
    for i in range(n_level):
        size = 2 << i
        brick_sizes[i] = size
        n_bricks_wavelet = len_year // size
        n_bricks = n_bricks_wavelet * 2

        #if size // 2 + n_bricks_wavelet * size < len_year:
        #    n_bricks += 2
        #else:
        #    n_bricks += 1
        ns_bricks[i] = n_bricks

    # For each level of the SWT
    for level in range(n_level):
        brick_size = brick_sizes[level]
        n_bricks = ns_bricks[level]
        level_matrice = np.empty(
            (n_year, n_bricks), dtype=np.float64)

        for year in range(n_year):
            i_year = year * len_year

            if level == 0:
                m = 1
                previous_aggregate = ts[i_year: i_year + len_year]
            else:
                m = 2
                previous_aggregate = levels_matrices[level - 1][year, :]

            for j in range(n_bricks - 2):
                level_matrice[year, j] = (
                    previous_aggregate[j * m] + previous_aggregate[(j + 1) * m]
                )

            if year != n_year - 1 or brick_size == 2:
                level_matrice[year, n_bricks - 2] = (
                    previous_aggregate[(n_bricks - 2) * m]
                    + previous_aggregate[(n_bricks - 1) * m]
                )
            else:
                level_matrice[year, n_bricks - 2] = np.nan
            if year != n_year - 1 and not (year == n_year - 2 and brick_size == 512):
                start = (
                    i_year + (n_bricks - 1) * brick_size // 2 + brick_size // 2
                )
                level_matrice[year, n_bricks - 1] = previous_aggregate[
                    (n_bricks - 1) * m
                ] + np.sum(ts[start: start + brick_size // 2])
            else:
                level_matrice[year, n_bricks - 1] = np.nan

        levels_matrices.append(level_matrice)
    return levels_matrices



@njit()
def columns_z_scores(column):
    mean = np.nanmean(column)
    std = np.nanstd(column)
    z_scores = (column - mean) / std

    return z_scores




def z_score_SWT(
    levels_matrices: list[np.ndarray], n_year: int, len_year: int, threshold: float, ts: np.ndarray, n_level = 9
):
    """
    Compute the z-score of the yearly shifted wavelet transform (SWT) data and return the flagged columns and their flagged bricks (years).

    Parameters:
    - yearly_SWT (Array): The yearly SWT data.
    - n_year (int): The number of years.
    - len_year (int): The length of each year's data.
    - threshold (float): The threshold value for flagging bricks.

    Returns:
    - flagged_columns (List): A list of flagged columns, where each item is a tuple containing the level, brick, and flagged years.
    """

    
    swt = copy.deepcopy(levels_matrices)
    for level in range(n_level):
        level_matrix = levels_matrices[level]
        n_brick = level_matrix.shape[1]

        for brick in range(n_brick):

            column = level_matrix[:, brick]

            column_z_scores = columns_z_scores(column)

            swt[level][:, brick] = column_z_scores

    return swt
