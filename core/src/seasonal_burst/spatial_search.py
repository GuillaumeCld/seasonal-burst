import numpy as np
import xarray as xr
from dask.distributed import Client

# Local imports
from utils.post_process import merge_bursts


def process_single_coordinate(ts: np.array, n_period: int, threshold: float, period_length: int, min_length: int, max_length: int, compute_fct: callable) -> list:
    """
    Compute the merged burst representation for a single coordinate using the given function and parameters.
    Parameters:
        ts (np.array): time series of the coordinate
        n_period (int): number of periods in the time series
        threshold (float): threshold for the bursts
        period_length (int): length of a period
        min_length (int): minimum length of a burst
        max_length (int): maximum length of a burst
        compute_fct (callable): function to compute the bursts
    
    Returns:
        list of the merged burst representation (i_min, i_max, sign) 
    """
    return merge_bursts(
        compute_fct(
            ts,
            n_period,
            threshold,
            period_length,
            min_length,
            max_length
        )
    )


def process_grid(grid: xr.DataArray, valid_coordinates: np.array, threshold: float, n_period: int, period_length: int, min_length: int, max_length: int, compute_fct: callable, n_workers: int) -> np.array:
    """
    Compute the merged burst representation for a grid of coordinates using the given function and parameters.
    Parameters:
        grid (xr.DataArray): grid of time series
        valid_coordinates (np.array): array of valid coordinates
        threshold (float): threshold for the bursts
        n_period (int): number of periods in the time series
        period_length (int): length of a period
        min_length (int): minimum length of a burst
        max_length (int): maximum length of a burst
        compute_fct (callable): function to compute the bursts
        n_workers (int): number of workers
        
    Returns:
        grid_list (np.array): grid of the merged burst representation
    """


    values = grid.values.astype(np.float64)
    threads_per_worker = 1

    with Client(n_workers=n_workers, threads_per_worker=threads_per_worker) as client:
        futures = {}
        # Submit tasks
        for (i, j) in valid_coordinates:
            ts = values[:, i, j]
            future = client.submit(process_single_coordinate, ts, n_period,
                                   threshold, period_length, min_length, max_length, compute_fct)
            futures[(i, j)] = future

        results = client.gather(futures)

    # Populate grid_list from results
    grid_list = np.empty((grid.shape[1], grid.shape[2]), dtype=object)
    for (i, j), bursts in results.items():
        grid_list[i, j] = bursts

    return grid_list
