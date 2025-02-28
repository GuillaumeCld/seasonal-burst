import numpy as np
from numba import boolean, float32, float64, int32, int64, njit
from numba.typed import List


@njit((float64[:], int64, float64, int64, int64, int64))
def exhaustive_search(ts: np.array, n_period: int, threshold: float, period_length: int, min_length: int, max_length: int):
    """
    Perform an exhaustive search for bursts in a time series.
    This function identifies seasonal bursts in a time series `ts` by calculating z-scores for different periods and lengths.
    A burst is defined as a period where the z-score exceeds a given threshold.
    
    Parameters:
        ts (np.array): The input time series data.
        n_period (int): The number of periods in the time series.
        threshold (float): The z-score threshold to identify bursts.
        period_length (int): The length of each period.
        min_length (int): The minimum length of a burst.
        max_length (int): The maximum length of a burst.
   
    Returns:
        List: A list of tuples where each tuple contains the start index, end index, and z-score of a burst.
    """



    n = len(ts)
    v = np.copy(ts).reshape(n_period, period_length)
    all_bursts = List()
    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        inv = 1/n_period
        for i in range(period_length):
            column = v[:n_period, i]
            # eval += len(column)
            mean = np.sum(column) * inv
            squared_dev = np.sum(np.square(column-mean)) * inv
            std = np.sqrt(squared_dev)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = i + y*period_length
                all_bursts.append((start, start, z_score[y]))
        end_index = n
    else:
        size = min_length
        end_index = n - size + 1
        e_span = period_length  - size + 1
        end_year = end_index // period_length

        v[0, 0] = np.sum(ts[:size])
        for j in range(1, period_length):
            v[0, j] = v[0, j-1] + ts[j+size-1] - ts[j-1]

        mean_v = np.copy(v[0, :]) 
        std_v = np.zeros(period_length, dtype=ts.dtype)


        for year in range(1, end_year):
            start = year * period_length
            v[year, 0] = v[year-1, period_length-1] + \
                ts[start+size-1] - ts[start-1]
            for j in range(1, period_length):
                v[year, j] = v[year, j-1] + ts[start+j+size-1] - ts[start+j-1]

            # welford update
            inv = 1/(year+1)
            tmp = v[year, :] - mean_v
            std_v += year*inv *  tmp*tmp            
            mean_v += tmp * inv
        # Last year special case to handle interval out of bounds
        start = end_year * period_length
        v[end_year, 0] = v[end_year-1, period_length-1] + \
            ts[start+size-1] - ts[start-1]
        for j in range(1, e_span):
            v[end_year, j] = v[end_year, j-1] + \
                ts[start+j+size-1] - ts[start+j-1]
        # welford update
        inv = 1/(end_year+1)
        tmp = v[end_year, :e_span] - mean_v[:e_span]
        std_v[:e_span] += end_year*inv * tmp*tmp
        mean_v[:e_span] += tmp * inv  


        std_v[:e_span] = std_v[:e_span]*inv
        std_v[e_span:] = std_v[e_span:] / end_year
        std_v = np.sqrt(std_v)  

        for year in range(end_year):
            z_score = v[year, :] - mean_v
            for date in range(period_length):
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                    start = date + year * period_length
                    all_bursts.append((start, start + size - 1, z_score[date]/std))

        z_score = v[end_year, :e_span] - mean_v[:e_span]
        for date in range(e_span):
            std = std_v[date]
            if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                start = date + end_year * period_length
                all_bursts.append((start, start + size - 1, z_score[date]/std))


    mean_v = np.zeros(period_length, dtype=ts.dtype)
    # Finding burst of length min_length+1 <= l <= max_length
    for size in range(min_length+1, max_length+1):
        end_index -= 1
        e_span = end_index % period_length
        end_year = end_index // period_length
        start = size - 1
        end = start + period_length
        v[0,:] += ts[start:end]
        mean_v = np.copy(v[0, :]) 
        std_v = np.zeros(period_length, dtype=ts.dtype)
        # Compute aggreagates, vectorized by year
        for year in range(1, end_year):
            inv = 1/(year+1)
            start = year * period_length + size - 1
            end = start + period_length
            v[year, :] += ts[start:end]
            # welford update
            tmp = v[year, :] - mean_v
            std_v += year*inv *  tmp*tmp            
            mean_v += tmp * inv
    
        # Last year special case to handle interval out of bounds
        start = end_year * period_length + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        # welford update
        inv = 1/(end_year+1)
        tmp = v[end_year, :e_span] - mean_v[:e_span]
        std_v[:e_span] += end_year*inv * tmp*tmp
        mean_v[:e_span] += tmp * inv  

        std_v[:e_span] = std_v[:e_span]*inv
        std_v[e_span:] = std_v[e_span:] / end_year
        std_v = np.sqrt(std_v)  

        for year in range(end_year):
            z_score = v[year, :] - mean_v
            for date in range(period_length):
                std = std_v[date]

                if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                    start = date + year * period_length
                    all_bursts.append((start, start + size - 1, z_score[date]/std))

        z_score = v[end_year, :e_span] - mean_v[:e_span]
        for date in range(e_span):
            std = std_v[date]
            if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                start = date + end_year * period_length
                all_bursts.append((start, start + size - 1, z_score[date]/std))

    return all_bursts


@njit((float64[:], int64, float64, int64, int64, int64))
def exhaustive_search_intra_period(ts: np.array, n_period: int, threshold: int, period_length: int, min_length: int, max_length: int):
    """
    Perform an exhaustive search for bursts in a time series that are strictly contained within a period (i.e., no overlap between periods).
    This function identifies seasonal bursts in a time series `ts` by calculating z-scores for different periods and lengths.
    A burst is defined as a period where the z-score exceeds a given threshold.
    
    Parameters:
        ts (np.array): The input time series data.
        n_period (int): The number of periods in the time series.
        threshold (float): The z-score threshold to identify bursts.
        period_length (int): The length of each period.
        min_length (int): The minimum length of a burst.
        max_length (int): The maximum length of a burst.
   
     Returns:
    List[Tuple[int, int, float]]: A list of tuples where each tuple contains the start index, end index, and z-score of a burst.
    """

    n = len(ts)
    v = np.copy(ts).reshape(n_period, period_length)
    all_bursts = List()
    inv = 1/n_period
    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        for i in range(period_length):
            column = v[:n_period, i]
            # eval += len(column)
            mean = np.sum(column) * inv
            squared_dev = np.sum(np.square(column-mean)) * inv
            std = np.sqrt(squared_dev)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = i + y*period_length
                all_bursts.append((start, start, z_score[y]))
        end_index = n
    else:
        size = min_length
        end_index = n - size + 1
        e_span = period_length  - size + 1
        end_year = end_index // period_length

        v[0, 0] = np.sum(ts[:size])
        for j in range(1, period_length):
            v[0, j] = v[0, j-1] + ts[j+size-1] - ts[j-1]

        mean_v = np.copy(v[0, :]) 
        std_v = np.zeros(period_length, dtype=ts.dtype)

        for year in range(1, end_year):
            start = year * period_length
            v[year, 0] = np.sum(ts[start:start+size])
            for j in range(1, period_length- size + 1):
                v[year, j] = v[year, j-1] + ts[start+j+size-1] - ts[start+j-1]

            # welford update
            inv = 1/(year+1)
            tmp = v[year, :] - mean_v
            std_v += year*inv *  tmp*tmp            
            mean_v += tmp * inv
        # Last year special case to handle interval out of bounds
        start = end_year * period_length
        v[end_year, 0] = np.sum(ts[start:start+size])
        for j in range(1, e_span):
            v[end_year, j] = v[end_year, j-1] + \
                ts[start+j+size-1] - ts[start+j-1]
        # welford update
        inv = 1/(end_year+1)
        tmp = v[end_year, :e_span] - mean_v[:e_span]
        std_v[:e_span] += end_year*inv * tmp*tmp
        mean_v[:e_span] += tmp * inv  


        std_v[:e_span] = std_v[:e_span]*inv
        std_v[e_span:] = std_v[e_span:] / end_year
        std_v = np.sqrt(std_v)  

        for year in range(end_year):
            z_score = v[year, :] - mean_v
            for date in range(period_length- size + 1):
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                    start = date + year * period_length
                    all_bursts.append((start, start + size - 1, z_score[date]/std))

        z_score = v[end_year, :e_span] - mean_v[:e_span]
        for date in range(e_span):
            std = std_v[date]
            if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                start = date + end_year * period_length
                all_bursts.append((start, start + size - 1, z_score[date]/std))


    mean_v = np.zeros(period_length, dtype=ts.dtype)
    # Finding burst of length 2 <= l <= max_length
    for size in range(min_length+1, max_length+1):
        end_index -= 1
        e_span = end_index % period_length
        end_year = end_index // period_length
        start = size - 1
        end = start + period_length
        v[0,:] += ts[start:end]
        mean_v = np.copy(v[0, :]) 
        std_v = np.zeros(period_length, dtype=ts.dtype)
        # Compute aggreagates, vectorized by year
        for year in range(1, end_year):
            inv = 1/(year+1)
            start = year * period_length + size - 1
            end = start + period_length - size + 1
            v[year, :] += ts[start:end]
            # welford update
            tmp = v[year, :] - mean_v
            std_v += year*inv *  tmp*tmp            
            mean_v += tmp * inv
    
        # Last year special case to handle interval out of bounds
        start = end_year * period_length + size - 1
        end = start + e_span 
        v[end_year, :e_span] += ts[start:end]
        # welford update
        inv = 1/(end_year+1)
        tmp = v[end_year, :e_span] - mean_v[:e_span]
        std_v[:e_span] += end_year*inv * tmp*tmp
        mean_v[:e_span] += tmp * inv  

        std_v[:e_span] = std_v[:e_span]*inv
        std_v[e_span:] = std_v[e_span:] / end_year
        std_v = np.sqrt(std_v)  

        for year in range(end_year):
            z_score = v[year, :] - mean_v
            for date in range(period_length - size + 1):
                # eval += 1
                std = std_v[date]

                if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                    start = date + year * period_length
                    all_bursts.append((start, start + size - 1, z_score[date]/std))

        z_score = v[end_year, :e_span] - mean_v[:e_span]
        for date in range(e_span):
            std = std_v[date]
            if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                start = date + end_year * period_length
                all_bursts.append((start, start + size - 1, z_score[date]/std))

    return all_bursts