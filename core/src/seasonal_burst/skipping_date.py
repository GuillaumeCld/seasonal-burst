import numpy as np
from numba import njit, float64, int64
from numba.typed import List



@njit((float64[:], int64, float64, int64, int64, int64, int64))
def burst_compute(ts, n_period, threshold, period_length, min_length, max_length, k):
    n = len(ts)
    v = np.copy(ts).reshape(n_period, period_length)
    all_bursts = List()
    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        inv = 1/n_period
        for i in range(period_length):
            column = v[:n_period, i]
            mean = np.sum(column) * inv
            squared_mean = np.sum(np.square(column)) * inv
            std = np.sqrt(squared_mean - mean*mean)
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
        std_v = np.zeros(period_length, dtype=np.float64)


        for year in range(1, end_year):
            start = year * period_length
            v[year, 0] = np.sum(ts[start:start+size])
            for j in range(1, period_length):
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
            for date in range(period_length):
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                    start = date + year * period_length
                    all_bursts.append((start, start + size - 1, z_score[date]))

        z_score = v[end_year, :e_span] - mean_v[:e_span]
        for date in range(e_span):
            std = std_v[date]
            if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                start = date + end_year * period_length
                all_bursts.append((start, start + size - 1, z_score[date]))
    
    # Finding burst of length min_length <= l <= max_length
    for size in range(min_length+1, max_length+1):
        s = max(1, size // k)
        end_index -= 1
        e_span = end_index % period_length
        end_year = end_index // period_length
        restr = int(np.ceil(e_span/s))

        
        start = size - 1
        end = start + period_length
        v[0,:] += ts[start:end]

        n_pts = int(np.ceil(period_length/s))
        mean_v = np.zeros(n_pts, dtype=np.float64)

        i = 0
        for j in range(0, period_length, s):
            mean_v[i] = v[0, j]
            i += 1
        std_v = np.zeros(n_pts, dtype=np.float64)
        
        # Compute aggreagates, vectorized by year
        for year in range(1, end_year):
            inv = 1/(year+1)
            start = year * period_length + size - 1
            end = start + period_length
            v[year, :] += ts[start:end]
            # welford update
            i = 0
            for j in range(0, period_length, s):
                tmp = v[year, j] - mean_v[i]

                std_v[i] += year*inv *  tmp*tmp  
                mean_v[i] += tmp * inv
                i += 1

        # Last year special case to handle interval out of bounds
        start = end_year * period_length + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        # welford update
        inv = 1/(end_year+1)
        i = 0
        for j in range(0, e_span, s):
            tmp = v[end_year, j] - mean_v[i]
            std_v[i] += end_year*inv * tmp*tmp
            mean_v[i] += tmp * inv
            i += 1
    
        std_v[:restr] *= inv
        std_v[restr:] /= end_year
        std_v = np.sqrt(std_v)


        for year in range(end_year):
            for date in range(n_pts):
                z_score = v[year, date*s] - mean_v[date]
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score/std) >= threshold:
                    start = date*s + year * period_length
                    all_bursts.append((start, start + size - 1, z_score/std))

        for date in range(restr):
            std = std_v[date]
            z_score = v[end_year, date*s] - mean_v[date]
            if std >= 1e-10 and np.abs(z_score/std) >= threshold:
                start = date*s + end_year * period_length
                all_bursts.append((start, start + size - 1, z_score/std))

    return all_bursts