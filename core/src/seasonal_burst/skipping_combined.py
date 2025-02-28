import numpy as np
from numba import njit, float64, int64
from numba.typed import List


@njit((float64[:], int64, float64, int64, int64, int64, int64))
def burst_compute(ts, n_year, threshold, len_year, min_length, max_length, k):
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = List()
    mean_v = np.zeros(len_year, dtype=np.float64)

    # Finding burst of length 1 (day) only the z-score are needed
    inv = 1/n_year
    if min_length == 1:
        for i in range(len_year):
            column = v[:n_year, i]
            mean = np.sum(column) * inv
            squared_mean = np.sum(np.square(column)) * inv
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = i + y*len_year
                all_bursts.append((start, start, z_score[y]))
        end_index = n
    else:
        size = min_length
        end_index = n - size + 1
        e_span = len_year  - size + 1
        end_year = end_index // len_year

        v[0, 0] = np.sum(ts[:size])
        for j in range(1, len_year):
            v[0, j] = v[0, j-1] + ts[j+size-1] - ts[j-1]

        mean_v = np.copy(v[0, :]) 
        std_v = np.zeros(len_year, dtype=np.float64)


        for year in range(1, end_year):
            start = year * len_year
            v[year, 0] = v[year-1, len_year-1] + \
                ts[start+size-1] - ts[start-1]
            for j in range(1, len_year):
                v[year, j] = v[year, j-1] + ts[start+j+size-1] - ts[start+j-1]

            # welford update
            inv = 1/(year+1)
            tmp = v[year, :] - mean_v
            std_v += year*inv *  tmp*tmp            
            mean_v += tmp * inv
        # Last year special case to handle interval out of bounds
        start = end_year * len_year
        v[end_year, 0] = v[end_year-1, len_year-1] + \
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
            for date in range(len_year):
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                    start = date + year * len_year
                    all_bursts.append((start, start + size - 1, z_score[date]))

        z_score = v[end_year, :e_span] - mean_v[:e_span]
        for date in range(e_span):
            std = std_v[date]
            if std >= 1e-10 and np.abs(z_score[date]/std) >= threshold:
                start = date + end_year * len_year
                all_bursts.append((start, start + size - 1, z_score[date]))

    mean_v = np.zeros(len_year, dtype=np.float64)
    # Finding burst of length 2 <= l <= max_length
    size = min_length + max(1, min_length // k)

    while size <= max_length:
        s = max(1, size // k)
        end_index = n - size + 1
        e_span = len_year  - size + 1
        end_year = end_index // len_year
        restr = int(np.ceil(e_span/s))


        v[0, 0] = np.sum(ts[:size])
        for j in range(1, len_year):
            v[0, j] = v[0, j-1] + ts[j+size-1] - ts[j-1]

        n_pts = int(np.ceil(len_year/s))
        mean_v = np.zeros(n_pts, dtype=np.float64)
        for j in range(n_pts):
            mean_v[j] = v[0, j*s] 

        std_v = np.zeros(n_pts, dtype=np.float64)

        for year in range(1, end_year):
            inv = 1/(year+1)

            start = year * len_year
            v[year, 0] = v[year-1, len_year-1] + \
                ts[start+size-1] - ts[start-1]
            

            tmp = v[year, 0] - mean_v[0]
            std_v[0] += year*inv *  tmp*tmp  
            mean_v[0] += tmp * inv

            for j in range(1, len_year):
                v[year, j] = v[year, j-1] + ts[start+j+size-1] - ts[start+j-1]
            
            i = 0
            for j in range(1, len_year, s):
                tmp = v[year, j] - mean_v[i]

                std_v[i] += year*inv *  tmp*tmp  
                mean_v[i] += tmp * inv
                i += 1

            # welford update
            # tmp = v[year, ::s] - mean_v
            # std_v += year*inv *  tmp*tmp            
            # mean_v += tmp * inv

        # Last year special case to handle interval out of bounds
        # max_sequence = len_year - size + 1
        start = end_year * len_year
        v[end_year, 0] = v[end_year-1, len_year-1] + \
            ts[start+size-1] - ts[start-1]
        for j in range(1, e_span):
            v[end_year, j] = v[end_year, j-1] + \
                ts[start+j+size-1] - ts[start+j-1]
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
            # z_score = v[year, ::s] - mean_v
            for date in range(n_pts):
                z_score = v[year, date*s]  - mean_v[date]
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score/std) >= threshold:
                    start = date*s + year * len_year
                    all_bursts.append((start, start + size - 1, z_score/std))

        # z_score = v[end_year, 0:e_span:s] - mean_v[:restr]
        for date in range(restr):
            std = std_v[date]
            z_score = v[end_year, date*s] - mean_v[date]
            if std >= 1e-10 and np.abs(z_score/std) >= threshold:
                start = date*s + end_year * len_year
                all_bursts.append((start, start + size - 1, z_score/std))

        size += max(1, size // k)

    return all_bursts