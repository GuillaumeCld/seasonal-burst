import numba
import numpy as np
from numba import boolean, float64, int64
from numba.experimental import jitclass

spec = [
    ('ts', float64[:]),
    ('n_year', int64),
    ('len_year', int64),
    ('threshold', float64),
    ('min_length', int64),
    ('max_length', int64),
    ('n_length', int64),
    ('counts', float64[:,::1]),
    ('current_index', int64),
    ('mean_matrix', float64[:,::1]),
    ('variance_matrix', float64[:,::1]),
    ('current_buffer', float64[::1]),
    ('moving', boolean),
    ('left_buffer', float64[::1]),
    ('all_bursts', numba.typeof([(0,0,0.0)])),
]

@jitclass(spec)
class WelfordBurstDetector:
    def __init__(self, ts, n_year, len_year, threshold, min_length, max_length, moving):
        self.ts = ts
        self.n_year = n_year
        self.len_year = len_year
        self.threshold = threshold
        self.min_length = min_length
        self.max_length = max_length
        self.n_length = max_length - min_length + 1
        self.counts = np.zeros((self.n_length, self.len_year))
        self.current_index = len(ts) - 1

        # Initialize rolling statistics matrices for Welford's mean and variance
        self.mean_matrix = np.zeros((self.n_length, self.len_year))
        self.variance_matrix = np.zeros((self.n_length, self.len_year))
        self._build_matrices()
        self.current_buffer = np.zeros(self.n_length)
        for l in range(self.min_length, self.max_length + 1):
            self.current_buffer[l-self.min_length] = np.sum(ts[-l:])

        if moving:
            self.left_buffer = np.zeros(self.n_length)
            for i, l in enumerate(range(self.min_length, self.max_length + 1)):
                self.left_buffer[i] = np.sum(ts[:l])

        self.moving = moving    
        self.all_bursts = [(0, 0, 0.0)]

    def compute_explicte_m_s(self, index, length):
        agg = []
        for year in range(len(self.ts) // self.len_year + 1):
            end = year * self.len_year + index
            start = end - length +1
            if start < 0 or end >= len(self.ts):
                continue

            agg.append(np.sum(self.ts[start:end+1]))

        mean = np.mean(agg)
        variance = np.var(agg, ddof=1)


        return mean, variance

    def _build_matrices(self):
        """Build the rolling matrices using the initial buffer."""


        if self.min_length == 1:
            l = 1
            v = np.copy(self.ts)
            for i in range(len(v)):
                id = i % self.len_year
                self.mean_matrix[0, id] += v[i]
                self.counts[0, id] += 1
            self.mean_matrix[0,:] /= self.counts[0,:]

            for i in range(len(v)):
                id = i % self.len_year
                self.variance_matrix[0, id] += (v[i] - self.mean_matrix[0, id]) ** 2
        else:
            v = np.zeros(len(self.ts))
            i = self.min_length - 1
            id = i % self.len_year
            v[i] = np.sum(self.ts[:self.min_length])

            self.mean_matrix[0, id] += v[i]
            self.counts[0, id] += 1

            for i in range(self.min_length, len(v)):
                id = i % self.len_year

                v[i] = v[i-1] + self.ts[i] - self.ts[i-self.min_length]


                self.mean_matrix[0, id] += v[i]
                self.counts[0, id] += 1

            self.mean_matrix[0,:] /= self.counts[0,:]
            
            for i in range(self.min_length-1, len(v)):
                id = i % self.len_year
                self.variance_matrix[0, id] += (v[i] - self.mean_matrix[0, id]) ** 2


        for it, l in enumerate(range(self.min_length+1, self.max_length + 1)):
            l_id = it + 1
            v[l-1:] += self.ts[:len(v)-l+1]

            for i in range(l-1, len(v)):
                self.mean_matrix[l_id, i % self.len_year] += v[i]
                self.counts[l_id, i % self.len_year] += 1
            self.mean_matrix[l_id,:] /= self.counts[l_id,:]
            
            for i in range(l-1, len(v)):
                self.variance_matrix[l_id, i % self.len_year] += (v[i] - self.mean_matrix[l_id, i % self.len_year]) ** 2

    def _update_rolling_statistics(self, index):
        """
        Update mean and variance for a given length and index using Welford's method.
        """
        new_values = self.current_buffer
        # Updating counts
        count = self.counts[:, index] + 1
        old_mean = self.mean_matrix[:, index]
        new_mean = old_mean + (new_values - old_mean) / count

        # Update variance incrementally
        delta = new_values - old_mean
        old_variance = self.variance_matrix[:, index]
        new_variance = old_variance + (count-1)/count * delta*delta

        # Set updated mean and variance
        self.mean_matrix[:, index] = new_mean
        self.variance_matrix[:, index] = new_variance
        self.counts[:, index] = count


    def _update_rolling_statistics_moving_window(self, index):
        """
        Update mean and variance for a given length and index using Welford's method.
        """
        for l in range(self.n_length):
            old_value = self.left_buffer[l]
            l_idx = l
            _index = (index + self.min_length + l_idx - 1) % self.len_year
            
            # Updating counts
            count = self.counts[l_idx, _index]
            old_mean = self.mean_matrix[l_idx, _index]
            new_mean = old_mean - (old_value - old_mean) / (count - 1)
            
            # Update variance incrementally
            delta = old_value - new_mean
            old_variance = self.variance_matrix[l_idx, _index]
            new_variance = old_variance - (count - 1) / count * delta * delta
            
            # Set updated mean and variance
            self.mean_matrix[l_idx, _index] = new_mean
            self.variance_matrix[l_idx, _index] = new_variance
            self.counts[l_idx, _index] = count - 1
   

    def _update_buffer(self, new_value):
        """Update the current buffer."""
        self.current_buffer[1:] = self.current_buffer[:-1] + new_value
        self.current_buffer[0] = np.sum(self.ts[-self.min_length:])

    def _update_left_buffer(self, old):
        """Update the left buffer."""

        self.left_buffer -= old
        self.left_buffer += self.ts[self.min_length-1:self.max_length]

    def _update_matrices(self, old):
        """Update the rolling matrices using the buffer."""
        index = self.current_index % self.len_year
        self._update_rolling_statistics(index)

        if self.moving:
            self._update_rolling_statistics_moving_window(index)


    def _get_z_score(self):
        """Compute the z-score based on current Welford statistics."""
        index = self.current_index % self.len_year
        mean = self.mean_matrix[:, index]
        variance = self.variance_matrix[:, index] / (self.counts[:, index]-1)
        z_score = (self.current_buffer - mean) / np.sqrt(variance)

        return z_score

    
    def _process_expandable_window(self, observation, t):
        """Process a new observation at time t."""
        self.ts = np.append(self.ts, observation)
        self.current_index = t
        self._update_buffer(observation)
        self._update_matrices(None)

        z_score = self._get_z_score()
        abnormal_lengths = np.where(np.abs(z_score) > self.threshold)[0]

        if len(abnormal_lengths) > 0:
            length = abnormal_lengths[-1]
            start = t - (length + self.min_length) + 1
            self.all_bursts.append((start, t, z_score[length]))

    def _process_moving_window(self, observation, t):
        """Process a new observation at time t for moving window."""

        old = self.ts[0]
        self.ts[:-1] = self.ts[1:]
        self.ts[-1] = observation
        self.current_index = t
        self._update_buffer(observation)
        self._update_matrices(old)
        self._update_left_buffer(old)

        z_score = self._get_z_score()
        abnormal_lengths = np.where(np.abs(z_score) > self.threshold)[0]

        if len(abnormal_lengths) > 0:
            length = abnormal_lengths[-1]
            start = t - (length + self.min_length) + 1
            self.all_bursts.append((start, t, z_score[length]))

    def get_all_bursts(self):
        """Return all detected bursts."""
        return self.all_bursts
        # return []

    def process(self, obs, i):
        if self.moving:
            self._process_moving_window(obs, i)
        else:
            self._process_expandable_window(obs, i)

