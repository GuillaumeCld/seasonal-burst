import numpy as np
import pytest
import seasonal_burst.generic_algorithms as ga
import seasonal_burst.skipping_length as sl


def test_exhaustive_search_matches_generic():
    """Test that exhaustive_search gives same results as generic implementation."""

    # Test parameters
    n_period = 10
    period_length = 365
    min_length = 5
    max_length = 30
    threshold = 3.0

    skip_factors = [1, 2, 5, 10, 20]
    # Generate random time series
    rng = np.random.default_rng(42)
    n_tests = 5

    for skip_factor in skip_factors:
        for _ in range(n_tests):
            # Generate random time series with some bursts
            ts = rng.normal(0, 1, n_period * period_length)

            # Add some random bursts
            n_bursts = rng.integers(1, 5)
            for _ in range(n_bursts):
                start = rng.integers(0, len(ts) - max_length)
                length = rng.integers(min_length, max_length)
                ts[start : start + length] += rng.uniform(2, 4)

            # Run both implementations
            bursts_es = sl.burst_compute(
                ts,
                n_period,
                threshold,
                period_length,
                min_length,
                max_length,
                skip_factor,
            )
            bursts_ges = ga.skipping_length(
                ts,
                n_period,
                threshold,
                period_length,
                np.add.reduce,
                skip_factor,
                min_length,
                max_length,
            )

            # Sort bursts for comparison
            bursts_es = sorted(bursts_es)
            bursts_ges = sorted(bursts_ges)

            # Check results match
            assert len(bursts_es) == len(bursts_ges)
            for b1, b2 in zip(bursts_es, bursts_ges):
                assert b1[0] == b2[0]  # Start index
                assert b1[1] == b2[1]  # End index
                assert np.isclose(b1[2], b2[2])  # Score


def test_exhaustive_search_matches_generic_mediterranean():
    """Test that exhaustive_search gives same results as generic implementation."""

    # Test parameters
    ts = np.load("Data/SST/mediterranean_average_processed.npy")
    period_length = 365
    n_period = ts.shape[0] // period_length
    min_length = 1
    max_length = 365
    threshold = 3.0

    skip_factors = [1, 2, 5, 10, 20]

    for skip_factor in skip_factors:
        # Run both implementations
        bursts_es = sl.burst_compute(
            ts, n_period, threshold, period_length, min_length, max_length, skip_factor
        )
        bursts_ges = ga.skipping_length(
            ts,
            n_period,
            threshold,
            period_length,
            np.add.reduce,
            skip_factor,
            min_length,
            max_length,
        )

        # Sort bursts for comparison
        bursts_es = sorted(bursts_es)
        bursts_ges = sorted(bursts_ges)

        # Check results match
        assert len(bursts_es) == len(bursts_ges)
        for b1, b2 in zip(bursts_es, bursts_ges):
            assert b1[0] == b2[0]  # Start index
            assert b1[1] == b2[1]  # End index
            assert np.isclose(b1[2], b2[2]), f"{b1[2]:.5f} != {b2[2]:.5f}"
