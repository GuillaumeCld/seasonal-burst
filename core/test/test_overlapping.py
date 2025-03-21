import random

import numpy as np
from utils.operators import overlapping


def generate_random_bursts(count, time_range):
    """Generate a random list of bursts as (start, end) tuples."""
    bursts = []
    for _ in range(count):
        start = random.randint(0, time_range - 1)
        end = random.randint(start, time_range)
        score = random.randint(-1, 1)
        bursts.append((start, end, score))
    return bursts


def test_merge_bursts():
    # Generate random bursts
    random.seed(42)  # For reproducibility
    bursts = generate_random_bursts(1000, 100)

    for burst1 in bursts:
        for burst2 in bursts:
            are_overlapping = overlapping(burst1, burst2)

            sign1 = np.sign(burst1[2])
            sign2 = np.sign(burst2[2])
            if sign1 == sign2 and (
                burst1[0] <= burst2[0] <= burst1[1]
                or burst2[0] <= burst1[0] <= burst2[1]
            ):
                assert (
                    are_overlapping
                ), f"Bursts should overlap but don't: {burst1}, {burst2}"
            else:
                assert (
                    not are_overlapping
                ), f"Bursts should not overlap but do: {burst1}, {burst2}"
