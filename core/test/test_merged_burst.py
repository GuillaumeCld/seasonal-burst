import random

import pytest
from utils.post_process import merge_bursts


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

    # Merge bursts
    merged = merge_bursts(bursts)

    postive_bursts = [b for b in merged if b[2] > 0]
    negative_bursts = [b for b in merged if b[2] < 0]

    # Verify the merged bursts are correct
    for burst in postive_bursts:
        for i in range(burst[0], burst[1] + 1):
            assert any(start <= i <= end for start, end, _ in merged)
    for burst in negative_bursts:
        for i in range(burst[0], burst[1] + 1):
            assert any(start <= i <= end for start, end, _ in merged)

    # Verify the merged result have no overlapping bursts
    for i, burst1 in enumerate(postive_bursts):
        for burst2 in postive_bursts[i + 1 :]:
            assert (
                burst1[1] < burst2[0] or burst2[1] < burst1[0]
            ), f"Overlap between {burst1} and {burst2}"

    for i, burst1 in enumerate(negative_bursts):
        for burst2 in negative_bursts[i + 1 :]:
            assert (
                burst1[1] < burst2[0] or burst2[1] < burst1[0]
            ), f"Overlap between {burst1} and {burst2}"

    # Verify the merged result matches expected output for known input
    input_bursts = [(1, 5, 1), (2, 6, 1), (8, 10, 1), (9, 12, 1)]
    expected_output = [(1, 6, 1), (8, 12, 1)]
    output = merge_bursts(input_bursts)
    assert output == expected_output, f"Expected {expected_output} for {input_bursts}"
