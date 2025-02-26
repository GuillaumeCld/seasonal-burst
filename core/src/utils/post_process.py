def merge_bursts(sequences):
    """
    Compute the Merged bursts representation i.e. the union of the given sequences with z_score of the same sign.

    Parameters:
        sequences (List): list of lists of integers (i_min, i_max, z_score)

    Returns:
        list (List): list of tuples of integers (i_min, i_max, sign(z_score))
    """

    # Separate the sequences with positive and negative z_score
    positive_intervals = [
        (interval[0], interval[1]) for interval in sequences if interval[2] > 0
    ]
    negative_intervals = [
        (interval[0], interval[1]) for interval in sequences if interval[2] < 0
    ]
    # Merge the sequences with positive and negative z_score
    positive_merged_intervals = signed_merged(positive_intervals, 1)
    negative_merged_intervals = signed_merged(negative_intervals, -1)
    # Concatenate the positive and negative sequences
    merged_intervals = positive_merged_intervals + negative_merged_intervals

    return merged_intervals

def signed_merged(intervals, sign):
    """
    Compute the union of the given intervals.

    Parameters:
        intervals (List): list of tuples of integers (i_min, i_max)
        sign (int): sign of the z_score (1 or -1)

    Returns:
        list (list): list of tuples of integers (i_min, i_max, sign)
    """
    if not intervals:
        return []
    # Sort intervals by the starting index
    intervals.sort()
    n = len(intervals)
    # Create the union of the intervals
    merged_intervals = []
    i = 0
    while i < n:
        start, end = intervals[i]

        j = i + 1
        while j < n and intervals[j][0] <= end + 1:
            end = max(end, intervals[j][1])
            j += 1

        merged_intervals.append((start, end, sign))
        i = j

    return merged_intervals
