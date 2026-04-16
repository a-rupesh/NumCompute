import numpy as np
def rank(data, method="average"):
    """
    Rank data with tie handling.

    Methods:
        average: average rank for ties
        dense: no gaps
        ordinal: unique ranking
    """
    data = np.asarray(data)
    temp = data.argsort()
    ranks = np.empty_like(temp, dtype=float)

    if method == "ordinal":
        ranks[temp] = np.arange(1, len(data) + 1)

    elif method == "dense":
        sorted_data = data[temp]
        unique_vals, inverse = np.unique(sorted_data, return_inverse=True)
        ranks[temp] = inverse + 1

    elif method == "average":
        sorted_data = data[temp]
        ranks[temp] = np.arange(1, len(data) + 1)

        # fix ties
        for val in np.unique(sorted_data):
            indices = np.where(data == val)[0]
            avg_rank = np.mean(ranks[indices])
            ranks[indices] = avg_rank
    else:
        raise ValueError("Invalid method")

    return ranks
def percentile(data, q):
    """
    Compute percentile (0–100)
    """
    data = np.asarray(data)
    return np.percentile(data, q)
