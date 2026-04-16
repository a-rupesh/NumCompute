import numpy as np
def sort_array(arr, axis=-1):
    """
    Sort array along given axis.
    """
    return np.sort(arr, axis=axis)
def top_k(arr, k):
    """
    Return top k elements (unsorted).
    """
    arr = np.asarray(arr)
    idx = np.argpartition(arr, -k)[-k:]
    return arr[idx]
def binary_search(arr, target):
    """
    Binary search on sorted array.
    Returns index or -1 if not found.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def quickselect(arr, k):
    """
    Return k-th smallest element (0-based index).
    """
    arr = np.asarray(arr)

    if k < 0 or k >= len(arr):
        raise ValueError("k out of bounds")

    return np.partition(arr, k)[k]
