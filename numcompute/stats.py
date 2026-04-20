import numpy as np

def mean(arr, axis = None):
    """
    Calculates mean of an array over a given axis. If the axis is not
    specificed the mean of all elements in the array is calculated.
    If the array contains NaN elements, the mean is calculated ignoring them.

    Args:
        arr (np.ndarray): Array to calculate mean of
        axis (int or None): Axis to calculate the mean over. If None,
            the maximum of all elements in the array is calculated. Negative
            axis values start indexing from the end of the array. Must have 
            -arr.ndim <= axis < arr.ndim.

    Returns:
        np.ndarray or np.float: Mean of the array over the given axis. If the
        axis is not specified or the array is one dimensional, an np.float is 
        returned. If the axis is specified, then the output is an np.ndarray
        of one less dimension than the input array, with the same shape as the
        input array except for the specified axis.
    """

    # Exception handling
    if not isinstance(arr, np.ndarray):
        raise TypeError("Array must be a NumPy array.")
    if (axis != None) and (not isinstance(axis, int)):
        raise TypeError("Axis must be an integer or None.")
    if (axis != None) and (axis < -arr.ndim or axis >= arr.ndim):
        raise ValueError("Axis must be between .")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were ignored to calculate array mean.")
        return np.nanmean(arr, axis)

    return np.mean(arr, axis)

def median(arr, axis = None):
    """
    Calculates the median of an array over a given axis. If the axis is not
    specificed, the median of all elements in the array is calculated.
    If the array contains NaN elements, the median is calculated ignoring them.

    Args:
        arr (np.ndarray): Array to calculate median of
        axis (int or None): Axis to calculate the median over. If None,
            the maximum of all elements in the array is calculated. Negative
            axis values start indexing from the end of the array. Must have 
            -arr.ndim <= axis < arr.ndim.

    Returns:
        np.ndarray or np.float: Median of the array over the given axis. If the
        axis is not specified or the array is one dimensional, an np.float is 
        returned. If the axis is specified, then the output is an np.ndarray
        of one less dimension than the input array, with the same shape as the
        input array except for the specified axis.
    """

    # Exception handling
    if not isinstance(arr, np.ndarray):
        raise TypeError("Array must be a NumPy array.")
    if (axis != None) and (not isinstance(axis, int)):
        raise TypeError("Axis must be an integer or None.")
    if (axis != None) and (axis < -arr.ndim or axis >= arr.ndim):
        raise ValueError("Axis must be between .")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were ignored to calculate array median.")
        return np.nanmedian(arr, axis)

    return np.median(arr, axis)

def std(arr, axis = None):
    """
    Calculates the standard deviation of an array over a given axis.
    If the axis is not specificed, the standard deviation of all elements
    in the array is calculated. If the array contains NaN elements,
    the standard deviation is calculated ignoring them.

    Args:
        arr (np.ndarray): Array to calculate standard deviation of
        axis (int or None): Axis to calculate the standard deviation over. 
        If None, the standard deviation of all elements in the array is 
        calculated. Negative axis values start indexing from the end of the 
        array. Must have -arr.ndim <= axis < arr.ndim.

    Returns:
        np.ndarray or np.float: Standard deviation of the array over the given 
        axis. If the axis is not specified or the array is one dimensional, 
        an np.float is returned. If the axis is specified, then the output is 
        an np.ndarray of one less dimension than the input array, with the same 
        shape as the input array except for the specified axis.
    """

    # Exception handling
    if not isinstance(arr, np.ndarray):
        raise TypeError("Array must be a NumPy array.")
    if (axis != None) and (not isinstance(axis, int)):
        raise TypeError("Axis must be an integer or None.")
    if (axis != None) and (axis < -arr.ndim or axis >= arr.ndim):
        raise ValueError("Axis must be between .")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were ignored to calculate array standard deviation.")
        return np.nanstd(arr, axis)

    return np.std(arr, axis)

def min(arr, axis = None):
    """
    Calculates the minimum of an array over a given axis. If the axis is not
    specificed, the minimum of all elements in the array is calculated.
    If the array contains NaN elements, the minimum is calculated ignoring them.

    Args:
        arr (np.ndarray): Array to calculate the minimum of
        axis (int or None): Axis to calculate the minimum over. 
        If None, the standard deviation of all elements in the array is 
        calculated. Negative axis values start indexing from the end of the 
        array. Must have -arr.ndim <= axis < arr.ndim.

    Returns:
        np.ndarray or np.float: Minimum of the array over the given axis. If the
        axis is not specified or the array is one dimensional, an np.float is 
        returned. If the axis is specified, then the output is an np.ndarray
        of one less dimension than the input array, with the same shape as the
        input array except for the specified axis.
    """

    # Exception handling
    if not isinstance(arr, np.ndarray):
        raise TypeError("Array must be a NumPy array.")
    if (axis != None) and (not isinstance(axis, int)):
        raise TypeError("Axis must be an integer or None.")
    if (axis != None) and (axis < -arr.ndim or axis >= arr.ndim):
        raise ValueError("Axis must be between .")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were ignored to calculate array minimum.")
        return np.nanmin(arr, axis)

    return np.min(arr, axis)

def max(arr, axis = None):
    """
    Calculates the maximum of an array over a given axis. If the axis is not
    specificed, the maximum of all elements in the array is calculated.
    If the array contains NaN elements, the maximum is calculated ignoring them.

    Args:
        arr (np.ndarray): Array to calculate the maximum of
        axis (int or None): Axis to calculate the maximum over. 
        If None, the standard deviation of all elements in the array is 
        calculated. Negative axis values start indexing from the end of the 
        array. Must have -arr.ndim <= axis < arr.ndim.

    Returns:
        np.ndarray or np.float: Maximum of the array over the given axis. If the
        axis is not specified or the array is one dimensional, an np.float is 
        returned. If the axis is specified, then the output is an np.ndarray
        of one less dimension than the input array, with the same shape as the
        input array except for the specified axis.
    """

    # Exception handling
    if not isinstance(arr, np.ndarray):
        raise TypeError("Array must be a NumPy array.")
    if (axis != None) and (not isinstance(axis, int)):
        raise TypeError("Axis must be an integer or None.")
    if (axis != None) and (axis < -arr.ndim or axis >= arr.ndim):
        raise ValueError("Axis must be between .")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were ignored to calculate array maximum.")
        return np.nanmax(arr, axis)

    return np.max(arr, axis)

def histogram(arr, bins, range = None):
    """
    Calculates the histogram of an array.

    Args:
        arr (np.ndarray): Array to calculate the histogram of
        bins (int): Number of bins in histogram
        range (None or tuple of floats): Optional tuple of floats defining the
        range of the histogram. If None, the range of the array is used.

    Returns:
        tuple of 2 np.ndarray's: Tuple (counts, bin_edges) where counts is the
        number of elements in each bin and bin_edges is contains the value at
        each bin edge.
    """

    # Exception handling
    if not isinstance(arr, np.ndarray):
        raise TypeError("Array must be a NumPy array.")
    if not isinstance(bins, int):
        raise TypeError("Bins must be an integer.")
    if bins < 1:
        raise ValueError("Bins must be positive.")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were removed in calculating histogram.")
        arr = arr[~np.isnan(arr)]

    return np.histogram(arr, bins, range)

def quantile(arr, q, axis = None):
    """
    Calculates the specificed quantile of an array over a given axis. 
    If the axis is not specificed the desired quantile is calculated using all
    the entries in the array. If the array contains NaN elements, 
    the quantile is calculated ignoring them. 

    Args:
        arr (np.ndarray): Array to calculate the quantile of
        q (float): Quantile to calculate. Must be between 0 and 1.
        axis (int or None): Axis to calculate the quantile over. 
        If None, the standard deviation of all elements in the array is 
        calculated. Negative axis values start indexing from the end of the 
        array. Must have -arr.ndim <= axis < arr.ndim.

    Returns:
        np.ndarray or np.float: Desired quantile of the array over the 
        given axis. If the axis is not specified or the array is one dimensional, 
        an np.float is returned. If the axis is specified, then the output is 
        an np.ndarray of one less dimension than the input array, with the same 
        shape as the input array except for the specified axis.
    """

    # Exception handling
    if not isinstance(arr, np.ndarray):
        raise TypeError("Array must be a NumPy array.")
    if (axis != None) and (not isinstance(axis, int)):
        raise TypeError("Axis must be an integer or None.")
    if not isinstance(q, float):
        raise TypeError("Quantile must be a float.")
    if (axis != None) and (axis < -arr.ndim or axis >= arr.ndim):
        raise ValueError("Axis must be between .")
    if q < 0 or q > 1:
        raise ValueError("Quantile must be between 0 and 1.")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were ignored to calculate array quantiles")
        return np.nanquantile(arr, q, axis)

    return np.quantile(arr, q, axis)
