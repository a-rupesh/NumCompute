import numpy as np

class StreamingStats:
    """
    Allows online calculation of mean, variance, standard deviation, maximum
    and minimum of NumPy arrays along a given axis.

    Example 1: 
        # Calculates mean, variance, standard deviation, max and min of 
        # np.array([1, 2, 3, 4, 5, 6])

        x = np.array([1, 2, 3]
        y = np.array([4, 5, 6])
        welford = StreamingStats()
        welford.update(x)
        welford.update(y)

    Example 2:
        # Calculates mean, variance, standard deviation, max and min of 
        # np.array([[1, 2], [3, 4], [5, 6]]) for axis = 0 (i.e along columns)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6]])
        welford = StreamingStats(axis = 0)
        welford.update(x)
        welford.update(y)

    """

    def __init__(self, axis = None):
        self.n = 0
        self.mean = None
        self.M2 = None
        self.min = None
        self.max = None
        self.shape = None
        self.dim = None
        self.axis = axis

    def update(self, x):

        """
        Updates the streaming statistics with a new batch of data.

        Args:
            x (np.ndarray): Batch of data. If axis is None, x can be an array
            of any shape. Otherwise, for arrays x of dimension at least 2, they
            must be the same shape in the non-axis dimensions as all arrays 
            previously passed into the update function.
        """

        # Exception handling
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a NumPy array")
        if np.isnan(x).any():
            raise ValueError("Array contains NaNs.")
        if self.axis != None:
            if not isinstance(self.axis, int):
                raise ValueError("Axis must be None of or type int")
            if self.axis < -x.ndim or self.axis >= x.ndim:
                raise ValueError("Axis must satisify -x.ndim <= axis < x.ndim.")
            if self.dim != None:
                if self.dim != x.ndim:
                    raise ValueError("x must be the same dimension as previous array")
                if self.dim > 1 and not np.array_equal(np.delete(self.shape, self.axis), np.delete(x.shape, self.axis)):
                    raise ValueError("x must be the same shape as previous array in non-axis dimensions")

        # Update shape and dimension
        self.dim = x.ndim
        self.shape = x.shape

        # Determine batch size
        if self.axis != None:
            batch_size = x.shape[self.axis]
        else:
            batch_size = x.size

        # Calculate batch statistics
        mean_x = x.mean(axis = self.axis)
        M2_x = ((x - mean_x) ** 2).sum(axis = self.axis)

        if self.mean is None:
            self.n = 0
            self.mean = np.zeros_like(mean_x)
            self.M2 = np.zeros_like(M2_x)
            self.min = np.full_like(mean_x, np.inf)
            self.max = np.full_like(mean_x, -np.inf)

        # Update min & max
        self.min = np.minimum(self.min, x.min(axis = self.axis))
        self.max = np.maximum(self.max, x.max(axis = self.axis))

        # Update existing stats
        delta = mean_x - self.mean
        self.mean += delta * batch_size / (self.n + batch_size)
        self.M2 += M2_x + delta ** 2 * self.n * batch_size / (self.n + batch_size)
        self.n += batch_size

    @property
    def variance(self):
        """
        Population variance of the array along given axis.
        """

        if self.n < 2:
            return np.nan

        return self.M2 / self.n

    @property
    def std(self):
        """
        Population standard deviation of the array along given axis.
        """

        return np.sqrt(self.variance)


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
        raise ValueError("Axis must satisify -arr.ndim <= axis < arr.ndim.")

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
        raise ValueError("Axis must satisify -arr.ndim <= axis < arr.ndim.")

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
        raise ValueError("Axis must satisify -arr.ndim <= axis < arr.ndim.")

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
        raise ValueError("Axis must satisify -arr.ndim <= axis < arr.ndim.")

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
        raise ValueError("Axis must satisify -arr.ndim <= axis < arr.ndim.")

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
        raise ValueError("Axis must satisify -arr.ndim <= axis < arr.ndim.")
    if q < 0 or q > 1:
        raise ValueError("Quantile must be between 0 and 1.")

    # NaN handling
    if np.isnan(arr).any():
        print("Warning: NaNs were ignored to calculate array quantiles")
        return np.nanquantile(arr, q, axis)

    return np.quantile(arr, q, axis)
