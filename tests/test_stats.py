import numpy as np
from stats import mean, median, std, min, max, histogram, quantile

def mean_test():
    """
    Testing for mean() function.
    """

    arr1 = np.array([0])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr4 = np.array([1, 2, np.nan, 3])

    assert mean(arr1) == 0
    assert mean(arr2) == 2
    assert (mean(arr3, 1) == np.array([2, 5, 8])).all()
    assert mean(arr4) == 2

    print("mean() passed all tests.\n")

def median_test():
    """
    Testing for median() function.
    """

    arr1 = np.array([0])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr4 = np.array([1, 2, np.nan, 3])

    assert median(arr1) == 0
    assert median(arr2) == 2
    assert (median(arr3, 1) == np.array([2, 5, 8])).all()
    assert median(arr4) == 2

    print("median() passed all tests.\n")

def std_test():
    """
    Testing for std() function.
    """

    arr1 = np.array([0])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr4 = np.array([1, 2, np.nan, 3])

    assert std(arr1) == 0
    assert std(arr2) == np.std(arr2)
    assert (std(arr3, 1) == np.std(arr3, axis=1)).all()
    assert std(arr4) == np.nanstd(arr4)

    print("std() passed all tests.\n")

def min_test():
    """
    Testing for min() function.
    """

    arr1 = np.array([0])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr4 = np.array([1, 2, np.nan, 3])

    assert min(arr1) == 0
    assert min(arr2) == 1
    assert (min(arr3, 1) == np.array([1, 4, 7])).all()
    assert min(arr4) == 1  # ignoring NaN

    print("min() passed all tests.\n")

def max_test():
    """
    Testing for max() function.
    """
    arr1 = np.array([0])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr4 = np.array([1, 2, np.nan, 3])

    assert max(arr1) == 0
    assert max(arr2) == 3
    assert (max(arr3, 1) == np.array([3, 6, 9])).all()
    assert max(arr4) == 3

    print("max() passed all tests.\n")

def histogram_test():
    """
    Testing for histogram() function.
    """

    arr1 = np.array([0])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1, 2, 3], [4, 5, 6]])
    arr4 = np.array([1, 2, np.nan, 3])

    # Test 1
    counts, bins = histogram(arr1, bins = 3, range = (-3, 3))
    assert (bins == np.array([-3, -1, 1, 3])).all()
    assert (counts == np.array([0, 1, 0])).all()

    # Test 2
    counts, bins = histogram(arr2, 4)
    assert (bins == np.array([1, 1.5, 2, 2.5, 3])).all()
    assert (counts == np.array([1, 0, 1, 1])).all()

    # Test 3
    counts, bins = histogram(arr3, 5, (0, 10))
    assert (bins == np.array([0, 2, 4, 6, 8, 10])).all()
    assert((counts == np.array([1, 2, 2, 1, 0]))).all()

    # Test 4
    counts, bins = histogram(arr4, 4)
    assert (bins == np.array([1, 1.5, 2, 2.5, 3])).all()
    assert (counts == np.array([1, 0, 1, 1])).all()

    print("histogram() passed all tests.\n")

def quantile_test():
    """
    Testing for quantile() function.
    """

    arr1 = np.array([0])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1, 2, 3], [4, 5, 6]])
    arr4 = np.array([1, 2, np.nan, 3])

    assert quantile(arr1, 0.5) == 0
    assert quantile(arr2, 0.5) == 2
    assert (quantile(arr3, 0.5, 1) == np.array([2, 5])).all()
    assert quantile(arr4, 0.5) == 2

    print("quantile() passed all tests.\n")

# Run tests
mean_test()
median_test()
std_test()
min_test()
max_test()
histogram_test()
quantile_test()

print("All tests were passed.")
