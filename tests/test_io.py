import numpy as np
import tempfile

from numcompute.io import load_csv


def test_load_csv_basic():
    data = "1,2\n3,4"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.flush()
        arr = load_csv(f.name)
    assert arr.shape == (2, 2)
    assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_load_csv_missing_values():
    data = "1,\n3,4"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.flush()
        arr = load_csv(f.name)
    assert np.isnan(arr[0, 1])


def test_load_csv_skip_header():
    data = "a,b\n1,2\n3,4"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.flush()
        arr = load_csv(f.name, skip_header=1)
    assert arr.shape == (2, 2)
