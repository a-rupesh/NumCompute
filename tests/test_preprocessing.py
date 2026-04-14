import numpy as np
from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
    OneHotEncoder
)


def test_standard_scaler_mean_zero():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(np.mean(X_scaled, axis=0), 0)


def test_minmax_scaler_range():
    X = np.array([[1], [2], [3]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.min(X_scaled) == 0
    assert np.max(X_scaled) == 1


def test_imputer_removes_nan():
    X = np.array([[1, np.nan], [3, 4]])
    imputer = SimpleImputer()
    X_filled = imputer.fit_transform(X)
    assert not np.isnan(X_filled).any()


def test_one_hot_encoder_shape():
    X = np.array([["A"], ["B"], ["A"]])
    encoder = OneHotEncoder()
    X_enc = encoder.fit_transform(X)
    assert X_enc.shape[1] == 2