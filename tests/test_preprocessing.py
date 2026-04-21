import numpy as np
import pytest

from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_standard_scaler_mean_zero():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0)


def test_standard_scaler_constant_column():
    X = np.array([[1, 2], [1, 4], [1, 6]], dtype=float)
    scaler = StandardScaler().fit(X)
    out = scaler.transform(X)
    assert np.allclose(out[:, 0], 0.0)


def test_minmax_scaler_range():
    X = np.array([[1], [2], [3]], dtype=float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    assert np.min(X_scaled) == -1
    assert np.max(X_scaled) == 1


def test_imputer_removes_nan():
    X = np.array([[1, np.nan], [3, 4]], dtype=float)
    imputer = SimpleImputer()
    X_filled = imputer.fit_transform(X)
    assert not np.isnan(X_filled).any()


def test_imputer_constant():
    X = np.array([[1, np.nan], [3, 4]], dtype=float)
    imputer = SimpleImputer(strategy="constant", fill_value=-5)
    X_filled = imputer.fit_transform(X)
    assert X_filled[0, 1] == -5


def test_one_hot_encoder_shape():
    X = np.array([["A"], ["B"], ["A"]], dtype=object)
    encoder = OneHotEncoder()
    X_enc = encoder.fit_transform(X)
    assert X_enc.shape == (3, 2)


def test_one_hot_encoder_unknown_category_raises():
    X = np.array([["A"], ["B"]], dtype=object)
    encoder = OneHotEncoder().fit(X)
    with pytest.raises(ValueError):
        encoder.transform(np.array([["C"]], dtype=object))
