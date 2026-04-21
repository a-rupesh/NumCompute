"""Preprocessing utilities for NumCompute."""

from __future__ import annotations

import numpy as np


def _as_2d_array(X, *, dtype=None):
    arr = np.asarray(X, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    return arr


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    NaNs are ignored during fitting and preserved during transformation.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = _as_2d_array(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler must be fitted before transform().")
        X = _as_2d_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scale features to a specified range.

    NaNs are ignored during fitting and preserved during transformation.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        if len(feature_range) != 2:
            raise ValueError("feature_range must be a length-2 tuple.")
        low, high = feature_range
        if low >= high:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = (float(low), float(high))
        self.data_min_ = None
        self.data_max_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = _as_2d_array(X, dtype=float)
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("MinMaxScaler must be fitted before transform().")
        X = _as_2d_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")
        denom = self.data_max_ - self.data_min_
        denom = np.where(denom == 0, 1.0, denom)
        X_std = (X - self.data_min_) / denom
        low, high = self.feature_range
        return X_std * (high - low) + low

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputer:
    """
    Replace NaN values using a simple column-wise strategy.

    Supported strategies are "mean" and "constant".
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in {"mean", "constant"}:
            raise ValueError("strategy must be 'mean' or 'constant'.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = _as_2d_array(X, dtype=float)
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        else:
            self.statistics_ = np.full(X.shape[1], self.fill_value, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise ValueError("SimpleImputer must be fitted before transform().")
        X = _as_2d_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")
        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    One-hot encode categorical columns.

    Output is a dense integer NumPy array.
    Unknown categories during transform raise ValueError.
    """

    def __init__(self):
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = _as_2d_array(X, dtype=object)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before transform().")
        X = _as_2d_array(X, dtype=object)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count does not match fitted data.")

        encoded_columns = []
        for i in range(X.shape[1]):
            col = X[:, i]
            cats = self.categories_[i]
            unknown = ~np.isin(col, cats)
            if np.any(unknown):
                raise ValueError(f"Unknown category found in column {i}.")
            encoded_columns.append((col[:, None] == cats[None, :]).astype(int))
        return np.hstack(encoded_columns)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
