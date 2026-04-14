import numpy as np


class StandardScaler:
    """Z-score normalization"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale features to a given range"""

    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.range = feature_range

    def fit(self, X):
        X = np.asarray(X)
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        denom = self.max_ - self.min_
        denom[denom == 0] = 1
        X_std = (X - self.min_) / denom
        return X_std * (self.range[1] - self.range[0]) + self.range[0]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace NaN values"""

    def __init__(self, strategy="mean", fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        X = np.asarray(X)

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError("Unsupported strategy")

        return self

    def transform(self, X):
        X = np.asarray(X)
        mask = np.isnan(X)
        return np.where(mask, self.statistics_, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """One-hot encode categorical data"""

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        X = np.asarray(X)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X)
        encoded_cols = []

        for i in range(X.shape[1]):
            col = X[:, i]
            categories = self.categories_[i]
            one_hot = (col[:, None] == categories).astype(int)
            encoded_cols.append(one_hot)

        return np.hstack(encoded_cols)

    def fit_transform(self, X):
        return self.fit(X).transform(X)