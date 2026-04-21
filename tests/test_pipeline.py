import numpy as np

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler


class DummyModel:
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        return (X[:, 0] > 0).astype(int)


class DoubleTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return 2 * X


def test_pipeline_fit_predict():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
    y = np.array([1, 1, 0])
    pipe = Pipeline([("scale", StandardScaler()), ("model", DummyModel())])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (3,)


def test_pipeline_fit_transform_with_transformers():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([("scale", StandardScaler()), ("double", DoubleTransformer())])
    Xt = pipe.fit_transform(X)
    assert Xt.shape == X.shape
