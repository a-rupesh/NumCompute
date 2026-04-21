import numpy as np
from numcompute.preprocessing import StandardScaler
from numcompute.pipeline import Pipeline

# Dummy model
class DummyModel:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[:, 0]


# Sample data
X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40]
])

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DummyModel())
])

pipe.fit(X)
preds = pipe.predict(X)

print("Predictions:", preds)