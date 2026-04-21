"""
NumCompute Demo Script

Run:
    python demo/demo.py

This demonstrates:
- CSV loading
- preprocessing
- sort/search
- ranking
- statistics
- metrics
- optimisation
- pipeline
- benchmarking
"""

import numpy as np
import tempfile

from numcompute.io import load_csv
from numcompute.preprocessing import StandardScaler, SimpleImputer, OneHotEncoder
from numcompute.sort_search import top_k, binary_search
from numcompute.rank import rank, percentile
from numcompute.stats import mean, std, histogram, quantile
from numcompute.metrics import accuracy, precision, recall, f1, confusion_matrix, mse
from numcompute.optim import grad, jacobian
from numcompute.pipeline import Pipeline
from numcompute.benchmarking import (
    compare_functions,
    vectorized_sum_of_squares,
    loop_sum_of_squares,
    format_benchmark_table,
)


def demo_io():
    print("\n=== CSV Loading ===")
    csv_text = "1,10,\n2,,0\n3,30,1\n4,40,0"

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        f.write(csv_text)
        path = f.name

    X = load_csv(path, delimiter=",")
    print(X)
    return X


def demo_preprocessing(X):
    print("\n=== Preprocessing ===")

    X_num = X[:, :2]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_num)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("Imputed:\n", X_imputed)
    print("Scaled:\n", X_scaled)

    X_cat = np.array([["A"], ["B"], ["A"], ["C"]])
    encoder = OneHotEncoder()
    print("One-hot:\n", encoder.fit_transform(X_cat))

    return X_num


def demo_sort_rank():
    print("\n=== Sort / Search / Rank ===")

    values = np.array([10, 30, 20, 20, 50, 40])

    print("Top-3:", top_k(values, k=3))
    print("Binary search (30):", binary_search(np.sort(values), 30))
    print("Rank:", rank(values))
    print("Percentile 75:", percentile(values, 75))


def demo_stats(values):
    print("\n=== Statistics ===")

    print("Mean:", mean(values))
    print("Std:", std(values))
    print("Quantile:", quantile(values, 0.5))

    counts, bins = histogram(values, bins=4)
    print("Histogram counts:", counts)
    print("Histogram bins:", bins)


def demo_metrics():
    print("\n=== Metrics ===")

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    print("Accuracy:", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    y_true_reg = np.array([1.0, 2.0, 3.0])
    y_pred_reg = np.array([1.2, 1.9, 2.8])
    print("MSE:", mse(y_true_reg, y_pred_reg))


def demo_optim():
    print("\n=== Optimisation ===")

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    def F(x):
        return np.array([
            x[0] + x[1],
            x[0] * x[1],
        ])

    x = np.array([3.0, 4.0])

    print("Gradient:", grad(f, x))
    print("Jacobian:\n", jacobian(F, x))


def demo_pipeline(X):
    print("\n=== Pipeline ===")

    class DummyModel:
        def fit(self, X, y=None):
            self.threshold_ = np.mean(X[:, 0])
            return self

        def predict(self, X):
            return (X[:, 0] > self.threshold_).astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", DummyModel()),
    ])

    y = np.array([0, 0, 1, 1])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    print("Predictions:", preds)


def demo_benchmark():
    print("\n=== Benchmark ===")

    x = np.random.rand(1_000_000)

    result = compare_functions(
        vectorized_sum_of_squares,
        loop_sum_of_squares,
        x,
        repeat=3,
        warmup=1,
    )

    print(format_benchmark_table([result["first"], result["second"]]))
    print("Speedup:", f"{result['speedup_vs_slow']:.2f}x")


def main():
    print("NumCompute Demo")
    print("=" * 40)

    X = demo_io()
    X_num = demo_preprocessing(X)

    demo_sort_rank()

    values = np.array([10, 30, 20, 20, 50, 40])
    demo_stats(values)

    demo_metrics()
    demo_optim()
    demo_pipeline(X_num)
    demo_benchmark()

    print("\nDone.")


if __name__ == "__main__":
    main()
