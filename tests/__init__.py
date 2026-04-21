"""Public package exports for NumCompute."""

from .benchmarking import (
    benchmark_function,
    compare_functions,
    format_benchmark_table,
    loop_sum_of_squares,
    vectorized_sum_of_squares,
)
from .io import load_csv
from .metrics import accuracy, confusion_matrix, f1, mse, precision, recall
from .optim import grad, jacobian
from .pipeline import FeatureUnion, Pipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .rank import percentile, rank
from .sort_search import binary_search, lexsort_rows, quickselect, sort_array, top_k
from .stats import histogram, max, mean, median, min, quantile, std
from .utils import (
    check_array,
    euclidean_distance,
    logsumexp,
    make_batches,
    pairwise_euclidean,
    relu,
    sigmoid,
    stable_softmax,
    topk_indices,
)

__all__ = [
    "load_csv",
    "StandardScaler",
    "MinMaxScaler",
    "SimpleImputer",
    "OneHotEncoder",
    "sort_array",
    "lexsort_rows",
    "top_k",
    "binary_search",
    "quickselect",
    "rank",
    "percentile",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "histogram",
    "quantile",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "grad",
    "jacobian",
    "Pipeline",
    "FeatureUnion",
    "check_array",
    "sigmoid",
    "relu",
    "logsumexp",
    "stable_softmax",
    "euclidean_distance",
    "pairwise_euclidean",
    "topk_indices",
    "make_batches",
    "benchmark_function",
    "compare_functions",
    "format_benchmark_table",
    "vectorized_sum_of_squares",
    "loop_sum_of_squares",
]
