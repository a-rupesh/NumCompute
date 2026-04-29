# NumCompute

NumCompute is a lightweight, modular, NumPy-based scientific computing toolkit built for the Programming in AI course.

The project implements reusable, vectorised numerical components from scratch using only plain Python and NumPy. It focuses on numerical correctness, clean API design, testing, and reproducible benchmarking.

---

## Project Goals

This toolkit is designed to provide:

- reusable NumPy-based components
- vectorised implementations where possible
- clean and consistent APIs
- strong edge-case handling
- unit tests and reproducible benchmarks

---

## Team Contributions

- Member 1 (Rupesh Ashok Kumar):
  - Implemented IO module (load_csv)
  - Implemented preprocessing (StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder)
  - Created demo notebook and demo script

- Member 2 (Harishkumar Sivakumar):
  - Implemented sorting and searching (sort_array, top_k, binary_search, quickselect)
  - Implemented ranking (rank, percentile)

- Member 3 (Alexander Tarin Smith):
  - Implemented statistics (mean, median, std, quantile, histogram)
  - Implemented metrics (accuracy, precision, recall, f1, confusion_matrix, mse)

- Member 4 (Aakash Sai Ganeshbabu):
  - Implemented optimization (grad, jacobian)
  - Implemented pipeline and feature union
  - Assisted in integration and testing

- Testing:
  - All members contributed to writing and validating test cases

---

## Package Structure

```text
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
├── demo/
├── README.md
└── requirements.txt
---

Modules
io.py

CSV loading utilities using NumPy.

Features:

configurable delimiter support
missing value handling
NumPy array output
preprocessing.py

Preprocessing transformers with a simple fit/transform API.

Includes:

StandardScaler
MinMaxScaler
SimpleImputer
OneHotEncoder
sort_search.py

Sorting, searching, and top-k helpers.

Includes:

stable sorting
multi-key sorting
top-k selection
quickselect
binary search
rank.py

Ranking and percentile utilities.

Includes:

tie-aware ranking
percentile computation
multiple ranking methods
stats.py

Descriptive statistics with NumPy.

Includes:

mean
median
standard deviation
min / max
histogram
quantiles
metrics.py

Evaluation metrics for classification and regression.

Classification:

accuracy
precision
recall
f1-score
confusion matrix

Regression:

mean squared error
optim.py

Finite-difference optimisation utilities.

Includes:

numerical gradient estimation
Jacobian estimation
forward and central difference methods
pipeline.py

Simple pipeline abstraction for chaining transformers and estimators.

Includes:

Pipeline
FeatureUnion
utils.py

Shared helper utilities.

Includes:

array validation
activation functions
numerically stable softmax
logsumexp
distance helpers
top-k helper functions
batching utilities
benchmarking.py

Micro-benchmark utilities for comparing vectorised implementations with Python-loop implementations.

Includes:

benchmark timing helpers
repeated runs
summary tables
vectorised vs loop comparison utilities

```bash
python -m pytest -q
