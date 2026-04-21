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

##  Modules

### Data Handling
- `io.py`  
  Load CSV files with support for delimiters and missing values.

---

### Preprocessing
- `StandardScaler` — z-score normalization  
- `MinMaxScaler` — feature scaling  
- `SimpleImputer` — missing value replacement  
- `OneHotEncoder` — categorical encoding  

---

### Sorting & Searching
- Stable sorting  
- Multi-key sorting  
- Top-k selection (`argpartition`)  
- Quickselect (k-th smallest element)  
- Binary search  

---

### Ranking
- `rank()` with tie handling:
  - average
  - dense
  - ordinal  
- `percentile()` computation  

---

### Statistics
- Mean, median, standard deviation  
- Min / max  
- Quantiles  
- Histogram  
- Axis-based computations  

---

### Metrics
#### Classification
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

#### Regression
- Mean Squared Error (MSE)  


---

### Optimisation
- Finite difference gradient:
  - forward difference  
  - central difference  
- Jacobian computation  

---

### Pipeline
- Chaining transformers and models  
- Supports:
  - `fit()`
  - `transform()`
  - `predict()`  

---

### Utilities
- Array validation helpers  
- Activation functions (sigmoid, relu)  
- Stable softmax  
- LogSumExp (numerical stability)  
- Distance calculations  
- Top-k helpers  
- Batch generation  

---

## Running Tests

```bash
python -m pytest -q
