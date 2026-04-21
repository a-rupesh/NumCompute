# NumCompute

NumCompute is a lightweight NumPy-based numerical computing toolkit built for the Programming in AI course.

The goal of this project is to implement core data processing, mathematical, and evaluation utilities from scratch using **vectorised NumPy operations only**.

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
