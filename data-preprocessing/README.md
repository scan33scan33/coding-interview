# Data Preprocessing Pipeline

**Problem:** Build a ML data preprocessing pipeline that cleans, validates, and normalizes a dataset. Given a CSV file with numerical features, handle missing values, detect outliers, and apply feature scaling to prepare data for model training.

---

## Core Requirements

The pipeline must implement:

1. **Data Loading & Validation**
   - Load CSV files with proper type handling
   - Validate that required columns exist
   - Track data quality metrics (missing counts, outliers)

2. **Missing Value Handling**
   - Detect columns with NaN / missing values
   - Impute using mean strategy (replace with column mean)
   - Preserve original values for "age" and "salary" columns if < 5% missing; otherwise impute

3. **Outlier Detection (IQR Method)**
   - For each numeric column, compute Q1 (25th percentile) and Q3 (75th percentile)
   - Calculate IQR = Q3 - Q1
   - Flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] as outliers
   - Cap outliers to the boundaries (don't remove rows)

4. **Feature Normalization (StandardScaler)**
   - Apply z-score normalization: `x_norm = (x - mean) / std_dev`
   - Fit on training data, transform both train and test sets
   - Handle edge case: if std_dev = 0, set normalized value to 0

5. **Output**
   - Return cleaned DataFrame with consistent dtypes
   - Log preprocessing statistics: rows processed, missing values handled, outliers capped, features normalized

---

## Interface

```python
class DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with empty state."""
        pass

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file and return as DataFrame."""
        pass

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using mean strategy."""
        pass

    def detect_and_cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers via IQR method and cap to boundaries."""
        pass

    def normalize_features(self, df: pd.DataFrame, 
                         fit: bool = False) -> pd.DataFrame:
        """Apply z-score normalization. If fit=True, learn from this data."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: load → handle missing → cap outliers → normalize.
        Fit scaler on this data."""
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned preprocessing to new data (same as fit_transform 
        but uses previously fit scaler)."""
        pass

    def get_stats(self) -> dict:
        """Return preprocessing statistics: {
            'rows_processed': int,
            'columns_processed': int,
            'missing_values_imputed': dict,  # {col: count}
            'outliers_capped': dict,         # {col: count}
        }"""
        pass
```

---

## Behavior Notes

- **Mean imputation:** Only fill NaN cells. Compute mean excluding NaN values.
- **Outlier capping:** Modify values in place (don't drop rows). Outlier count = number of values capped.
- **Normalization edge case:** If a feature has std_dev = 0 (all values identical), normalized values should be 0.
- **Train/test split:** The preprocessor must remember the mean and std_dev from `fit_transform` and apply them during `transform`.
- **Data types:** After processing, all numeric columns should be float64.

---

## Example Workflow

```python
# Create sample data
train_df = pd.DataFrame({
    'age': [25, 30, None, 45, 200],  # 200 is an outlier
    'salary': [50000, 60000, 55000, None, 150000],  # None and 150k
    'score': [0.5, 0.7, 0.6, 0.8, 0.9]
})

# Fit on training data
preprocessor = DataPreprocessor()
train_clean = preprocessor.fit_transform(train_df)
# - age: mean=(25+30+45)/3=33.33, impute None → 33.33, cap 200 → upper_bound
# - salary: mean=(50k+60k+55k+150k)/4=78.75k, impute None → 78.75k
# - score: no missing, no outliers, just normalize
# - All features z-score normalized using train stats

test_df = pd.DataFrame({
    'age': [28, None, 50],
    'salary': [52000, 61000, None],
    'score': [0.6, 0.75, 0.85]
})

# Transform test data using train statistics
test_clean = preprocessor.transform(test_df)
# - age: impute None using train mean, cap outliers, normalize using train stats
# - salary: same
# - score: normalize using train stats

stats = preprocessor.get_stats()
# {'rows_processed': 5, 'columns_processed': 3, 
#  'missing_values_imputed': {'age': 1, 'salary': 1},
#  'outliers_capped': {'age': 1, 'salary': 1}}
```

---

## Test Cases

| Test | Validates | Input | Expected Output |
|------|-----------|-------|-----------------|
| 1 — Load & Validate | CSV loading and basic validation | CSV file with 3 cols × 5 rows | DataFrame shape (5, 3), numeric dtypes |
| 2 — Missing Values | Mean imputation with NaN handling | Column with values [10, 20, NaN, 40] | [10, 20, 23.33, 40] (filled NaN) |
| 3 — Outlier Detection | IQR method caps extreme values | [1, 2, 3, 4, 5, 100] (100 is outlier) | [1, 2, 3, 4, 5, upper_bound] |
| 4 — Z-Score Normalization | Features scaled to mean=0, std=1 | [1, 2, 3, 4, 5] | Approx [-1.41, -0.71, 0, 0.71, 1.41] |
| 5 — Fit/Transform Split | Train scaler on one set, apply to another | Train: [10, 20, 30]; Test: [15, 25] | Test normalized using train mean/std |
| 6 — Zero Variance Column | Handle columns where all values are identical | [5, 5, 5, 5] (std_dev = 0) | [0, 0, 0, 0] (no division by zero) |
| 7 — End-to-End Pipeline | Full preprocessing on realistic data | DataFrame with mixed missing/outliers | Clean, normalized, stats logged |

---

## Implementation Tips

- Use `pandas` for DataFrames and `numpy` for numerical operations.
- `df.describe()` and `df.isnull().sum()` are useful for inspection.
- For IQR: `Q1 = df[col].quantile(0.25)`, `Q3 = df[col].quantile(0.75)`.
- For z-score: track `self.means` and `self.stds` after fit so you can reuse them in transform.
- Edge case: when learning from data with all identical values, store std as 1.0 (or handle division by zero gracefully).
