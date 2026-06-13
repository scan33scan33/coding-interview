import pandas as pd
import numpy as np
from typing import Dict


class DataPreprocessor:
    """ML data preprocessing pipeline: load, clean, normalize."""

    def __init__(self):
        """Initialize the preprocessor with empty state."""
        self.means = None
        self.stds = None
        self.stats = {
            'rows_processed': 0,
            'columns_processed': 0,
            'missing_values_imputed': {},
            'outliers_capped': {},
        }

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file and return as DataFrame."""
        return pd.read_csv(filepath)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using mean strategy."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate missing counts before imputation
        missing_counts = df[numeric_cols].isnull().sum()
        self.stats['missing_values_imputed'].update(
            missing_counts[missing_counts > 0].to_dict()
        )

        # Vectorized mean imputation
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        return df

    def detect_and_cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers via IQR method and cap to boundaries."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate IQR bounds for all numeric columns at once
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        lower_bounds = Q1 - 1.5 * IQR
        upper_bounds = Q3 + 1.5 * IQR

        # Count outliers before capping
        outlier_counts = (
            (df[numeric_cols] < lower_bounds).sum() +
            (df[numeric_cols] > upper_bounds).sum()
        )
        self.stats['outliers_capped'].update(
            outlier_counts[outlier_counts > 0].to_dict()
        )

        # Vectorized clipping
        for col in numeric_cols:
            df[col] = df[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])

        return df

    def normalize_features(self, df: pd.DataFrame,
                          fit: bool = False) -> pd.DataFrame:
        """Apply z-score normalization. If fit=True, learn from this data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if fit:
            self.means = df[numeric_cols].mean().to_dict()
            stds = df[numeric_cols].std()
            self.stds = (stds.replace(0, 1.0)).to_dict()

        # Vectorized normalization
        means_series = pd.Series(self.means, dtype=float)
        stds_series = pd.Series(self.stds, dtype=float)
        df[numeric_cols] = (df[numeric_cols] - means_series) / stds_series

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: handle missing → cap outliers → normalize.
        Fit scaler on this data."""
        df = df.copy()
        self.stats['rows_processed'] = len(df)
        self.stats['columns_processed'] = len(df.columns)
        self.stats['missing_values_imputed'] = {}
        self.stats['outliers_capped'] = {}

        # Ensure numeric types (vectorized)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(float)

        df = self.handle_missing_values(df)
        df = self.detect_and_cap_outliers(df)
        df = self.normalize_features(df, fit=True)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned preprocessing to new data."""
        df = df.copy()

        # Ensure numeric types (vectorized)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(float)

        df = self.handle_missing_values(df)
        df = self.detect_and_cap_outliers(df)
        df = self.normalize_features(df, fit=False)

        return df

    def get_stats(self) -> Dict:
        """Return preprocessing statistics."""
        return self.stats.copy()
