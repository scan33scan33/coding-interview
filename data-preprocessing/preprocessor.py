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
        df = pd.read_csv(filepath)
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using mean strategy."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                self.stats['missing_values_imputed'][col] = missing_count

        return df

    def detect_and_cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers via IQR method and cap to boundaries."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Count outliers and cap them
            mask_lower = df[col] < lower_bound
            mask_upper = df[col] > upper_bound
            outlier_count = mask_lower.sum() + mask_upper.sum()

            df.loc[mask_lower, col] = lower_bound
            df.loc[mask_upper, col] = upper_bound

            if outlier_count > 0:
                self.stats['outliers_capped'][col] = outlier_count

        return df

    def normalize_features(self, df: pd.DataFrame,
                          fit: bool = False) -> pd.DataFrame:
        """Apply z-score normalization. If fit=True, learn from this data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if fit:
            self.means = {}
            self.stds = {}
            for col in numeric_cols:
                self.means[col] = df[col].mean()
                std_val = df[col].std()
                self.stds[col] = std_val if std_val > 0 else 1.0

        # Apply normalization
        for col in numeric_cols:
            mean = self.means[col]
            std = self.stds[col]
            df[col] = (df[col] - mean) / std

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: handle missing → cap outliers → normalize.
        Fit scaler on this data."""
        df = df.copy()
        self.stats['rows_processed'] = len(df)
        self.stats['columns_processed'] = len(df.columns)
        self.stats['missing_values_imputed'] = {}
        self.stats['outliers_capped'] = {}

        # Ensure numeric types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        df = self.handle_missing_values(df)
        df = self.detect_and_cap_outliers(df)
        df = self.normalize_features(df, fit=True)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned preprocessing to new data."""
        df = df.copy()

        # Ensure numeric types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        df = self.handle_missing_values(df)
        df = self.detect_and_cap_outliers(df)
        df = self.normalize_features(df, fit=False)

        return df

    def get_stats(self) -> Dict:
        """Return preprocessing statistics."""
        return self.stats.copy()
