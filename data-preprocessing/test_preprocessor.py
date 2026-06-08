import pandas as pd
import numpy as np
import pytest
import tempfile
import os
from preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""

    def test_load_csv(self):
        """Test 1: Load CSV and validate shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.csv")
            df = pd.DataFrame({
                'age': [25, 30, 35],
                'salary': [50000, 60000, 70000],
                'score': [0.5, 0.7, 0.9]
            })
            df.to_csv(filepath, index=False)

            preprocessor = DataPreprocessor()
            loaded_df = preprocessor.load_csv(filepath)

            assert loaded_df.shape == (3, 3)
            assert list(loaded_df.columns) == ['age', 'salary', 'score']

    def test_missing_values(self):
        """Test 2: Handle missing values with mean imputation."""
        df = pd.DataFrame({
            'values': [10.0, 20.0, np.nan, 40.0]
        })

        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(df)

        # Mean of [10, 20, 40] = 23.33
        assert result['values'].isnull().sum() == 0
        assert abs(result.iloc[2, 0] - 23.333333) < 0.01
        assert preprocessor.stats['missing_values_imputed']['values'] == 1

    def test_outlier_detection_iqr(self):
        """Test 3: Detect and cap outliers using IQR method."""
        df = pd.DataFrame({
            'values': [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        })

        preprocessor = DataPreprocessor()
        result = preprocessor.detect_and_cap_outliers(df)

        # Q1=2.25, Q3=4.75, IQR=2.5, bounds=[2.25-3.75, 4.75+3.75]=[-1.5, 8.5]
        # 100 > 8.5, so it should be capped to 8.5
        assert result.iloc[5, 0] == 8.5
        assert preprocessor.stats['outliers_capped']['values'] == 1

    def test_normalize_features(self):
        """Test 4: Z-score normalization (fit)."""
        df = pd.DataFrame({
            'values': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        preprocessor = DataPreprocessor()
        result = preprocessor.normalize_features(df, fit=True)

        # Mean = 3, Std = 1.581... (pandas sample std), normalized ~ [-1.265, -0.632, 0, 0.632, 1.265]
        assert abs(result.iloc[0, 0] - (-1.265)) < 0.01
        assert abs(result.iloc[2, 0] - 0.0) < 0.01
        assert abs(result.iloc[4, 0] - 1.265) < 0.01

    def test_fit_transform_split(self):
        """Test 5: Fit on train data, transform test data independently."""
        train_df = pd.DataFrame({
            'values': [10.0, 20.0, 30.0]
        })
        test_df = pd.DataFrame({
            'values': [15.0, 25.0]
        })

        preprocessor = DataPreprocessor()
        train_normalized = preprocessor.fit_transform(train_df)
        test_normalized = preprocessor.transform(test_df)

        # Train mean = 20, train std = 10.0 (sample std)
        # Train 10: (10-20)/10 = -1.0
        # Test 15: (15-20)/10 = -0.5
        # Test 25: (25-20)/10 = 0.5
        assert abs(train_normalized.iloc[0, 0] - (-1.0)) < 0.01
        assert abs(test_normalized.iloc[0, 0] - (-0.5)) < 0.01

    def test_zero_variance_column(self):
        """Test 6: Handle zero-variance (constant) columns."""
        df = pd.DataFrame({
            'constant': [5.0, 5.0, 5.0, 5.0]
        })

        preprocessor = DataPreprocessor()
        result = preprocessor.normalize_features(df, fit=True)

        # All values should map to 0 (no division by zero)
        assert all(result['constant'] == 0.0)

    def test_end_to_end_pipeline(self):
        """Test 7: Full preprocessing on realistic data."""
        df = pd.DataFrame({
            'age': [25.0, 30.0, np.nan, 45.0, 200.0],  # has missing and outlier
            'salary': [50000.0, 60000.0, 55000.0, np.nan, 150000.0],  # has missing and outlier
            'score': [0.5, 0.7, 0.6, 0.8, 0.9]  # clean
        })

        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(df)

        # Verify stats
        stats = preprocessor.get_stats()
        assert stats['rows_processed'] == 5
        assert stats['columns_processed'] == 3
        assert stats['missing_values_imputed']['age'] == 1
        assert stats['missing_values_imputed']['salary'] == 1
        assert stats['outliers_capped']['age'] >= 1  # 200 is outlier
        assert stats['outliers_capped']['salary'] >= 1  # 150k is outlier

        # Verify no NaN remain
        assert result.isnull().sum().sum() == 0

        # Verify normalization (mean ~0, std ~1)
        assert abs(result['score'].mean()) < 0.1
        assert abs(result['score'].std() - 1.0) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
