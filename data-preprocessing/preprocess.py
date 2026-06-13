"""Data preprocessing pipeline for machine learning.

Handles CSV loading, missing value imputation, outlier capping, and missing
value indicators. Production-ready with comprehensive error handling.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def is_missing(value: float) -> bool:
    """Check if a value is missing (NaN or Inf)."""
    return np.isnan(value) or np.isinf(value)


def read_csv(filename: str) -> Dict[str, List[float]]:
    """Load CSV file and convert to float arrays.

    Args:
        filename: Path to CSV file. First row is column headers.

    Returns:
        Dict mapping column names to lists of float values.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If CSV is malformed or values can't be converted to float.
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {filename}")

    column_names = []
    feature_values: List[List[float]] = []
    num_features = 0

    try:
        with open(file_path, encoding='utf-8') as f:
            csv_reader = csv.reader(f)

            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    if not row:
                        raise ValueError("CSV file is empty")
                    column_names = row
                    feature_values = [[] for _ in range(len(column_names))]
                    num_features = len(column_names)
                else:
                    if len(row) != num_features:
                        raise ValueError(
                            f"Row {row_idx}: expected {num_features} columns, got {len(row)}"
                        )

                    for col_idx, raw_value in enumerate(row):
                        try:
                            # Empty string becomes NaN
                            value = np.nan if raw_value.strip() == '' else float(raw_value)
                        except ValueError as e:
                            col_name = column_names[col_idx]
                            raise ValueError(
                                f"Row {row_idx}, column '{col_name}': "
                                f"cannot convert '{raw_value}' to float"
                            ) from e
                        feature_values[col_idx].append(value)

        if len(feature_values[0]) == 0:
            raise ValueError("CSV has no data rows")

        return {column_names[i]: feature_values[i] for i in range(num_features)}

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}") from e


def process_column(
    values: List[float],
    use_median: bool = False,
    percentile: float = 99.0,
) -> Tuple[List[float], List[float]]:
    """Impute missing values and cap outliers.

    Args:
        values: List of feature values (may contain NaN/Inf).
        use_median: Use median instead of mean for imputation (default: False).
        percentile: Percentile threshold for outlier capping (default: 99).

    Returns:
        Tuple of (processed_values, missing_indicators).
        - processed_values: Missing values imputed, outliers capped.
        - missing_indicators: 1.0 if missing, 0.0 if valid.
    """
    if not values:
        raise ValueError("values cannot be empty")

    # Identify valid (non-missing) values
    valid_values = [v for v in values if not is_missing(v)]

    # Compute imputation value
    if len(valid_values) == 0:
        logger.warning("Column has all missing values; using 0.0 for imputation")
        imputation_value = 0.0
    else:
        imputation_value = (
            float(np.median(valid_values))
            if use_median
            else float(np.mean(valid_values))
        )

    # Step 1: Impute missing values and track them
    processed_values = []
    missing_indicators = []

    for v in values:
        if is_missing(v):
            processed_values.append(imputation_value)
            missing_indicators.append(1.0)
        else:
            processed_values.append(v)
            missing_indicators.append(0.0)

    # Step 2: Cap outliers at the percentile threshold
    if len(valid_values) > 1:
        cap_value = float(np.percentile(valid_values, percentile))
        for i in range(len(processed_values)):
            if processed_values[i] > cap_value:
                processed_values[i] = cap_value

    return processed_values, missing_indicators


def preprocess_features(
    feature_dict: Dict[str, List[float]],
    use_median: bool = False,
    percentile: float = 99.0,
    add_missing_indicators: bool = True,
) -> Dict[str, List[float]]:
    """Preprocess all features: impute missing values, cap outliers, add indicators.

    Args:
        feature_dict: Dict mapping feature names to value lists.
        use_median: Use median instead of mean for imputation (default: False).
        percentile: Percentile for outlier capping (default: 99.0).
        add_missing_indicators: Add missing value indicator features (default: True).

    Returns:
        Dict with processed features. Original features preserved; new features
        added with suffix "_missing" for missing value indicators.

    Raises:
        ValueError: If feature_dict is empty or features have different lengths.
    """
    if not feature_dict:
        raise ValueError("feature_dict cannot be empty")

    # Validate all features have the same length
    lengths = {name: len(values) for name, values in feature_dict.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All features must have the same length. Got: {lengths}")

    modified_feature_dict: Dict[str, List[float]] = {}

    for feature_name, feature_values in feature_dict.items():
        processed, indicators = process_column(
            feature_values,
            use_median=use_median,
            percentile=percentile,
        )
        modified_feature_dict[feature_name] = processed

        if add_missing_indicators and any(indicators):
            modified_feature_dict[f"{feature_name}_missing"] = indicators

    return modified_feature_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage:
    # data = read_csv("data.csv")
    # processed = preprocess_features(data, use_median=True, percentile=95.0)
    # print(f"Features: {list(processed.keys())}")
