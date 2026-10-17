"""Dataset loading and validation for drift-guard.

Demo task: binary classification on the Wisconsin Breast Cancer dataset
(sklearn-bundled, 569 rows x 30 numeric features). The pipeline itself is
dataset-agnostic: swap `load_dataset()` to point at any tabular CSV and the
rest (training, evaluation, drift, serving) works unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE


TARGET_COL = "target"
DATASET_NAME = "breast_cancer_wisconsin"
POSITIVE_LABEL = "malignant"


def load_dataset() -> pd.DataFrame:
    """Load the demo dataset as a validated DataFrame."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df = df.rename(columns={"target": TARGET_COL})
    validate_dataframe(df)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Fail fast on the data problems that silently ruin model training."""
    if df.empty:
        raise ValueError("Dataset is empty.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column {TARGET_COL!r}.")
    if df[TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values.")
    if df[TARGET_COL].nunique() != 2:
        raise ValueError("Expected a binary target, got "
                         f"{df[TARGET_COL].nunique()} classes.")
    dupes = int(df.duplicated().sum())
    if dupes:
        raise ValueError(f"Dataset contains {dupes} duplicate rows.")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    non_numeric = [c for c in feature_cols
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    if df[feature_cols].isna().any().any():
        # Not fatal: the training pipeline imputes, but surface it loudly.
        n_missing = int(df[feature_cols].isna().sum().sum())
        print(f"[data] note: {n_missing} missing feature values "
              f"(will be median-imputed in the pipeline)")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COL]


def split_data(df: pd.DataFrame):
    """Stratified train/holdout split. Returns X_train, X_test, y_train, y_test."""
    X = df[feature_columns(df)]
    y = df[TARGET_COL]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def describe(df: pd.DataFrame) -> dict:
    """Small dataset profile used in the model card."""
    y = df[TARGET_COL]
    return {
        "dataset": DATASET_NAME,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_columns(df))),
        "positive_rate": float(y.mean()),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
    }


if __name__ == "__main__":
    df = load_dataset()
    print(describe(df))
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"train={X_train.shape} test={X_test.shape} "
          f"train_pos_rate={y_train.mean():.3f} test_pos_rate={y_test.mean():.3f}")
