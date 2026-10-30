"""Shared feature preparation: one-hot age_band, fixed column order."""

import pandas as pd

from .data_gen import AGE_BANDS

NUMERIC = [
    "income",
    "loan_amount",
    "credit_history_years",
    "employment_years",
    "debt_to_income",
    "late_payments",
]

COLUMNS = NUMERIC + [f"age_band_{b}" for b in AGE_BANDS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df, columns=["age_band"], dtype=float)
    for col in COLUMNS:
        if col not in X.columns:
            X[col] = 0.0
    return X[COLUMNS]
