"""Data-drift monitoring: Population Stability Index (PSI) per feature.

Compares an incoming batch against the training baseline. PSI per feature:
    sum((actual% - expected%) * ln(actual% / expected%))
Rule of thumb: PSI < 0.1 no shift, 0.1-0.25 moderate shift, > 0.25
significant shift (flags the batch). Bin edges come from the baseline
quantiles so the check is stable across features.

Usage:
    python drift.py            # checks a simulated shifted batch vs baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data
from config import PSI_WARN_THRESHOLD

N_BINS = 10
EPS = 1e-6


def baseline_bins(X_train: pd.DataFrame, n_bins: int = N_BINS) -> dict[str, np.ndarray]:
    """Quantile bin edges per feature, computed on the training data."""
    edges = {}
    for col in X_train.columns:
        qs = np.linspace(0, 1, n_bins + 1)
        e = X_train[col].quantile(qs).to_numpy()
        # de-duplicate edges for low-cardinality features
        e = np.unique(e)
        if len(e) < 3:
            e = np.array([X_train[col].min() - EPS, X_train[col].max() + EPS])
        e[0], e[-1] = -np.inf, np.inf
        edges[col] = e
    return edges


def psi(expected: pd.Series, actual: pd.Series, edges: np.ndarray) -> float:
    """Population Stability Index between baseline and a new batch."""
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / max(e_counts.sum(), 1) + EPS
    a_pct = a_counts / max(a_counts.sum(), 1) + EPS
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def check_batch(X_train: pd.DataFrame, X_batch: pd.DataFrame,
                threshold: float = PSI_WARN_THRESHOLD) -> dict:
    """PSI per feature + overall verdict for an incoming batch."""
    edges = baseline_bins(X_train)
    per_feature = {col: round(psi(X_train[col], X_batch[col], edges[col]), 4)
                   for col in X_train.columns}
    drifted = sorted([c for c, v in per_feature.items() if v > threshold],
                     key=lambda c: per_feature[c], reverse=True)
    return {
        "n_baseline": len(X_train),
        "n_batch": len(X_batch),
        "psi_per_feature": per_feature,
        "max_psi": max(per_feature.values()),
        "drifted_features": drifted,
        "verdict": "DRIFT" if drifted else "OK",
        "threshold": threshold,
    }


def simulate_shifted_batch(X_test: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Demo: a batch where two features have genuinely shifted upstream."""
    rng = np.random.default_rng(seed)
    batch = X_test.copy()
    cols = list(batch.columns)
    # simulate a sensor recalibration on the first feature and a population
    # shift on the second: mean + 1.5 std for each
    for col in cols[:2]:
        batch[col] = batch[col] + 1.5 * batch[col].std() + rng.normal(
            0, 0.01 * batch[col].std(), size=len(batch))
    return batch


def main() -> dict:
    df = data.load_dataset()
    X_train, X_test, _, _ = data.split_data(df)

    print("[drift] same-distribution batch (a second sample of test data):")
    clean = check_batch(X_train, X_test.sample(n=100, random_state=1))
    print(f"  verdict={clean['verdict']} max_psi={clean['max_psi']:.4f} "
          f"drifted={clean['drifted_features']}")

    print("[drift] shifted batch (two features moved +1.5 std):")
    shifted = check_batch(X_train, simulate_shifted_batch(X_test))
    print(f"  verdict={shifted['verdict']} max_psi={shifted['max_psi']:.4f} "
          f"drifted={shifted['drifted_features']}")
    return shifted


if __name__ == "__main__":
    main()
