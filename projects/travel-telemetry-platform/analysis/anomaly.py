"""Anomaly detection: rolling median plus MAD modified z-score.

Method, exactly as specified for this platform: for each point, compare it to
the rolling median of its own recent history. Scale the deviation by the
rolling median absolute deviation (MAD) to get the modified z-score
(Iglewicz and Hoaglin): z = 0.6745 * (x - median) / MAD. Points with
|z| above the threshold (default 3.5) are flagged. The current point is
excluded from its own baseline window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.baselines import WINDOWS

MODIFIED_Z_CONSTANT = 0.6745


def modified_z_scores(series: pd.Series, frequency: str = "daily", window: int | None = None) -> pd.Series:
    """Modified z-score of each point against its rolling median and MAD."""
    if frequency not in WINDOWS:
        raise ValueError(f"frequency must be one of {sorted(WINDOWS)}, got {frequency!r}")
    win = window or WINDOWS[frequency]
    prior = series.shift(1)
    median = prior.rolling(window=win, min_periods=3).median()
    mad = prior.rolling(window=win, min_periods=3).apply(
        lambda w: float(pd.Series(w).sub(w.median()).abs().median()), raw=False
    )
    mad = mad.replace(0, np.nan)
    return MODIFIED_Z_CONSTANT * (series - median) / mad


def seasonal_modified_z_scores(series: pd.Series, period: int = 7, lookback: int = 12) -> pd.Series:
    """Modified z-score against the same seasonal phase in prior periods.

    Telemetry has a strong weekly cycle: a Monday compared against a Sunday
    baseline looks anomalous every week. The estimator is unchanged (rolling
    median plus MAD modified z-score), but the baseline pool for each point is
    that point's own weekday over the prior `lookback` weeks, which is the
    standard seasonal treatment for this method.
    """
    lanes = pd.concat([series.shift(period * k) for k in range(1, lookback + 1)], axis=1)
    lanes.columns = range(lookback)
    counts = lanes.notna().sum(axis=1)
    median = lanes.median(axis=1)
    # MAD via scipy (scale=1 returns the raw median absolute deviation).
    mad = lanes.apply(lambda row: float(stats.median_abs_deviation(row.dropna(), scale=1.0)), axis=1)
    mad = mad.replace(0, np.nan)
    scores = MODIFIED_Z_CONSTANT * (series - median) / mad
    return scores.where(counts >= 3)


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.5,
    frequency: str = "daily",
    date_column: str = "date",
    seasonal_period: int | None = 7,
) -> pd.DataFrame:
    """Return flagged points for one metric column, with scores attached.

    Daily telemetry is scored with the weekday-matched (seasonal) variant by
    default; pass seasonal_period=None for the plain rolling variant.
    """
    if seasonal_period:
        scores = seasonal_modified_z_scores(df[column], period=seasonal_period)
    else:
        scores = modified_z_scores(df[column], frequency=frequency)
    out = pd.DataFrame(
        {
            "date": df[date_column],
            "metric": column,
            "value": df[column],
            "modified_z": scores,
        }
    )
    flagged = out[out["modified_z"].abs() > threshold].copy()
    flagged["direction"] = np.where(flagged["modified_z"] > 0, "high", "low")
    return flagged.sort_values("date").reset_index(drop=True)


def score_against_ground_truth(flagged_dates: list[str], truth_dates: list[str]) -> dict:
    """Confusion counts for one metric: detected dates vs injected dates."""
    flagged, truth = set(flagged_dates), set(truth_dates)
    return {
        "true_positives": len(flagged & truth),
        "false_positives": len(flagged - truth),
        "missed": len(truth - flagged),
        "detected_dates": sorted(flagged),
    }
