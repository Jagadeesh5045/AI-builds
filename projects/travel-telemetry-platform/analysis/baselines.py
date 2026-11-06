"""Frequency-aware rolling baselines.

A baseline is what a metric was expected to be, given its own recent history.
The window depends on the reporting frequency the caller is working at:
daily views compare against the last 7 days, weekly views against the last
4 weeks (28 days), monthly views against the last 12 weeks (84 days). The
current point is always excluded (shift by 1) so a spike cannot hide itself
inside its own baseline.
"""
from __future__ import annotations

import pandas as pd

WINDOWS = {"daily": 7, "weekly": 28, "monthly": 84}


def rolling_baseline(series: pd.Series, frequency: str = "daily", min_periods: int = 3) -> pd.Series:
    """Rolling median of prior values, window set by reporting frequency."""
    if frequency not in WINDOWS:
        raise ValueError(f"frequency must be one of {sorted(WINDOWS)}, got {frequency!r}")
    window = WINDOWS[frequency]
    return series.shift(1).rolling(window=window, min_periods=min_periods).median()


def rolling_mad(series: pd.Series, frequency: str = "daily", min_periods: int = 3) -> pd.Series:
    """Rolling median absolute deviation of prior values, same windowing."""
    if frequency not in WINDOWS:
        raise ValueError(f"frequency must be one of {sorted(WINDOWS)}, got {frequency!r}")
    window = WINDOWS[frequency]
    prior = series.shift(1)
    return prior.rolling(window=window, min_periods=min_periods).apply(
        lambda w: float(pd.Series(w).sub(w.median()).abs().median()), raw=False
    )


def add_baselines(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of df with daily, weekly and monthly baselines for column."""
    out = df.copy()
    for freq in WINDOWS:
        out[f"{column}_baseline_{freq}"] = rolling_baseline(out[column], frequency=freq)
    return out
