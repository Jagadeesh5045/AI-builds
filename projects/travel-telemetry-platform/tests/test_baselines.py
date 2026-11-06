"""Baselining correctness on small, hand-checkable series."""
import numpy as np
import pandas as pd
import pytest

from analysis.baselines import WINDOWS, add_baselines, rolling_baseline


def test_constant_series_baseline_equals_value():
    series = pd.Series([10.0] * 30)
    baseline = rolling_baseline(series, frequency="daily")
    assert baseline.iloc[-1] == 10.0
    assert baseline.iloc[10] == 10.0


def test_baseline_excludes_current_point():
    # A single huge current value must not move its own baseline.
    series = pd.Series([10.0] * 10 + [1000.0])
    baseline = rolling_baseline(series, frequency="daily")
    assert baseline.iloc[-1] == 10.0


def test_hand_checked_rolling_median():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    baseline = rolling_baseline(series, frequency="daily", min_periods=3)
    # At index 4 the prior window is [1,2,3,4] (window 7, shifted), median 2.5.
    assert baseline.iloc[4] == pytest.approx(2.5)
    # At index 7 the prior window is [1..7], median 4.0.
    assert baseline.iloc[7] == pytest.approx(4.0)


def test_frequency_windows_and_validation():
    assert WINDOWS == {"daily": 7, "weekly": 28, "monthly": 84}
    with pytest.raises(ValueError):
        rolling_baseline(pd.Series([1.0, 2.0]), frequency="hourly")


def test_add_baselines_columns(telemetry):
    df, _ = telemetry
    out = add_baselines(df.head(100), "dau")
    for freq in ("daily", "weekly", "monthly"):
        assert f"dau_baseline_{freq}" in out.columns
    assert np.isnan(out["dau_baseline_daily"].iloc[0])
