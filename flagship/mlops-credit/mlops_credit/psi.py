"""Population Stability Index (PSI) per feature.

PSI compares the distribution of a feature in a reference (training)
sample against a new batch. Quantile bins are cut on the reference data
and reused for the new batch, so a shift shows up as a PSI rise.

Rule of thumb used by this project:
  PSI < 0.10  no significant shift
  0.10-0.25  moderate shift, monitor
  PSI > 0.25  significant drift, retrain
"""

import numpy as np
import pandas as pd

PSI_THRESHOLD = 0.25


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI between two samples of one numeric feature."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    e_perc = np.histogram(expected, breakpoints)[0] / len(expected)
    a_perc = np.histogram(actual, breakpoints)[0] / len(actual)
    e_perc = np.clip(e_perc, 0.0001, None)
    a_perc = np.clip(a_perc, 0.0001, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def psi_report(reference: pd.DataFrame, new_batch: pd.DataFrame,
               features: list) -> dict:
    return {f: psi(reference[f], new_batch[f]) for f in features}


def drift_detected(report: dict, threshold: float = PSI_THRESHOLD) -> bool:
    return any(v > threshold for v in report.values())
