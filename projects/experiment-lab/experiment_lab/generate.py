"""Synthetic e-commerce A/B test data generator.

Generates a realistic, reproducible checkout-experiment dataset:
- 50/50 randomised assignment to control / treatment
- daily seasonality in traffic and conversion
- a pre-experiment covariate (pre-period spend) correlated with behaviour,
  which the analysis later exploits with CUPED variance reduction
- binary conversion outcome and revenue per user

Everything is seeded, so the dataset (and every number in the README) is
fully reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_P_CONTROL = 0.050
TRUE_P_TREATMENT = 0.068  # +1.8pp true lift, +36% relative


def generate_experiment(
    n_users: int = 4000,
    days: int = 14,
    seed: int = 42,
    p_control: float = TRUE_P_CONTROL,
    p_treatment: float = TRUE_P_TREATMENT,
) -> pd.DataFrame:
    """Generate one row per user for a simulated checkout experiment."""
    rng = np.random.default_rng(seed)

    user_id = [f"u{i:05d}" for i in range(n_users)]
    group = rng.choice(["control", "treatment"], size=n_users)

    # users enter the experiment across `days`, with weekend traffic dips
    day_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.65] * 2)[:days]
    day_weights /= day_weights.sum()
    day = rng.choice(np.arange(1, days + 1), size=n_users, p=day_weights)

    # latent shopping propensity: persists across pre/post periods (drives CUPED)
    propensity = rng.normal(0.0, 1.0, size=n_users)

    # pre-experiment covariate: spend in the 14 days before the experiment
    pre_spend = np.clip(28.0 + 14.0 * propensity + rng.normal(0, 10, n_users), 0, None)

    # conversion: group rate shifted by propensity, plus a weekend dip
    base = np.where(group == "control", p_control, p_treatment)
    weekend = np.isin(day % 7, [6, 0]).astype(float)  # day 6/7 -> weekend
    logit = np.log(base / (1 - base)) + 0.55 * propensity - 0.25 * weekend
    prob = 1.0 / (1.0 + np.exp(-logit))
    converted = (rng.random(n_users) < prob).astype(int)

    # revenue: converted users spend log-normally; past spenders stay big spenders
    # (strong pre/post correlation is exactly what CUPED exploits)
    spend = np.exp(np.log(42.0) + 0.80 * propensity + rng.normal(0, 0.35, n_users))
    revenue = np.round(converted * spend, 2)

    return pd.DataFrame(
        {
            "user_id": user_id,
            "group": group,
            "day": day,
            "pre_spend": np.round(pre_spend, 2),
            "converted": converted,
            "revenue": revenue,
        }
    )


def save_experiment(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
