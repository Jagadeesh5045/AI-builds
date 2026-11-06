"""Synthetic mobile app telemetry generator (SYNTHETIC DATA ONLY).

Produces about 180 days of daily telemetry for a travel mobile app across the
domains the platform reports on: Bookings, Adoption, Engagement, App Health
and QE (quality engineering). A small, known set of anomalies is injected on
purpose so anomaly detection can be scored against ground truth.

All data is synthetic, generated from a seeded random process. It is not real
Amex Travel data, not real user data, and must never be presented as such.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 180
START_DATE = "2026-01-01"

# Injected anomalies: day offsets from START_DATE, recorded exactly.
CRASH_SPIKE_DAYS = [40, 95, 150]
BOOKING_DROP_DAYS = [70, 71]

DOMAINS = {
    "bookings": ["sessions", "booking_starts", "booking_completions", "revenue_gbp"],
    "adoption": ["new_installs", "registrations", "activations", "ios_installs", "android_installs"],
    "engagement": ["dau", "sessions_per_user", "search_users", "itinerary_views"],
    "app_health": ["crash_rate_pct", "anr_rate_pct", "store_rating", "app_version"],
    "quality": ["defects_opened", "defects_closed", "regression_failures"],
}


def generate_telemetry(n_days: int = N_DAYS, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Return (daily telemetry DataFrame, ground truth dict of injected anomalies)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 1.0 + 0.18 * np.sin(2 * np.pi * (dates.day_of_week.to_numpy()) / 7.0 + 0.6)
    trend = 1.0 + 0.0009 * t

    sessions = (52000 * weekly * trend * rng.normal(1.0, 0.03, n_days)).round().astype(int)
    booking_starts = (sessions * rng.normal(0.31, 0.008, n_days)).round().astype(int)
    booking_completions = (booking_starts * rng.normal(0.46, 0.010, n_days)).round().astype(int)
    revenue = booking_completions * rng.normal(212.0, 9.0, n_days)

    new_installs = (6800 * weekly * trend * rng.normal(1.0, 0.04, n_days)).round().astype(int)
    ios_share = rng.normal(0.57, 0.01, n_days)
    ios_installs = (new_installs * ios_share).round().astype(int)
    android_installs = new_installs - ios_installs
    registrations = (new_installs * rng.normal(0.62, 0.015, n_days)).round().astype(int)
    activations = (registrations * rng.normal(0.78, 0.015, n_days)).round().astype(int)

    dau = (38500 * weekly * trend * rng.normal(1.0, 0.03, n_days)).round().astype(int)
    sessions_per_user = np.clip(rng.normal(1.62, 0.05, n_days) * (2 - weekly) / 1.0, 1.1, 2.4)
    search_users = (dau * rng.normal(0.44, 0.012, n_days)).round().astype(int)
    itinerary_views = (dau * rng.normal(0.27, 0.010, n_days)).round().astype(int)

    crash_rate = np.clip(rng.normal(0.82, 0.09, n_days), 0.4, None)
    anr_rate = np.clip(rng.normal(0.31, 0.05, n_days), 0.1, None)
    store_rating = np.clip(rng.normal(4.55, 0.05, n_days), 1.0, 5.0)
    app_version = np.where(t < 60, "7.2.0", np.where(t < 130, "7.3.0", "7.4.0"))

    defects_opened = rng.poisson(14, n_days)
    defects_closed = rng.poisson(13, n_days)
    regression_failures = rng.poisson(3, n_days)

    # Inject the known anomalies.
    for d in CRASH_SPIKE_DAYS:
        crash_rate[d] = crash_rate[d] * 4.2
        anr_rate[d] = anr_rate[d] * 2.0
        regression_failures[d] += 9
    for d in BOOKING_DROP_DAYS:
        booking_completions[d] = int(booking_completions[d] * 0.48)
        revenue[d] = revenue[d] * 0.48

    df = pd.DataFrame(
        {
            "date": dates,
            "sessions": sessions,
            "booking_starts": booking_starts,
            "booking_completions": booking_completions,
            "revenue_gbp": revenue.round(2),
            "new_installs": new_installs,
            "registrations": registrations,
            "activations": activations,
            "ios_installs": ios_installs,
            "android_installs": android_installs,
            "dau": dau,
            "sessions_per_user": sessions_per_user.round(3),
            "search_users": search_users,
            "itinerary_views": itinerary_views,
            "crash_rate_pct": crash_rate.round(3),
            "anr_rate_pct": anr_rate.round(3),
            "store_rating": store_rating.round(2),
            "app_version": app_version,
            "defects_opened": defects_opened,
            "defects_closed": defects_closed,
            "regression_failures": regression_failures,
        }
    )

    ground_truth = {
        "crash_rate_pct": [str(dates[d].date()) for d in CRASH_SPIKE_DAYS],
        "booking_completions": [str(dates[d].date()) for d in BOOKING_DROP_DAYS],
    }
    return df, ground_truth


def main(out_dir: str | Path = "data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, truth = generate_telemetry()
    df.to_csv(out / "telemetry_daily.csv", index=False)
    (out / "anomalies_ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(df)} synthetic daily rows to {out / 'telemetry_daily.csv'}")
    print(f"Injected anomalies: {sum(len(v) for v in truth.values())} points, ground truth in anomalies_ground_truth.json")


if __name__ == "__main__":
    main()
