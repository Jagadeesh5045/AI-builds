"""Synthetic card transaction generator.

Deterministic (seed 42). Fraud labels come from an explicit risk-score
process over observable signals (foreign, online, night hour, new device,
velocity, amount, risky category combinations), so a model trained on the
observable columns can genuinely learn the pattern. A fixed set of dirty
rows is then injected so the data-quality stage has known defects to catch:
duplicate transaction ids, negative or zero amounts, missing customers and
invalid channels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_CLEAN = 12_000
START = pd.Timestamp("2026-08-01")
DAYS = 90

CATEGORIES = {
    "groceries": (38.0, 0.55),
    "fuel": (52.0, 0.45),
    "retail": (64.0, 0.70),
    "dining": (34.0, 0.60),
    "travel": (210.0, 0.75),
    "electronics": (180.0, 0.80),
    "online_marketplace": (58.0, 0.75),
    "gambling": (45.0, 0.90),
    "utilities": (95.0, 0.40),
    "entertainment": (28.0, 0.65),
}
CATEGORY_NAMES = list(CATEGORIES)
CHANNELS = ["in_store", "online", "contactless", "atm"]
CHANNEL_P = [0.42, 0.31, 0.22, 0.05]
RISKY_ONLINE_CATEGORIES = {"gambling", "electronics", "travel"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_clean(rng: np.random.Generator, n: int = N_CLEAN) -> pd.DataFrame:
    category = rng.choice(CATEGORY_NAMES, size=n, p=[0.16, 0.09, 0.15, 0.12, 0.07, 0.08, 0.13, 0.04, 0.08, 0.08])
    means = np.array([CATEGORIES[c][0] for c in category])
    sigmas = np.array([CATEGORIES[c][1] for c in category])
    amount = np.round(rng.lognormal(mean=np.log(means), sigma=sigmas), 2)
    amount = np.clip(amount, 1.0, 4_800.0)

    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_P)
    # ATM only makes sense for cash-like use; keep it but amounts stay as drawn.
    day_offset = rng.integers(0, DAYS, size=n)
    hour = np.where(
        rng.random(n) < 0.78,
        rng.integers(7, 23, size=n),
        rng.integers(0, 24, size=n),
    )
    minute = rng.integers(0, 60, size=n)
    ts = START + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")

    is_foreign = (rng.random(n) < 0.08).astype(int)
    device_new = (rng.random(n) < 0.12).astype(int)
    velocity = rng.poisson(0.5, size=n)

    is_online = (channel == "online").astype(int)
    is_night = ((hour >= 0) & (hour <= 5)).astype(int)
    risky_combo = np.array(
        [1 if (c in RISKY_ONLINE_CATEGORIES and ch == "online") else 0 for c, ch in zip(category, channel)]
    )
    # A burst process for a minority of accounts drives high velocity.
    burst = rng.random(n) < 0.05
    velocity = np.where(burst, velocity + rng.integers(3, 7, size=n), velocity)

    risk = (
        -5.10
        + 1.50 * is_foreign
        + 1.10 * is_online
        + 1.10 * is_night
        + 0.90 * device_new
        + 0.50 * velocity
        + 0.65 * np.log1p(amount / 100.0)
        + 1.40 * risky_combo
        + 1.20 * burst.astype(int)
    )
    fraud = (rng.random(n) < _sigmoid(risk)).astype(int)

    df = pd.DataFrame(
        {
            "txn_id": [f"T{100000 + i}" for i in range(n)],
            "customer_id": [f"C{int(c):04d}" for c in rng.integers(1, 1501, size=n)],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_gbp": amount,
            "currency": "GBP",
            "channel": channel,
            "merchant_category": category,
            "merchant_id": [f"M{int(m):03d}" for m in rng.integers(1, 301, size=n)],
            "is_foreign": is_foreign,
            "device_new": device_new,
            "velocity_10min": velocity.astype(int),
            "is_fraud": fraud,
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def inject_dirty(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject exactly 75 dirty rows: 40 duplicates, 12 negative amounts,
    5 zero amounts, 10 missing customers, 8 invalid channels."""
    dirty = []

    dup_idx = rng.choice(len(df), size=40, replace=False)
    dirty.append(df.iloc[dup_idx].copy())

    base = df.iloc[rng.choice(len(df), size=35, replace=False)].copy().reset_index(drop=True)
    neg = base.iloc[:12].copy()
    neg["txn_id"] = [f"D{900000 + i}" for i in range(12)]
    neg["amount_gbp"] = -np.abs(neg["amount_gbp"].values)
    dirty.append(neg)

    zero = base.iloc[12:17].copy()
    zero["txn_id"] = [f"D{900100 + i}" for i in range(5)]
    zero["amount_gbp"] = 0.0
    dirty.append(zero)

    missing = base.iloc[17:27].copy()
    missing["txn_id"] = [f"D{900200 + i}" for i in range(10)]
    missing["customer_id"] = ""
    dirty.append(missing)

    bad_channel = base.iloc[27:35].copy()
    bad_channel["txn_id"] = [f"D{900300 + i}" for i in range(8)]
    bad_channel["channel"] = "carrier_pigeon"
    dirty.append(bad_channel)

    raw = pd.concat([df] + dirty, ignore_index=True)
    return raw.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def generate_raw(n: int = N_CLEAN) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    clean = generate_clean(rng, n=n)
    return inject_dirty(clean, rng)


if __name__ == "__main__":
    raw = generate_raw()
    print(f"raw rows: {len(raw)}")
    print(f"fraud rate (clean rows carry labels): {raw['is_fraud'].mean():.4f}")
