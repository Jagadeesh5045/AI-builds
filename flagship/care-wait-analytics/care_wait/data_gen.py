"""Synthetic elective referral data generator.

All data is synthetic. No real patient records are used anywhere in this
project. The generator is seeded, so every run produces the same dataset
and the same published numbers.
"""
import numpy as np
import pandas as pd

SPECIALTIES = {
    "Orthopaedics": 140,
    "Ophthalmology": 112,
    "ENT": 120,
    "Dermatology": 95,
    "Cardiology": 80,
    "General Surgery": 90,
}
AGE_BANDS = ["0-17", "18-39", "40-59", "60-74", "75+"]
AGE_WEIGHTS = [0.10, 0.24, 0.28, 0.23, 0.15]


def generate_clean(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    specs = list(SPECIALTIES)
    spec = rng.choice(specs, size=n, p=[0.22, 0.20, 0.16, 0.16, 0.13, 0.13])
    base = np.array([SPECIALTIES[s] for s in spec], dtype=float)
    priority = rng.choice(["routine", "urgent"], size=n, p=[0.78, 0.22])
    imd = rng.integers(1, 6, size=n)  # deprivation quintile, 5 = most deprived
    age_band = rng.choice(AGE_BANDS, size=n, p=AGE_WEIGHTS)
    wait = rng.normal(loc=base, scale=base * 0.32)
    wait = wait * np.where(priority == "urgent", 0.55, 1.0)
    wait = wait * (1.0 + (imd - 3) * 0.06)
    wait = wait * np.where(age_band == "75+", 1.05, 1.0)
    wait_days = np.clip(np.round(wait), 5, 400).astype(int)
    start = pd.Timestamp("2026-01-01")
    referral_date = start + pd.to_timedelta(rng.integers(0, 181, size=n), unit="D")
    df = pd.DataFrame(
        {
            "referral_id": [f"REF-{i:05d}" for i in range(1, n + 1)],
            "referral_date": referral_date.strftime("%Y-%m-%d"),
            "specialty": spec,
            "priority": priority,
            "age_band": age_band,
            "imd_quintile": imd.astype(int),
            "wait_days": wait_days,
        }
    )
    df["treatment_date"] = (
        pd.to_datetime(df["referral_date"]) + pd.to_timedelta(df["wait_days"], unit="D")
    ).dt.strftime("%Y-%m-%d")
    df["breach_18wk"] = (df["wait_days"] > 126).astype(int)
    return df


def generate_raw(n=6000, seed=42):
    """Clean data plus 75 deliberately dirty rows, to exercise the quality layer."""
    clean = generate_clean(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    dirty = []
    dup_idx = rng.choice(len(clean), size=30, replace=False)
    dirty.append(clean.iloc[dup_idx].copy())  # duplicate referral_ids
    missing = clean.iloc[:20].copy()
    missing["referral_id"] = [f"DIRTY-M{i:02d}" for i in range(20)]
    missing["specialty"] = None
    dirty.append(missing)
    negative = clean.iloc[20:35].copy()
    negative["referral_id"] = [f"DIRTY-N{i:02d}" for i in range(15)]
    negative["wait_days"] = -5
    dirty.append(negative)
    bad_age = clean.iloc[35:45].copy()
    bad_age["referral_id"] = [f"DIRTY-A{i:02d}" for i in range(10)]
    bad_age["age_band"] = "999"
    dirty.append(bad_age)
    raw = pd.concat([clean] + dirty, ignore_index=True)
    return raw.sample(frac=1.0, random_state=seed).reset_index(drop=True)
