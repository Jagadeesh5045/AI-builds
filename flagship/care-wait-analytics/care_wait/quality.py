"""Data quality checks. Rows that fail any check are quarantined, never
silently dropped: the caller gets the clean frame, the quarantine frame
and a per-check failure count."""
import pandas as pd

from .data_gen import AGE_BANDS, SPECIALTIES

CHECKS = [
    "missing_referral_id",
    "duplicate_referral_id",
    "missing_specialty",
    "unknown_specialty",
    "invalid_priority",
    "invalid_age_band",
    "invalid_imd_quintile",
    "negative_wait",
    "wait_over_400",
    "bad_referral_date",
]


def run_quality_checks(raw):
    df = raw.copy()
    failed = pd.Series(False, index=df.index)
    counts = {}
    checks = {
        "missing_referral_id": df["referral_id"].isna() | (df["referral_id"] == ""),
        "duplicate_referral_id": df.duplicated(subset=["referral_id"], keep="first"),
        "missing_specialty": df["specialty"].isna() | (df["specialty"] == ""),
        "unknown_specialty": ~df["specialty"].isin(list(SPECIALTIES)) & df["specialty"].notna(),
        "invalid_priority": ~df["priority"].isin(["routine", "urgent"]),
        "invalid_age_band": ~df["age_band"].isin(AGE_BANDS),
        "invalid_imd_quintile": ~df["imd_quintile"].isin([1, 2, 3, 4, 5]),
        "negative_wait": df["wait_days"] < 0,
        "wait_over_400": df["wait_days"] > 400,
        "bad_referral_date": pd.to_datetime(df["referral_date"], errors="coerce").isna(),
    }
    for name, mask in checks.items():
        mask = mask.fillna(False)
        counts[name] = int(mask.sum())
        failed = failed | mask
    clean = df[~failed].reset_index(drop=True)
    quarantine = df[failed].reset_index(drop=True)
    return clean, quarantine, counts
