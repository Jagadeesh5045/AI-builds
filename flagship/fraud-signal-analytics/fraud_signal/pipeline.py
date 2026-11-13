"""Data-quality pipeline: 10 checks, quarantine, SQLite warehouse load.

Every raw row is tested against all 10 checks. Rows that fail any check go
to the quarantine table with the failed check names recorded; clean rows go
to the transactions table. Raw = clean + quarantined, always.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .generate import CATEGORY_NAMES, CHANNELS

VALID_CURRENCIES = {"GBP"}


def run_checks(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    checks = pd.DataFrame(index=df.index)
    checks["txn_id_present"] = df["txn_id"].astype(str).str.strip().ne("") & df["txn_id"].notna()
    checks["txn_id_unique"] = ~df.duplicated(subset=["txn_id"], keep="first")
    checks["customer_id_present"] = df["customer_id"].astype(str).str.strip().ne("") & df["customer_id"].notna()
    checks["amount_positive"] = df["amount_gbp"] > 0
    checks["amount_within_limit"] = df["amount_gbp"] <= 10_000
    checks["channel_valid"] = df["channel"].isin(CHANNELS)
    checks["category_valid"] = df["merchant_category"].isin(CATEGORY_NAMES)
    checks["timestamp_valid"] = ts.notna() & (ts >= "2026-08-01") & (ts < "2026-11-01")
    checks["currency_gbp"] = df["currency"].isin(VALID_CURRENCIES)
    checks["merchant_id_present"] = df["merchant_id"].astype(str).str.strip().ne("") & df["merchant_id"].notna()
    df["_failed_checks"] = checks.apply(lambda r: ",".join(checks.columns[~r]) if (~r).any() else "", axis=1)
    df["_passed"] = df["_failed_checks"].eq("")
    return df


def split_clean_quarantine(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    checked = run_checks(raw)
    clean = checked[checked["_passed"]].drop(columns=["_passed", "_failed_checks"]).reset_index(drop=True)
    quarantine = checked[~checked["_passed"]].reset_index(drop=True)
    quarantine = quarantine.rename(columns={"_failed_checks": "failed_checks"})
    quarantine = quarantine.drop(columns=["_passed"])
    return clean, quarantine


def load_warehouse(clean: pd.DataFrame, quarantine: pd.DataFrame, db_path: str | Path) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        clean.to_sql("transactions", conn, index=False)
        quarantine.to_sql("quarantine", conn, index=False)
        conn.execute("CREATE INDEX idx_txn_channel ON transactions(channel)")
        conn.execute("CREATE INDEX idx_txn_category ON transactions(merchant_category)")
        conn.execute("CREATE INDEX idx_txn_ts ON transactions(timestamp)")
    return db_path


def run_pipeline(raw: pd.DataFrame, db_path: str | Path) -> dict:
    clean, quarantine = split_clean_quarantine(raw)
    load_warehouse(clean, quarantine, db_path)
    by_check: dict[str, int] = {}
    for failed in quarantine["failed_checks"]:
        for name in str(failed).split(","):
            if name:
                by_check[name] = by_check.get(name, 0) + 1
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "quarantined_rows": int(len(quarantine)),
        "checks_run": 10,
        "quarantine_by_check": by_check,
        "clean": clean,
        "quarantine": quarantine,
    }
