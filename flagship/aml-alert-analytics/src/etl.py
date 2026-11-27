"""ETL: validate the raw alert feed and load a SQLite warehouse.

Every record is checked field by field. A record that fails any check
is never silently dropped and never loaded: it goes to the
``quarantine`` table with a human-readable reason, so data-quality
problems stay visible and countable. Only records that pass every
check reach the ``alerts`` warehouse table that analytics and
modelling read from.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .generate import ALERT_RULES, CHANNELS, RISK_TIERS

DISPOSITIONS = ["confirmed_suspicious", "false_positive"]

_WAREHOUSE_DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    alert_date TEXT NOT NULL,
    transaction_amount REAL NOT NULL,
    channel TEXT NOT NULL,
    alert_rule TEXT NOT NULL,
    customer_risk_tier TEXT NOT NULL,
    prior_alerts INTEGER NOT NULL,
    account_age_days INTEGER NOT NULL,
    confirmed_suspicious INTEGER NOT NULL,
    disposition TEXT NOT NULL
)
"""

_QUARANTINE_DDL = """
CREATE TABLE IF NOT EXISTS quarantine (
    alert_id TEXT,
    customer_id TEXT,
    alert_date TEXT,
    transaction_amount REAL,
    channel TEXT,
    alert_rule TEXT,
    customer_risk_tier TEXT,
    prior_alerts INTEGER,
    account_age_days INTEGER,
    confirmed_suspicious INTEGER,
    disposition TEXT,
    quarantine_reason TEXT NOT NULL
)
"""


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or value == ""


def _as_int(value):
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _valid_date(value) -> bool:
    if _is_missing(value) or not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def validate_row(row: dict, seen_ids: set) -> str | None:
    """Return a quarantine reason for ``row``, or None if it is clean."""
    for field in ("alert_id", "customer_id", "alert_date", "channel",
                  "alert_rule", "customer_risk_tier", "disposition"):
        if _is_missing(row.get(field)):
            return f"missing value: {field}"
    for field in ("transaction_amount", "prior_alerts",
                  "account_age_days", "confirmed_suspicious"):
        if _is_missing(row.get(field)):
            return f"missing value: {field}"

    if row["alert_id"] in seen_ids:
        return "duplicate alert_id"
    if not _valid_date(row["alert_date"]):
        return f"invalid alert_date: {row['alert_date']}"
    try:
        amount = float(row["transaction_amount"])
    except (TypeError, ValueError):
        return f"invalid transaction_amount: {row['transaction_amount']}"
    if not 0 < amount <= 1000000:
        return f"transaction_amount out of range: {row['transaction_amount']}"
    if row["channel"] not in CHANNELS:
        return f"invalid channel: {row['channel']}"
    if row["alert_rule"] not in ALERT_RULES:
        return f"invalid alert_rule: {row['alert_rule']}"
    if row["customer_risk_tier"] not in RISK_TIERS:
        return f"invalid customer_risk_tier: {row['customer_risk_tier']}"
    prior = _as_int(row["prior_alerts"])
    if prior is None or not 0 <= prior <= 20:
        return f"prior_alerts out of range: {row['prior_alerts']}"
    age = _as_int(row["account_age_days"])
    if age is None or not 0 <= age <= 10000:
        return f"account_age_days out of range: {row['account_age_days']}"
    confirmed = _as_int(row["confirmed_suspicious"])
    if confirmed not in (0, 1):
        return f"invalid confirmed_suspicious flag: {row['confirmed_suspicious']}"
    if row["disposition"] not in DISPOSITIONS:
        return f"invalid disposition: {row['disposition']}"
    expected = "confirmed_suspicious" if confirmed == 1 else "false_positive"
    if row["disposition"] != expected:
        return "disposition inconsistent with confirmed_suspicious flag"
    return None


def run_etl(raw: pd.DataFrame, db_path: str | Path) -> dict:
    """Validate ``raw``, load clean rows, quarantine the rest."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    clean_rows, quarantined_rows = [], []
    seen_ids: set = set()
    for record in raw.to_dict("records"):
        reason = validate_row(record, seen_ids)
        if reason is None:
            seen_ids.add(record["alert_id"])
            clean_rows.append(record)
        else:
            quarantined_rows.append({**record, "quarantine_reason": reason})

    clean = pd.DataFrame(clean_rows, columns=list(raw.columns))
    quarantined = pd.DataFrame(
        quarantined_rows, columns=list(raw.columns) + ["quarantine_reason"]
    )

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_WAREHOUSE_DDL)
        conn.execute(_QUARANTINE_DDL)
        clean.to_sql("alerts", conn, if_exists="append", index=False)
        if not quarantined.empty:
            quarantined.to_sql("quarantine", conn, if_exists="append", index=False)
        conn.commit()
        warehouse_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        quarantine_count = conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
    finally:
        conn.close()

    return {
        "raw_rows": int(len(raw)),
        "warehouse_rows": int(warehouse_count),
        "quarantined_rows": int(quarantine_count),
        "quarantine_reasons": (
            quarantined["quarantine_reason"].value_counts().to_dict()
            if not quarantined.empty else {}
        ),
    }
