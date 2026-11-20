"""ETL: validate the raw discharge feed and load a SQLite warehouse.

Every record is checked field by field. A record that fails any check is
never silently dropped and never loaded: it goes to the ``quarantine``
table with a human-readable reason, so data-quality problems stay
visible and countable. Only records that pass every check reach the
``discharges`` warehouse table that analytics and modelling read from.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .generate import AGE_BANDS, CONDITIONS, DESTINATIONS

_WAREHOUSE_DDL = """
CREATE TABLE IF NOT EXISTS discharges (
    discharge_id TEXT PRIMARY KEY,
    age_band TEXT NOT NULL,
    sex TEXT NOT NULL,
    primary_condition TEXT NOT NULL,
    admission_type TEXT NOT NULL,
    length_of_stay_days INTEGER NOT NULL,
    prior_admissions INTEGER NOT NULL,
    comorbidity_count INTEGER NOT NULL,
    discharge_destination TEXT NOT NULL,
    deprivation_quintile INTEGER NOT NULL,
    readmitted_30d INTEGER NOT NULL
)
"""

_QUARANTINE_DDL = """
CREATE TABLE IF NOT EXISTS quarantine (
    discharge_id TEXT,
    age_band TEXT,
    sex TEXT,
    primary_condition TEXT,
    admission_type TEXT,
    length_of_stay_days INTEGER,
    prior_admissions INTEGER,
    comorbidity_count INTEGER,
    discharge_destination TEXT,
    deprivation_quintile INTEGER,
    readmitted_30d INTEGER,
    quarantine_reason TEXT NOT NULL
)
"""


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or value == ""


def _as_int(value):
    """Return ``value`` as an int, or None if it is not a clean integer."""
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def validate_row(row: dict, seen_ids: set) -> str | None:
    """Return a quarantine reason for ``row``, or None if it is clean."""
    for field in ("discharge_id", "age_band", "sex", "primary_condition",
                  "admission_type", "discharge_destination"):
        if _is_missing(row.get(field)):
            return f"missing value: {field}"
    for field in ("length_of_stay_days", "prior_admissions", "comorbidity_count",
                  "deprivation_quintile", "readmitted_30d"):
        if _is_missing(row.get(field)):
            return f"missing value: {field}"

    if row["discharge_id"] in seen_ids:
        return "duplicate discharge_id"
    if row["age_band"] not in AGE_BANDS:
        return f"invalid age_band: {row['age_band']}"
    if row["sex"] not in ("F", "M"):
        return f"invalid sex code: {row['sex']}"
    if row["primary_condition"] not in CONDITIONS:
        return f"invalid primary_condition: {row['primary_condition']}"
    if row["admission_type"] not in ("emergency", "elective"):
        return f"invalid admission_type: {row['admission_type']}"
    if row["discharge_destination"] not in DESTINATIONS:
        return f"invalid discharge_destination: {row['discharge_destination']}"

    los = _as_int(row["length_of_stay_days"])
    if los is None or not 1 <= los <= 60:
        return f"length_of_stay_days out of range: {row['length_of_stay_days']}"
    prior = _as_int(row["prior_admissions"])
    if prior is None or not 0 <= prior <= 10:
        return f"prior_admissions out of range: {row['prior_admissions']}"
    comorbidity = _as_int(row["comorbidity_count"])
    if comorbidity is None or not 0 <= comorbidity <= 15:
        return f"comorbidity_count out of range: {row['comorbidity_count']}"
    deprivation = _as_int(row["deprivation_quintile"])
    if deprivation is None or not 1 <= deprivation <= 5:
        return f"deprivation_quintile out of range: {row['deprivation_quintile']}"
    readmitted = _as_int(row["readmitted_30d"])
    if readmitted not in (0, 1):
        return f"invalid readmitted_30d flag: {row['readmitted_30d']}"
    return None


def run_etl(raw: pd.DataFrame, db_path: str | Path) -> dict:
    """Validate ``raw``, load clean rows, quarantine the rest.

    Returns counts for the raw feed, the warehouse and quarantine.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    clean_rows, quarantined_rows = [], []
    seen_ids: set = set()
    for record in raw.to_dict("records"):
        reason = validate_row(record, seen_ids)
        if reason is None:
            seen_ids.add(record["discharge_id"])
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
        clean.to_sql("discharges", conn, if_exists="append", index=False)
        if not quarantined.empty:
            quarantined.to_sql("quarantine", conn, if_exists="append", index=False)
        conn.commit()
        warehouse_count = conn.execute("SELECT COUNT(*) FROM discharges").fetchone()[0]
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
