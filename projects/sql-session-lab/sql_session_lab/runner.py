"""Query runner: executes the lab's .sql files against the events database.

Each SQL file is a single self-contained query (analytics SQL in the wild is
usually shared exactly like this: one file, one question). The runner keeps
the file as the source of truth, so the SQL printed in the repo is the SQL
that produced every number in the report.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
QUERIES = [
    "01_sessionize",
    "02_funnel",
    "03_daily_windows",
    "04_cohort_retention",
    "05_calendar_spine",
]


def load_sql(name: str) -> str:
    return (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")


def run_query(con: sqlite3.Connection, name: str) -> tuple[list[str], list[tuple]]:
    cur = con.execute(load_sql(name))
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def run_all(db_path: str, out_dir: str) -> dict[str, tuple[list[str], list[tuple]]]:
    """Run every query, write one CSV per query, return {name: (cols, rows)}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[list[str], list[tuple]]] = {}
    con = sqlite3.connect(db_path)
    try:
        for name in QUERIES:
            cols, rows = run_query(con, name)
            results[name] = (cols, rows)
            with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                writer.writerows(rows)
    finally:
        con.close()
    return results
