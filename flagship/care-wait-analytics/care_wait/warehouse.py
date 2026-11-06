"""SQLite warehouse layer. One fact table, loaded idempotently."""
import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS referrals (
    referral_id TEXT PRIMARY KEY,
    referral_date TEXT NOT NULL,
    specialty TEXT NOT NULL,
    priority TEXT NOT NULL,
    age_band TEXT NOT NULL,
    imd_quintile INTEGER NOT NULL,
    wait_days INTEGER NOT NULL,
    treatment_date TEXT,
    breach_18wk INTEGER NOT NULL
)
"""


def load_warehouse(clean, db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS referrals")
        conn.execute(SCHEMA)
        clean.to_sql("referrals", conn, if_exists="append", index=False)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM referrals").fetchone()[0]
    finally:
        conn.close()
    return int(count)


def read_warehouse(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql("SELECT * FROM referrals", conn)
    finally:
        conn.close()
