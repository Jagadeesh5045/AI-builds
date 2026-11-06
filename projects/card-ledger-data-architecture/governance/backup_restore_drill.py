"""Backup and restore drill for the CQRS write database.

Uses pg_dump and a fresh PostgreSQL database when PostgreSQL is the engine
that ran. Otherwise drills the DuckDB write model: copies it through an
export and re-import into a fresh DuckDB file. Compares row counts and total
pence before and after, and times the restore. Labelled by the engine that
actually runs.
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "project3-cqrs",
    "data",
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "project3-cqrs"))
from write_model import PG_DSN, WriteModel  # noqa: E402


def dsn_with_db(dsn, dbname):
    if "dbname=" in dsn:
        import re

        return re.sub(r"dbname=\S+", f"dbname={dbname}", dsn)
    return dsn


def drill_postgres():
    env = dict(os.environ)
    dump_path = os.path.join(BASE, "write_backup.sql")
    t0 = time.time()
    subprocess.run(["pg_dump", PG_DSN, "-f", dump_path], check=True, env=env)
    import psycopg2

    admin = psycopg2.connect(dsn_with_db(PG_DSN, "postgres"))
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS cqrs_write_restored")
        cur.execute("CREATE DATABASE cqrs_write_restored")
    admin.close()
    subprocess.run(
        ["psql", dsn_with_db(PG_DSN, "cqrs_write_restored"), "-f", dump_path],
        check=True,
        env=env,
    )
    restore_seconds = time.time() - t0
    restored = WriteModel.__new__(WriteModel)
    conn = psycopg2.connect(dsn_with_db(PG_DSN, "cqrs_write_restored"))
    conn.autocommit = True
    restored.conn, restored.engine = conn, "postgresql"
    return restored, restore_seconds


def drill_duckdb():
    import duckdb

    src = os.path.join(BASE, "write_model.duckdb")
    export_dir = os.path.join(BASE, "backup_export")
    os.makedirs(export_dir, exist_ok=True)
    con = duckdb.connect(src, read_only=True)
    before_rows = int(con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
    before_total = int(
        con.execute("SELECT COALESCE(SUM(amount_pence),0) FROM transactions").fetchone()[0]
    )
    con.execute(f"EXPORT DATABASE '{export_dir}' (FORMAT PARQUET)")
    con.close()
    restored_path = os.path.join(BASE, "write_model_restored.duckdb")
    if os.path.exists(restored_path):
        os.remove(restored_path)
    t0 = time.time()
    con2 = duckdb.connect(restored_path)
    con2.execute(f"IMPORT DATABASE '{export_dir}'")
    restore_seconds = time.time() - t0
    after_rows = int(con2.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
    after_total = int(
        con2.execute("SELECT COALESCE(SUM(amount_pence),0) FROM transactions").fetchone()[0]
    )
    con2.close()
    return {
        "engine": "duckdb",
        "method": "EXPORT DATABASE (Parquet) then IMPORT DATABASE into a fresh file",
        "rows_before": before_rows,
        "rows_after": after_rows,
        "total_pence_before": before_total,
        "total_pence_after": after_total,
        "match": before_rows == after_rows and before_total == after_total,
        "restore_seconds": round(restore_seconds, 3),
    }


def main():
    result = None
    try:
        import psycopg2

        probe = psycopg2.connect(PG_DSN, connect_timeout=3)
        probe.close()
        restored, seconds = drill_postgres()
        src = WriteModel(BASE)
        result = {
            "engine": "postgresql",
            "method": "pg_dump then psql restore into a fresh database cqrs_write_restored",
            "rows_before": src.row_count(),
            "rows_after": restored.row_count(),
            "total_pence_before": src.total_pence(),
            "total_pence_after": restored.total_pence(),
            "restore_seconds": round(seconds, 3),
        }
        result["match"] = (
            result["rows_before"] == result["rows_after"]
            and result["total_pence_before"] == result["total_pence_after"]
        )
    except Exception:
        result = drill_duckdb()
    out = os.path.join(BASE, "restore_drill.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
