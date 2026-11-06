"""SQL-source ingestor (DuckDB) plus Snowflake-dialect validation (sqlglot).

DuckDB runs locally and holds the telemetry table. The KPI queries are kept in
Snowflake dialect in queries_snowflake.sql; here they are transpiled with
sqlglot (snowflake -> duckdb) and the transpiled SQL is what actually executes.
That proves dialect portability. It does not prove anything about a live
Snowflake account, and the README says so.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd
import sqlglot

HERE = Path(__file__).parent


def load_into_duckdb(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> None:
    con.register("telemetry_stage", df)
    con.execute("CREATE OR REPLACE TABLE telemetry_daily AS SELECT * FROM telemetry_stage")


def transpile_snowflake_queries(sql_path: str | Path = HERE / "queries_snowflake.sql") -> dict[str, str]:
    """Split the Snowflake SQL file on '-- name: x' markers and transpile to DuckDB."""
    raw = Path(sql_path).read_text()
    blocks = re.split(r"--\s*name:\s*", raw)
    named: dict[str, str] = {}
    for block in blocks[1:]:
        name, _, sql = block.partition("\n")
        sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--")).strip()
        if sql:
            named[name.strip()] = sqlglot.transpile(sql, read="snowflake", write="duckdb")[0]
    return named


def run_kpi_queries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(":memory:")
    load_into_duckdb(df, con)
    queries = transpile_snowflake_queries()
    results = {name: con.execute(sql).df() for name, sql in queries.items()}
    con.close()
    return results


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE.parent))
    from ingestion.generate_telemetry import generate_telemetry

    frame, _ = generate_telemetry()
    for name, result in run_kpi_queries(frame).items():
        print(f"--- {name} ({len(result)} rows) ---")
        print(result.head(3).to_string(index=False))
