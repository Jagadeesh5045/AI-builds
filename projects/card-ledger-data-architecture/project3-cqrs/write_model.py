"""Write model for the CQRS slice.

Prefers PostgreSQL when a server is reachable, otherwise falls back to a
DuckDB file. Amounts are stored as integer pence so totals compare exactly.
"""
import os

PG_DSN = os.environ.get(
    "CQRS_PG_DSN", "host=127.0.0.1 dbname=cqrs_write user=postgres password=cqrs"
)

DDL_PG = """
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    auth_id TEXT,
    amount_pence BIGINT NOT NULL,
    currency TEXT NOT NULL,
    txn_type TEXT NOT NULL,
    status TEXT NOT NULL,
    transaction_date DATE NOT NULL
);
"""

DDL_DUCK = DDL_PG  # DuckDB accepts the same DDL used here


class WriteModel:
    def __init__(self, base_dir):
        self.engine = None
        self.conn = None
        self.base_dir = base_dir
        try:
            import psycopg2

            conn = psycopg2.connect(PG_DSN, connect_timeout=3)
            conn.autocommit = True
            conn.cursor().execute(DDL_PG)
            self.conn, self.engine = conn, "postgresql"
        except Exception:
            import duckdb

            self.conn = duckdb.connect(os.path.join(base_dir, "write_model.duckdb"))
            self.conn.execute(DDL_DUCK)
            self.engine = "duckdb"

    def insert_transactions(self, rows):
        cols = (
            "txn_id, card_id, member_id, merchant_id, auth_id, amount_pence,"
            " currency, txn_type, status, transaction_date"
        )
        if self.engine == "postgresql":
            from psycopg2.extras import execute_values

            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO transactions ({cols}) VALUES %s"
                    " ON CONFLICT (txn_id) DO NOTHING",
                    rows,
                )
        else:
            # DuckDB row-by-row executemany is very slow at 20k rows; COPY a
            # temp CSV instead, then drop any duplicate txn_ids.
            import csv
            import tempfile

            with tempfile.NamedTemporaryFile(
                "w", newline="", suffix=".csv", dir=self.base_dir, delete=False
            ) as f:
                csv.writer(f).writerows(rows)
                tmp = f.name
            self.conn.execute(
                f"CREATE OR REPLACE TEMP TABLE stage AS SELECT * FROM transactions LIMIT 0"
            )
            self.conn.execute(f"COPY stage FROM '{tmp}' (HEADER false)")
            self.conn.execute(
                "INSERT INTO transactions SELECT * FROM stage"
                " WHERE txn_id NOT IN (SELECT txn_id FROM transactions)"
            )
            os.remove(tmp)

    def totals_by_member_period(self):
        q = (
            "SELECT member_id, date_trunc('month', transaction_date),"
            " SUM(amount_pence), COUNT(*) FROM transactions"
            " GROUP BY 1, 2 ORDER BY 1, 2"
        )
        if self.engine == "postgresql":
            with self.conn.cursor() as cur:
                cur.execute(q)
                return [(r[0], r[1].strftime("%Y-%m"), int(r[2]), int(r[3])) for r in cur.fetchall()]
        return [
            (r[0], r[1].strftime("%Y-%m"), int(r[2]), int(r[3]))
            for r in self.conn.execute(q).fetchall()
        ]

    def row_count(self):
        if self.engine == "postgresql":
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM transactions")
                return int(cur.fetchone()[0])
        return int(self.conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])

    def total_pence(self):
        if self.engine == "postgresql":
            with self.conn.cursor() as cur:
                cur.execute("SELECT COALESCE(SUM(amount_pence),0) FROM transactions")
                return int(cur.fetchone()[0])
        return int(
            self.conn.execute("SELECT COALESCE(SUM(amount_pence),0) FROM transactions").fetchone()[0]
        )
