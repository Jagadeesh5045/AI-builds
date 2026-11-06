"""Read model projector.

Consumes TransactionRecorded events into a DuckDB statement-enquiry read
model. Idempotent: processed event ids are stored, replays are dropped and
counted. Invalid events go to a dead letter table with the failure reason.
"""
import json
import os

import duckdb

READ_DDL = [
    """
    CREATE TABLE IF NOT EXISTS processed_events (event_id TEXT PRIMARY KEY)
    """,
    """
    CREATE TABLE IF NOT EXISTS statement_enquiry (
        member_id TEXT, period TEXT, txn_id TEXT, card_id TEXT,
        merchant_id TEXT, amount_pence BIGINT, currency TEXT,
        txn_type TEXT, status TEXT, transaction_date DATE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dead_letters (
        event_id TEXT, reason TEXT, payload TEXT
    )
    """,
]


def validate(event):
    if not isinstance(event, dict):
        return "event is not an object"
    for field in ("event_id", "txn_id", "member_id", "amount_pence", "currency", "transaction_date"):
        if event.get(field) in (None, ""):
            return f"missing field: {field}"
    if not isinstance(event["amount_pence"], int) or event["amount_pence"] <= 0:
        return "amount_pence must be a positive integer"
    if event["currency"] != "GBP":
        return f"unsupported currency: {event['currency']}"
    return None


class Projector:
    def __init__(self, base_dir):
        path = os.path.join(base_dir, "read_model.duckdb")
        if os.path.exists(path):
            os.remove(path)
        self.conn = duckdb.connect(path)
        for ddl in READ_DDL:
            self.conn.execute(ddl)
        self.seen = {
            r[0] for r in self.conn.execute("SELECT event_id FROM processed_events").fetchall()
        }
        self.applied = 0
        self.duplicates = 0
        self.dead = 0

    def apply(self, events):
        # Idempotency state lives in the processed_events table; the in-memory
        # set mirrors it so a 20k event stream does not need a query per event.
        new_ids, rows, dead_rows = [], [], []
        for e in events:
            eid = e.get("event_id") if isinstance(e, dict) else None
            if eid and eid in self.seen:
                self.duplicates += 1
                continue
            reason = validate(e)
            if reason:
                dead_rows.append([eid or "unknown", reason, json.dumps(e, default=str)])
                if eid:
                    self.seen.add(eid)
                    new_ids.append([eid])
                self.dead += 1
                continue
            self.seen.add(eid)
            new_ids.append([eid])
            rows.append(
                [
                    e["member_id"],
                    e["transaction_date"][:7],
                    e["txn_id"],
                    e["card_id"],
                    e["merchant_id"],
                    e["amount_pence"],
                    e["currency"],
                    e["txn_type"],
                    e["status"],
                    e["transaction_date"],
                ]
            )
            self.applied += 1
        # DuckDB row-by-row executemany is slow at this volume; stage via CSV.
        import csv
        import tempfile

        def bulk(table, data):
            with tempfile.NamedTemporaryFile(
                "w", newline="", suffix=".csv", delete=False
            ) as f:
                csv.writer(f).writerows(data)
                tmp = f.name
            self.conn.execute(f"COPY {table} FROM '{tmp}' (HEADER false)")
            os.remove(tmp)

        if new_ids:
            bulk("processed_events", new_ids)
        if rows:
            bulk("statement_enquiry", rows)
        if dead_rows:
            bulk("dead_letters", dead_rows)

    def totals_by_member_period(self):
        return [
            (r[0], r[1], int(r[2]), int(r[3]))
            for r in self.conn.execute(
                "SELECT member_id, period, SUM(amount_pence), COUNT(*)"
                " FROM statement_enquiry GROUP BY 1, 2 ORDER BY 1, 2"
            ).fetchall()
        ]

    def total_pence(self):
        return int(
            self.conn.execute(
                "SELECT COALESCE(SUM(amount_pence),0) FROM statement_enquiry"
            ).fetchone()[0]
        )
