#!/usr/bin/env python3
"""Benchmarks and reconciliation against the YugabyteDB card_ledger database.

Runs the same headline queries as Project 1 (queries/benchmarks.sql there):
Q1 member statement lookup (member 42, August 2026) and Q2 merchant daily
totals (merchant 7). Each is timed client-side over N runs and the median is
reported. Then the Project 1 reconciliation pack (three checks) is run, plus
aggregate ledger-vs-transaction totals. Everything printed here is written
to results.json verbatim; RESULTS.md quotes that file.
"""
import json
import os
import statistics
import time
from pathlib import Path

import psycopg

DSN = os.environ.get("YB_DATABASE_URL",
                     "postgresql://yugabyte@127.0.0.1:5433/card_ledger")
RUNS = 5

Q1 = """
SELECT txn_id, transaction_date, merchant_id, txn_type, amount
FROM transactions
WHERE member_id = 42
  AND transaction_date >= DATE '2026-08-01'
  AND transaction_date <  DATE '2026-09-01'
ORDER BY transaction_date
"""

Q2 = """
SELECT transaction_date,
       COUNT(*) AS txn_count,
       SUM(CASE WHEN txn_type = 'purchase' THEN amount ELSE -amount END)
           AS signed_total
FROM transactions
WHERE merchant_id = 7
GROUP BY transaction_date
ORDER BY transaction_date
"""

TABLES = ["card_members", "cards", "merchants", "authorisations", "transactions",
          "ledger_entries", "statements", "settlements", "reconciliation_exceptions"]


def timed(cur, sql, runs=RUNS):
    times = []
    rows = None
    for _ in range(runs):
        t0 = time.perf_counter()
        cur.execute(sql)
        rows = cur.fetchall()
        times.append(round((time.perf_counter() - t0) * 1000, 3))
    return {"runs_ms": times, "median_ms": statistics.median(times),
            "rows_returned": len(rows)}


def main() -> None:
    out = {}
    with psycopg.connect(DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            out["server_version"] = cur.fetchone()[0]

            out["row_counts"] = {}
            for tbl in TABLES[:-1]:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                out["row_counts"][tbl] = cur.fetchone()[0]

            out["q1_member_statement"] = timed(cur, Q1)
            out["q2_merchant_daily_totals"] = timed(cur, Q2)

            cur.execute("EXPLAIN ANALYZE " + Q1)
            out["q1_explain_analyze"] = [r[0] for r in cur.fetchall()]

            # Aggregate reconciliation: ledger vs transactions.
            cur.execute("SELECT SUM(debit), SUM(credit) FROM ledger_entries")
            debit, credit = cur.fetchone()
            cur.execute("""
                SELECT SUM(debit - credit) FROM ledger_entries
                WHERE account_code = '1100'""")
            ledger_signed_1100 = cur.fetchone()[0]
            cur.execute("""
                SELECT SUM(CASE WHEN txn_type = 'purchase' THEN amount
                                ELSE -amount END) FROM transactions""")
            txn_signed = cur.fetchone()[0]
            out["aggregate_reconciliation"] = {
                "ledger_total_debit": str(debit),
                "ledger_total_credit": str(credit),
                "debits_equal_credits": debit == credit,
                "ledger_signed_account_1100": str(ledger_signed_1100),
                "transactions_signed_total": str(txn_signed),
                "ledger_matches_transactions": ledger_signed_1100 == txn_signed,
            }

            # Full Project 1 reconciliation pack (same three checks).
            cur.execute("DELETE FROM reconciliation_exceptions")
            checks = {
                "auth_amount_mismatch": """
                    INSERT INTO reconciliation_exceptions
                        (source_table, source_id, issue_type, expected_amount, actual_amount)
                    SELECT 'transactions', t.txn_id, 'auth_amount_mismatch', a.amount, t.amount
                    FROM transactions t
                    JOIN authorisations a ON a.auth_id = t.auth_id
                    WHERE a.amount <> t.amount""",
                "missing_ledger_entries": """
                    INSERT INTO reconciliation_exceptions
                        (source_table, source_id, issue_type, expected_amount, actual_amount)
                    SELECT 'transactions', t.txn_id, 'missing_ledger_entries',
                           CASE WHEN t.txn_type = 'purchase' THEN t.amount ELSE -t.amount END,
                           COALESCE(le.ledger_signed, 0)
                    FROM transactions t
                    LEFT JOIN (
                        SELECT txn_id, SUM(debit - credit) AS ledger_signed
                        FROM ledger_entries WHERE account_code = '1100'
                        GROUP BY txn_id
                    ) le ON le.txn_id = t.txn_id
                    WHERE COALESCE(le.ledger_signed, 0)
                          <> CASE WHEN t.txn_type = 'purchase' THEN t.amount ELSE -t.amount END""",
                "settlement_amount_mismatch": """
                    INSERT INTO reconciliation_exceptions
                        (source_table, source_id, issue_type, expected_amount, actual_amount)
                    SELECT 'settlements', s.settlement_id, 'settlement_amount_mismatch',
                           tx.gross_from_txns, s.gross_amount
                    FROM settlements s
                    JOIN (
                        SELECT merchant_id, transaction_date,
                               SUM(CASE WHEN txn_type = 'purchase' THEN amount ELSE -amount END)
                                   AS gross_from_txns
                        FROM transactions GROUP BY merchant_id, transaction_date
                    ) tx ON tx.merchant_id = s.merchant_id
                        AND tx.transaction_date = s.settlement_date
                    WHERE s.gross_amount <> tx.gross_from_txns""",
            }
            out["reconciliation_pack"] = {}
            for name, sql in checks.items():
                cur.execute(sql)
                out["reconciliation_pack"][name] = cur.rowcount
            cur.execute("SELECT COUNT(*) FROM reconciliation_exceptions")
            out["reconciliation_pack"]["total_exceptions"] = cur.fetchone()[0]
            out["row_counts"]["reconciliation_exceptions"] = \
                out["reconciliation_pack"]["total_exceptions"]

    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
