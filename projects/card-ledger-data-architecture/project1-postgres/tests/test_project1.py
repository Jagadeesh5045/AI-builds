"""Card Ledger Project 1 tests. Require a loaded PostgreSQL database.

Run: DATABASE_URL=postgresql://cardledger:cardledger@localhost:5432/cardledger \
     pytest project1-postgres/tests -v
The reconciliation test runs queries/reconciliation.sql through psql and then
compares the exceptions table with injected.json written by generate_data.py.
"""
import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest

BASE = Path(__file__).resolve().parent.parent
DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://cardledger:cardledger@localhost:5432/cardledger")


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(DSN) as c:
        yield c


@pytest.fixture(scope="module")
def injected():
    return json.loads((BASE / "injected.json").read_text())


def scalar(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def test_row_counts(conn):
    assert scalar(conn, "SELECT COUNT(*) FROM card_members") == 2000
    assert scalar(conn, "SELECT COUNT(*) FROM merchants") == 150
    assert scalar(conn, "SELECT COUNT(*) FROM transactions") == 100000
    assert scalar(conn, "SELECT COUNT(*) FROM authorisations") == 100400
    assert scalar(conn, "SELECT COUNT(*) FROM statements") > 0
    assert scalar(conn, "SELECT COUNT(*) FROM settlements") > 0


def test_double_entry_balances(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT SUM(debit), SUM(credit) FROM ledger_entries")
        debits, credits = cur.fetchone()
    assert debits == credits
    assert debits > 0


def test_partition_row_total_matches_parent(conn):
    parts = scalar(conn, """
        SELECT COALESCE(SUM(c), 0) FROM (
            SELECT COUNT(*) AS c FROM transactions_2026_07
            UNION ALL SELECT COUNT(*) FROM transactions_2026_08
            UNION ALL SELECT COUNT(*) FROM transactions_2026_09
            UNION ALL SELECT COUNT(*) FROM transactions_default
        ) p""")
    assert parts == 100000


def test_reconciliation_finds_exactly_the_injected_mismatches(conn, injected):
    subprocess.run(
        ["psql", DSN, "-v", "ON_ERROR_STOP=1", "-q",
         "-f", str(BASE / "queries" / "reconciliation.sql")],
        check=True, capture_output=True, text=True)
    with conn.cursor() as cur:
        cur.execute("SELECT issue_type, COUNT(*), "
                    "array_agg(source_id ORDER BY source_id) "
                    "FROM reconciliation_exceptions GROUP BY issue_type")
        found = {r[0]: (r[1], list(r[2])) for r in cur.fetchall()}
    conn.rollback()
    assert sum(v[0] for v in found.values()) == injected["total"]
    assert found["auth_amount_mismatch"] == (
        injected["auth_amount_mismatch"]["count"],
        injected["auth_amount_mismatch"]["txn_ids"])
    assert found["missing_ledger_entries"] == (
        injected["missing_ledger_entries"]["count"],
        injected["missing_ledger_entries"]["txn_ids"])
    assert found["settlement_amount_mismatch"] == (
        injected["settlement_amount_mismatch"]["count"],
        injected["settlement_amount_mismatch"]["settlement_ids"])
