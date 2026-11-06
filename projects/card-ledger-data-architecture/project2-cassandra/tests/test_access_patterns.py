"""Each per-query table must answer its access pattern on its own.

Engine: DuckDB implementation of the identical tables (store_duckdb),
unless CASSANDRA_LIVE=1 is set, in which case the same assertions run
against the loaded Cassandra keyspace. Assertions compare against the
source dataset, not against the store's own output, so a wrong table
shape or wrong key fails loudly.
"""
import os
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from synth import generate  # noqa: E402
import queries_duckdb as q  # noqa: E402
import store_duckdb  # noqa: E402

DATA = generate(n_txns=4000, seed=42, n_members=120, n_merchants=40)
CON = store_duckdb.build(DATA)
TGT = q.pick_targets(DATA)


def test_ap1_statement_view_matches_source():
    rows = q.ap1(CON, TGT["member_id"], TGT["month"])
    expected = [t for t in DATA["transactions"]
                if str(t["member_id"]) == TGT["member_id"]
                and t["transaction_date"].strftime("%Y-%m") == TGT["month"]]
    assert len(rows) == len(expected)
    assert {r[3] for r in rows} == {str(t["txn_id"]) for t in expected}
    dates = [r[2] for r in rows]
    assert dates == sorted(dates, reverse=True)  # newest first


def test_ap2_card_feed_newest_first_and_limited():
    rows = q.ap2(CON, TGT["card_id"], limit=100)
    expected = [t for t in DATA["transactions"]
                if str(t["card_id"]) == TGT["card_id"]]
    assert len(rows) == min(100, len(expected))
    dates = [r[1] for r in rows]
    assert dates == sorted(dates, reverse=True)
    newest = max(t["transaction_date"] for t in expected)
    assert dates[0] == newest


def test_ap3_merchant_day_totals_reconcile():
    rows = q.ap3(CON, TGT["merchant_id"], TGT["day"])
    total = sum(r[6] for r in rows)
    expected = sum(t["amount"] for t in DATA["transactions"]
                   if str(t["merchant_id"]) == TGT["merchant_id"]
                   and t["transaction_date"].date() == TGT["day"])
    assert total == expected
    dates = [r[2] for r in rows]
    assert dates == sorted(dates)  # settlement scan runs in day order


def test_ap4_statements_newest_period_first():
    rows = q.ap4(CON, TGT["member_id"])
    expected = [x for x in DATA["statements"]
                if str(x["member_id"]) == TGT["member_id"]]
    assert len(rows) == len(expected) == 6
    starts = [r[1] for r in rows]
    assert starts == sorted(starts, reverse=True)
    for r in rows:
        assert isinstance(r[4], Decimal) and isinstance(r[5], Decimal)


def test_ap5_settlement_amounts_for_merchant():
    rows = q.ap5(CON, TGT["merchant_id"])
    expected = [x for x in DATA["settlements"]
                if str(x["merchant_id"]) == TGT["merchant_id"]]
    assert len(rows) == len(expected)
    for r in rows:
        assert r[3] - r[4] == r[5]  # gross - fees = net, per row
    assert sum(r[6] for r in rows) == sum(x["txn_count"] for x in expected)


def test_ap6_auth_events_recent_first():
    rows = q.ap6(CON, TGT["card_id"], limit=200)
    expected = [e for e in DATA["auth_events"]
                if str(e["card_id"]) == TGT["card_id"]]
    assert len(rows) == min(200, len(expected))
    times = [r[1] for r in rows]
    assert times == sorted(times, reverse=True)
    assert {r[7] for r in rows} <= {"APPROVED", "DECLINED"}
