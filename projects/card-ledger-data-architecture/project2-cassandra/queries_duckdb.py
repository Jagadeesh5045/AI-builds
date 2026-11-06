"""Access pattern queries against the DuckDB per-query tables.

Each function mirrors one CQL query shape from the Cassandra design.
"""
from collections import Counter


def pick_targets(data):
    """Deterministic busy targets so benches and tests hit real data."""
    txns = data["transactions"]
    mm = Counter((str(t["member_id"]), t["transaction_date"].strftime("%Y-%m"))
                 for t in txns)
    (member_id, month), _ = mm.most_common(1)[0]
    by_card = Counter(str(t["card_id"]) for t in txns)
    card_id, _ = by_card.most_common(1)[0]
    md = Counter((str(t["merchant_id"]), t["transaction_date"].date())
                 for t in txns)
    (merchant_id, day), _ = md.most_common(1)[0]
    return {"member_id": member_id, "month": month, "card_id": card_id,
            "merchant_id": merchant_id, "day": day}


def ap1(con, member_id, month):
    return con.execute(
        "SELECT * FROM transactions_by_member_month "
        "WHERE member_id=? AND month_bucket=? "
        "ORDER BY transaction_date DESC", [member_id, month]).fetchall()


def ap2(con, card_id, limit=100):
    return con.execute(
        "SELECT * FROM transactions_by_card WHERE card_id=? "
        "ORDER BY transaction_date DESC LIMIT ?", [card_id, limit]).fetchall()


def ap3(con, merchant_id, day):
    return con.execute(
        "SELECT * FROM transactions_by_merchant_day "
        "WHERE merchant_id=? AND day_bucket=? "
        "ORDER BY transaction_date ASC", [merchant_id, day]).fetchall()


def ap4(con, member_id):
    return con.execute(
        "SELECT * FROM statements_by_member WHERE member_id=? "
        "ORDER BY period_start DESC", [member_id]).fetchall()


def ap5(con, merchant_id):
    return con.execute(
        "SELECT * FROM settlements_by_merchant WHERE merchant_id=? "
        "ORDER BY settlement_date DESC", [merchant_id]).fetchall()


def ap6(con, card_id, limit=200):
    return con.execute(
        "SELECT * FROM auth_events_by_card WHERE card_id=? "
        "ORDER BY event_time DESC LIMIT ?", [card_id, limit]).fetchall()
