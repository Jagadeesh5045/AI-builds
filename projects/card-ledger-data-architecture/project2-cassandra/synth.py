"""Seeded synthetic card-ledger data, GBP only, no real people.

Same style as the relational Project 1 data: deterministic with seed 42
so Cassandra, DuckDB and the tests all see the identical dataset.
IDs are uuid5 values derived from row indexes, so they are stable
across runs and engines.
"""
import random
import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal

NS = uuid.UUID("12345678-1234-5678-1234-567812345678")
FIRST = ["Amelia", "Oliver", "Priya", "George", "Fatima", "Harry", "Meera",
         "Jack", "Ananya", "Thomas", "Sofia", "Arjun", "Emily", "Rohan", "Lucy"]
LAST = ["Sharma", "Patel", "Jones", "Kaur", "Smith", "Ahmed", "Brown",
        "Reddy", "Wilson", "Nair", "Taylor", "Khan", "Davies", "Iyer", "Evans"]
MERCHANTS = ["Tesco", "Sainsburys", "Shell", "BP", "Amazon UK", "Trainline",
             "Costa", "Greggs", "Boots", "Argos", "IKEA UK", "Uber UK",
             "Deliveroo", "ASOS", "M&S", "Aldi", "Lidl", "Currys", "Halfords",
             "Waterstones"]
MCCS = ["5411", "5411", "5541", "5541", "5969", "4112", "5814", "5814",
        "5912", "5969", "5712", "4121", "5812", "5651", "5411", "5411",
        "5411", "5732", "5942", "5942"]


def _uid(kind, idx):
    return uuid.uuid5(NS, f"{kind}-{idx}")


def generate(n_txns=50000, seed=42, n_members=1500, n_merchants=250):
    rng = random.Random(seed)
    members = []
    for i in range(n_members):
        members.append({
            "member_id": _uid("member", i),
            "full_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "email": f"member{i}@example.invalid",
            "joined_date": date(2022, 1, 1) + timedelta(days=rng.randrange(900)),
        })
    cards = []
    card_owner = {}
    ci = 0
    for m in members:
        for _ in range(1 + (1 if rng.random() < 0.35 else 0)):
            cards.append({
                "card_id": _uid("card", ci),
                "member_id": m["member_id"],
                "card_number_token": f"tok_{ci:06d}",
                "network": rng.choice(["VISA", "MASTERCARD"]),
                "status": "ACTIVE",
                "issued_date": date(2023, 1, 1) + timedelta(days=rng.randrange(700)),
            })
            card_owner[_uid("card", ci)] = m["member_id"]
            ci += 1
    merchants = []
    for i in range(n_merchants):
        k = i % len(MERCHANTS)
        merchants.append({
            "merchant_id": _uid("merchant", i),
            "merchant_name": MERCHANTS[k],
            "mcc": MCCS[k],
            "country": "GB",
        })
    start = datetime(2025, 1, 1)
    span_days = 181  # Jan to Jun 2025
    txns = []
    for i in range(n_txns):
        card = rng.choice(cards)
        merch = rng.choice(merchants)
        when = start + timedelta(days=rng.randrange(span_days),
                                 seconds=rng.randrange(86400))
        ttype = rng.choices(["PURCHASE", "PAYMENT", "REFUND", "FEE"],
                            weights=[86, 8, 4, 2])[0]
        amt = Decimal(rng.randrange(199, 45000)) / Decimal(100)
        txns.append({
            "txn_id": _uid("txn", i),
            "card_id": card["card_id"],
            "member_id": card_owner[card["card_id"]],
            "merchant_id": merch["merchant_id"],
            "merchant_name": merch["merchant_name"],
            "card_number_token": card["card_number_token"],
            "auth_id": _uid("auth", i),
            "amount": amt,
            "currency": "GBP",
            "txn_type": ttype,
            "status": rng.choices(["POSTED", "PENDING"], weights=[95, 5])[0],
            "transaction_date": when,
        })
    txns.sort(key=lambda t: t["transaction_date"])
    # Statements: one per member per month, balances from a simple walk:
    # closing = opening - purchases - fees + payments + refunds.
    signed = {}
    for t in txns:
        key = (t["member_id"], t["transaction_date"].strftime("%Y-%m"))
        delta = -t["amount"] if t["txn_type"] in ("PURCHASE", "FEE") else t["amount"]
        signed[key] = signed.get(key, Decimal(0)) + delta
    statements = []
    si = 0
    for m in members:
        opening = Decimal(rng.randrange(0, 50000)) / Decimal(100)
        for month in range(1, 7):
            key = (m["member_id"], f"2025-{month:02d}")
            closing = opening + signed.get(key, Decimal(0))
            statements.append({
                "statement_id": _uid("statement", si),
                "member_id": m["member_id"],
                "period_start": date(2025, month, 1),
                "period_end": date(2025, month, 28),
                "opening_balance": opening,
                "closing_balance": closing,
                "due_date": date(2025, month, 28) + timedelta(days=21)
                if month < 12 else date(2026, 1, 18),
            })
            opening = closing
            si += 1
    # Settlements: one per merchant per active day, from POSTED purchases.
    agg = {}
    for t in txns:
        if t["status"] == "POSTED" and t["txn_type"] == "PURCHASE":
            key = (t["merchant_id"], t["transaction_date"].date())
            a = agg.setdefault(key, [Decimal(0), 0])
            a[0] += t["amount"]
            a[1] += 1
    settlements = []
    for j, ((mid, day), (gross, count)) in enumerate(sorted(agg.items())):
        fees = (gross * Decimal("0.015")).quantize(Decimal("0.01"))
        settlements.append({
            "settlement_id": _uid("settlement", j),
            "merchant_id": mid,
            "settlement_date": day,
            "gross_amount": gross,
            "fees": fees,
            "net_amount": gross - fees,
            "txn_count": count,
        })
    # Authorisation events: derived sample, includes declines.
    events = []
    for i, t in enumerate(txns[:8000]):
        declined = rng.random() < 0.07
        events.append({
            "event_id": _uid("event", i),
            "card_id": t["card_id"],
            "member_id": t["member_id"],
            "merchant_id": t["merchant_id"],
            "event_time": t["transaction_date"] - timedelta(minutes=2),
            "amount": t["amount"],
            "currency": "GBP",
            "outcome": "DECLINED" if declined else "APPROVED",
            "reason": "INSUFFICIENT_FUNDS" if declined else "",
        })
    return {"members": members, "cards": cards, "merchants": merchants,
            "transactions": txns, "statements": statements,
            "settlements": settlements, "auth_events": events}


if __name__ == "__main__":
    d = generate()
    for k, v in d.items():
        print(k, len(v))
