#!/usr/bin/env python3
"""Card Ledger synthetic data generator for YugabyteDB (YSQL). Seed 42.

Adapted from Project 1 (project1-postgres/generate_data.py): same generator
logic, same table and column naming, same seed, scaled to 50,000 transactions
and with NO injected mismatches, so the reconciliation pack is expected to
find zero exceptions and the ledger is expected to balance exactly.

Loads with psycopg (v3) COPY where the server accepts it, and falls back to
batched multi-row INSERTs otherwise. The path actually used is printed.
Connection from YB_DATABASE_URL env var, default
postgresql://yugabyte@127.0.0.1:5433/card_ledger
"""
import json
import os
import random
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import psycopg

SEED = 42
N_MEMBERS = 2000
N_MERCHANTS = 150
N_TRANSACTIONS = 50_000
N_DECLINED_AUTHS = 200
FEE_RATE = Decimal("0.015")
CENT = Decimal("0.01")

FIRST = ["Amelia", "Oliver", "George", "Isla", "Harry", "Grace", "Noah", "Freya",
         "Leo", "Ivy", "Jack", "Ella", "Oscar", "Mia", "Alfie", "Evie", "Arthur",
         "Lily", "Charlie", "Sophie", "Priya", "Arjun", "Ananya", "Rohan", "Kavya",
         "Mei", "Chen", "Fatima", "Omar", "Zainab", "Lukas", "Emilia", "Jan", "Anya"]
LAST = ["Smith", "Jones", "Patel", "Brown", "Taylor", "Wilson", "Sharma", "Khan",
        "Evans", "Thomas", "Roberts", "Walker", "Wright", "Thompson", "White",
        "Hughes", "Edwards", "Green", "Hall", "Wood", "Harris", "Martin", "Clarke"]
BIZ_A = ["Tesco", "Sainsbury", "Asda", "Waitrose", "Costa", "Pret", "Greggs",
         "Boots", "Currys", "Argos", "Waterstones", "Next", "M&S", "Aldi", "Lidl"]
BIZ_B = ["Store", "Express", "Market", "Kitchen", "Pharmacy", "Books", "Travel",
         "Fuel", "Cafe", "Retail", "Direct", "Outlet", "Foods", "Goods"]
MCCS = ["5411", "5812", "5734", "5912", "5541", "5311", "4900", "7832"]
COUNTRIES = ["United Kingdom"] * 9 + ["Ireland"]


def money(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def signed(txn_type: str, amount: Decimal) -> Decimal:
    return amount if txn_type == "purchase" else -amount


def build_rows():
    rng = random.Random(SEED)

    members = []
    for mid in range(1, N_MEMBERS + 1):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        joined = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 900))
        members.append((mid, name, f"member{mid}@example.com", joined))

    cards = []
    card_member = {}
    cid = 0
    for mid, *_ in members:
        for _ in range(2 if rng.random() < 0.30 else 1):
            cid += 1
            cards.append((cid, mid, f"tok_{cid:08d}",
                          rng.choice(["VISA", "MASTERCARD", "AMEX"]),
                          "active" if rng.random() < 0.95 else "blocked",
                          date(2024, 1, 1) + timedelta(days=rng.randrange(0, 700))))
            card_member[cid] = mid

    merchants = []
    for m in range(1, N_MERCHANTS + 1):
        merchants.append((m, f"{rng.choice(BIZ_A)} {rng.choice(BIZ_B)} {m}",
                          rng.choice(MCCS), rng.choice(COUNTRIES)))

    start = date(2026, 7, 1)
    auths = []
    txns = []
    for tid in range(1, N_TRANSACTIONS + 1):
        card_id = rng.randrange(1, len(cards) + 1)
        merchant_id = rng.randrange(1, N_MERCHANTS + 1)
        amount = money(Decimal(str(round(rng.triangular(1.0, 500.0, 40.0), 2))))
        txn_type = "refund" if rng.random() < 0.05 else "purchase"
        tdate = start + timedelta(days=rng.randrange(0, 92))
        at = datetime.combine(tdate, time(rng.randrange(0, 24), rng.randrange(0, 60)))
        auths.append((tid, card_id, merchant_id, amount, "GBP", "approved", at))
        txns.append((tid, card_id, card_member[card_id], merchant_id, tid,
                     amount, "GBP", txn_type, "posted", tdate))

    declined = []
    for i in range(N_DECLINED_AUTHS):
        tdate = start + timedelta(days=rng.randrange(0, 92))
        declined.append((N_TRANSACTIONS + i + 1,
                         rng.randrange(1, len(cards) + 1),
                         rng.randrange(1, N_MERCHANTS + 1),
                         money(Decimal(str(round(rng.uniform(1, 300), 2)))),
                         "GBP", "declined",
                         datetime.combine(tdate, time(rng.randrange(0, 24),
                                                       rng.randrange(0, 60)))))

    ledger = []
    eid = 0
    for t in txns:
        amt, ttype, tdate = t[5], t[7], t[9]
        pair = (("1100", amt, Decimal("0.00")), ("2100", Decimal("0.00"), amt)) \
            if ttype == "purchase" else \
               (("1100", Decimal("0.00"), amt), ("2100", amt, Decimal("0.00")))
        for acct, dr, cr in pair:
            eid += 1
            ledger.append((eid, t[0], acct, dr, cr, tdate))

    activity = {}
    for t in txns:
        key = (t[2], t[9].year, t[9].month)
        activity[key] = activity.get(key, Decimal("0.00")) + signed(t[7], t[5])
    statements = []
    sid = 0
    running = {}
    for (mid, y, m) in sorted(activity):
        p_start = date(y, m, 1)
        p_end = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
        opening = running.get(mid, Decimal("0.00"))
        closing = opening + activity[(mid, y, m)]
        running[mid] = closing
        sid += 1
        statements.append((sid, mid, p_start, p_end, opening, closing,
                           p_end + timedelta(days=21)))

    settle = {}
    for t in txns:
        key = (t[3], t[9])
        g, purch, c = settle.get(key, (Decimal("0.00"), Decimal("0.00"), 0))
        settle[key] = (g + signed(t[7], t[5]),
                       purch + (t[5] if t[7] == "purchase" else Decimal("0.00")),
                       c + 1)
    settlements = []
    for i, ((mm, sdate), (gross, purch, cnt)) in enumerate(sorted(settle.items()), 1):
        fees = money(purch * FEE_RATE)
        settlements.append((i, mm, sdate, gross, fees, gross - fees, cnt))

    loads = [
        ("card_members", "(member_id, full_name, email, joined_date)", members),
        ("cards", "(card_id, member_id, card_number_token, network, status, issued_date)", cards),
        ("merchants", "(merchant_id, merchant_name, mcc, country)", merchants),
        ("authorisations", "(auth_id, card_id, merchant_id, amount, currency, auth_status, authorised_at)",
         list(auths) + declined),
        ("transactions", "(txn_id, card_id, member_id, merchant_id, auth_id, amount, currency, txn_type, status, transaction_date)",
         txns),
        ("ledger_entries", "(entry_id, txn_id, account_code, debit, credit, entry_date)", ledger),
        ("statements", "(statement_id, member_id, period_start, period_end, opening_balance, closing_balance, due_date)", statements),
        ("settlements", "(settlement_id, merchant_id, settlement_date, gross_amount, fees, net_amount, txn_count)",
         settlements),
    ]
    return loads


def load_copy(cur, loads):
    for tbl, cols, rows in loads:
        with cur.copy(f"COPY {tbl} {cols} FROM STDIN") as copy:
            for r in rows:
                copy.write_row(r)
        print(f"  COPY {tbl}: {len(rows)} rows")


def load_inserts(cur, loads, batch=500):
    for tbl, cols, rows in loads:
        ncols = cols.count(",") + 1
        placeholder = "(" + ",".join(["%s"] * ncols) + ")"
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            sql = f"INSERT INTO {tbl} {cols} VALUES " + ",".join([placeholder] * len(chunk))
            cur.execute(sql, [v for r in chunk for v in r])
        print(f"  INSERT {tbl}: {len(rows)} rows")


def main() -> None:
    dsn = os.environ.get("YB_DATABASE_URL",
                         "postgresql://yugabyte@127.0.0.1:5433/card_ledger")
    loads = build_rows()
    counts = {tbl: len(rows) for tbl, _, rows in loads}
    path_used = "copy"
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for tbl, _, _ in reversed(loads):
                cur.execute(f"DELETE FROM {tbl}")
            try:
                load_copy(cur, loads)
            except Exception as exc:
                path_used = "batched_inserts"
                print(f"COPY path failed ({type(exc).__name__}: {exc}); "
                      f"falling back to batched INSERTs", file=sys.stderr)
                for tbl, _, _ in reversed(loads):
                    cur.execute(f"DELETE FROM {tbl}")
                load_inserts(cur, loads)
    meta = {"seed": SEED, "load_path": path_used, "generated_counts": counts,
            "injected_mismatches": 0}
    Path(__file__).with_name("generated.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(counts, default=str))
    print(f"LOAD_PATH={path_used}")


if __name__ == "__main__":
    main()
