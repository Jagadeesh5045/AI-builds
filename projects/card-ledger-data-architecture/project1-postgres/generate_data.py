#!/usr/bin/env python3
"""Card Ledger synthetic data generator. Seed 42. Synthetic data only.

Loads about 100,000 transactions plus consistent members, cards, merchants,
authorisations, balanced double entry ledger entries, statements and
settlements. It deliberately injects exactly 12 reconciliation mismatches
(5 settlement amount, 4 missing ledger postings, 3 authorisation amount) and
writes the exact injected ids and counts to injected.json next to this file.

Requires: psycopg (v3). Connection from DATABASE_URL env var, default
postgresql://cardledger:cardledger@localhost:5432/cardledger
"""
import json
import os
import random
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import psycopg

SEED = 42
N_MEMBERS = 2000
N_MERCHANTS = 150
N_TRANSACTIONS = 100_000
N_DECLINED_AUTHS = 400
INJECT_AUTH = 3
INJECT_LEDGER = 4
INJECT_SETTLEMENT = 5
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


def main() -> None:
    rng = random.Random(SEED)
    dsn = os.environ.get(
        "DATABASE_URL",
        "postgresql://cardledger:cardledger@localhost:5432/cardledger")

    # ---- reference data -------------------------------------------------
    members = []
    for mid in range(1, N_MEMBERS + 1):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        joined = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 900))
        members.append((mid, name, f"member{mid}@example.com", joined))

    cards = []          # (card_id, member_id, token, network, status, issued)
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

    # ---- transactions and authorisations --------------------------------
    start = date(2026, 7, 1)
    auths = []          # approved, one per transaction, ids 1..N_TRANSACTIONS
    txns = []
    for tid in range(1, N_TRANSACTIONS + 1):
        card_id = rng.randrange(1, len(cards) + 1)
        merchant_id = rng.randrange(1, N_MERCHANTS + 1)
        amount = money(Decimal(str(round(rng.triangular(1.0, 500.0, 40.0), 2))))
        txn_type = "refund" if rng.random() < 0.05 else "purchase"
        tdate = start + timedelta(days=rng.randrange(0, 92))  # Jul to Sep 2026
        at = datetime.combine(tdate, time(rng.randrange(0, 24), rng.randrange(0, 60)))
        auths.append([tid, card_id, merchant_id, amount, "GBP", "approved", at])
        txns.append([tid, card_id, card_member[card_id], merchant_id, tid,
                     amount, "GBP", txn_type, "posted", tdate])

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

    # ---- inject authorisation amount mismatches --------------------------
    auth_mismatch_ids = sorted(rng.sample(range(1, N_TRANSACTIONS + 1), INJECT_AUTH))
    auth_deltas = [Decimal("7.50"), Decimal("12.00"), Decimal("3.25")]
    for tid, delta in zip(auth_mismatch_ids, auth_deltas):
        auths[tid - 1][3] = auths[tid - 1][3] + delta   # auth amount only

    # ---- ledger entries, skipping 4 transactions -------------------------
    remaining = [t for t in range(1, N_TRANSACTIONS + 1) if t not in auth_mismatch_ids]
    missing_ledger_ids = sorted(rng.sample(remaining, INJECT_LEDGER))
    skip = set(missing_ledger_ids)
    ledger = []
    eid = 0
    for t in txns:
        if t[0] in skip:
            continue
        amt, ttype, tdate = t[5], t[7], t[9]
        pair = (("1100", amt, Decimal("0.00")), ("2100", Decimal("0.00"), amt)) \
            if ttype == "purchase" else \
               (("1100", Decimal("0.00"), amt), ("2100", amt, Decimal("0.00")))
        for acct, dr, cr in pair:
            eid += 1
            ledger.append((eid, t[0], acct, dr, cr, tdate))

    # ---- statements (derived, opening chains from 0) ----------------------
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

    # ---- settlements (derived), then inject 5 amount mismatches ----------
    settle = {}
    for t in txns:
        key = (t[3], t[9])
        g, purch, c = settle.get(key, (Decimal("0.00"), Decimal("0.00"), 0))
        settle[key] = (g + signed(t[7], t[5]),
                       purch + (t[5] if t[7] == "purchase" else Decimal("0.00")),
                       c + 1)
    settlements = []
    for i, ((mid_m, sdate), (gross, purch, cnt)) in enumerate(sorted(settle.items()), 1):
        fees = money(purch * FEE_RATE)
        settlements.append([i, mid_m, sdate, gross, fees, gross - fees, cnt])
    stl_mismatch_ids = sorted(rng.sample(range(1, len(settlements) + 1), INJECT_SETTLEMENT))
    stl_deltas = [Decimal("25.00"), Decimal("9.99"), Decimal("41.50"),
                  Decimal("6.75"), Decimal("18.20")]
    for sid_, delta in zip(stl_mismatch_ids, stl_deltas):
        row = settlements[sid_ - 1]
        row[3] = row[3] + delta   # gross
        row[5] = row[5] + delta   # net moves equally, so net = gross - fees still holds

    injected = {
        "total": INJECT_AUTH + INJECT_LEDGER + INJECT_SETTLEMENT,
        "auth_amount_mismatch": {"txn_ids": auth_mismatch_ids,
                                 "count": INJECT_AUTH},
        "missing_ledger_entries": {"txn_ids": missing_ledger_ids,
                                   "count": INJECT_LEDGER},
        "settlement_amount_mismatch": {"settlement_ids": stl_mismatch_ids,
                                       "count": INJECT_SETTLEMENT},
    }

    # ---- load -------------------------------------------------------------
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for tbl in ("reconciliation_exceptions", "settlements", "statements",
                        "ledger_entries", "transactions", "authorisations",
                        "cards", "merchants", "card_members"):
                cur.execute(f"TRUNCATE {tbl} RESTART IDENTITY CASCADE")
            loads = [
                ("card_members", "(member_id, full_name, email, joined_date)", members),
                ("cards", "(card_id, member_id, card_number_token, network, status, issued_date)", cards),
                ("merchants", "(merchant_id, merchant_name, mcc, country)", merchants),
                ("authorisations", "(auth_id, card_id, merchant_id, amount, currency, auth_status, authorised_at)",
                 [tuple(a) for a in auths] + declined),
                ("transactions", "(txn_id, card_id, member_id, merchant_id, auth_id, amount, currency, txn_type, status, transaction_date)",
                 [tuple(t) for t in txns]),
                ("ledger_entries", "(entry_id, txn_id, account_code, debit, credit, entry_date)", ledger),
                ("statements", "(statement_id, member_id, period_start, period_end, opening_balance, closing_balance, due_date)", statements),
                ("settlements", "(settlement_id, merchant_id, settlement_date, gross_amount, fees, net_amount, txn_count)",
                 [tuple(s) for s in settlements]),
            ]
            for tbl, cols, rows in loads:
                with cur.copy(f"COPY {tbl} {cols} FROM STDIN") as copy:
                    for r in rows:
                        copy.write_row(r)
            # reset identity sequences for tables loaded with explicit ids
            for tbl, col in (("card_members", "member_id"), ("cards", "card_id"),
                             ("merchants", "merchant_id"),
                             ("authorisations", "auth_id"),
                             ("ledger_entries", "entry_id"),
                             ("statements", "statement_id"),
                             ("settlements", "settlement_id")):
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence('{tbl}', '{col}'), "
                    f"(SELECT max({col}) FROM {tbl}))")
        conn.commit()

    Path(__file__).with_name("injected.json").write_text(json.dumps(injected, indent=2))
    print(f"members={len(members)} cards={len(cards)} merchants={len(merchants)}")
    print(f"authorisations={len(auths) + len(declined)} "
          f"(approved={len(auths)}, declined={len(declined)})")
    print(f"transactions={len(txns)} ledger_entries={len(ledger)}")
    print(f"statements={len(statements)} settlements={len(settlements)}")
    print(f"INJECTED_MISMATCHES total={injected['total']} "
          f"auth={INJECT_AUTH} ledger={INJECT_LEDGER} settlement={INJECT_SETTLEMENT}")
    print("injected ids written to injected.json")


if __name__ == "__main__":
    main()
