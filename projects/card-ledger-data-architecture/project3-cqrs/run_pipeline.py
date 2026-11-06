"""Run the CQRS pipeline end to end and verify it.

Generates seeded synthetic card transactions, writes them to the write model,
publishes a TransactionRecorded event per transaction, projects them into the
statement-enquiry read model, injects malformed events (dead letter proof) and
replays a slice of events (idempotency proof). Writes RESULTS.md with the
actual measured numbers.
"""
import json
import os
import random
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from events import get_transport, make_event
from projector import Projector
from write_model import WriteModel

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_TXN = int(os.environ.get("CQRS_N_TXN", "20000"))
N_MALFORMED = 25
N_REPLAY = 1000
SEED = 42


def generate(n=N_TXN, seed=SEED):
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    rows = []
    for i in range(n):
        member = f"MBR-{rng.randint(1, 500):04d}"
        rows.append(
            (
                f"TXN-{i + 1:06d}",
                f"CARD-{rng.randint(1, 800):04d}",
                member,
                f"MERCH-{rng.randint(1, 120):03d}",
                f"AUTH-{i + 1:06d}",
                rng.randint(100, 25000),
                "GBP",
                rng.choice(["purchase", "purchase", "purchase", "cash_advance", "payment"]),
                "posted",
                start + timedelta(days=rng.randint(0, 89)),
            )
        )
    return rows


def malformed_events():
    bad = []
    for i in range(N_MALFORMED):
        e = {
            "event_id": f"evt-BAD-{i + 1:03d}",
            "event_type": "TransactionRecorded",
            "txn_id": f"BAD-{i + 1:03d}",
            "card_id": "CARD-0001",
            "member_id": "MBR-0001",
            "merchant_id": "MERCH-001",
            "auth_id": None,
            "amount_pence": -500,
            "currency": "GBP",
            "txn_type": "purchase",
            "status": "posted",
            "transaction_date": "2026-02-01",
        }
        if i % 3 == 1:
            e["amount_pence"] = 1000
            e["currency"] = "EUR"
        if i % 3 == 2:
            e["amount_pence"] = 1000
            e["member_id"] = ""
        bad.append(e)
    return bad


def main():
    os.makedirs(BASE, exist_ok=True)
    t0 = time.time()
    rows = generate()
    write = WriteModel(BASE)
    t_write = time.time()
    write.insert_transactions(rows)
    write_seconds = time.time() - t_write

    events = [make_event(r) for r in rows]
    transport = get_transport(BASE)
    t_pub = time.time()
    transport.publish(events)
    bad = malformed_events()
    transport.publish(bad)
    publish_seconds = time.time() - t_pub

    stream = transport.consume_all(len(events) + len(bad))
    projector = Projector(BASE)
    t_proj = time.time()
    projector.apply(stream)
    projector.apply(stream[:N_REPLAY])  # deliberate replay for idempotency proof
    project_seconds = time.time() - t_proj

    write_totals = write.totals_by_member_period()
    read_totals = projector.totals_by_member_period()
    totals_match = write_totals == read_totals
    elapsed = time.time() - t0

    results = {
        "write_engine": write.engine,
        "transport": transport.name,
        "transactions_generated": len(rows),
        "write_rows": write.row_count(),
        "events_in_stream": len(stream),
        "applied": projector.applied,
        "duplicates_dropped": projector.duplicates,
        "dead_letters": projector.dead,
        "write_total_pence": write.total_pence(),
        "read_total_pence": projector.total_pence(),
        "totals_match": totals_match,
        "write_seconds": round(write_seconds, 2),
        "publish_seconds": round(publish_seconds, 2),
        "project_seconds": round(project_seconds, 2),
        "total_seconds": round(elapsed, 2),
    }
    with open(os.path.join(BASE, "run_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0 if totals_match else 1


if __name__ == "__main__":
    sys.exit(main())
