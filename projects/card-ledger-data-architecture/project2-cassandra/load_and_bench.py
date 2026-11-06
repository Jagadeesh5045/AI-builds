"""Load the seeded synthetic dataset and time the access patterns.

Usage: python load_and_bench.py [--engine auto|cassandra|duckdb]

- cassandra: connects to 127.0.0.1:9042, applies model/schema.cql,
  loads every per-query table with the cassandra-driver, times AP1-AP6.
- duckdb: builds the identical per-query tables in DuckDB and times
  the same patterns. Used as the honest fallback when no Cassandra
  server can run here; RESULTS.md states which engine produced the
  numbers.
- auto: tries Cassandra first, falls back to DuckDB.

Every timing written to results_<engine>.json is measured in this run.
Nothing is estimated.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from synth import generate  # noqa: E402

REPEATS = 5


def timed(fn):
    samples = []
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return result, {"median_ms": round(statistics.median(samples), 3),
                    "min_ms": round(min(samples), 3),
                    "samples_ms": [round(x, 3) for x in samples]}


def run_duckdb(data):
    import store_duckdb
    import queries_duckdb as q
    t0 = time.perf_counter()
    con = store_duckdb.build(data)
    load_s = time.perf_counter() - t0
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["transactions_by_member_month", "transactions_by_card",
                        "transactions_by_merchant_day", "statements_by_member",
                        "settlements_by_merchant", "auth_events_by_card"]}
    tgt = q.pick_targets(data)
    out = {"engine": "duckdb", "load_seconds": round(load_s, 3),
           "row_counts": counts, "targets": {k: str(v) for k, v in tgt.items()},
           "queries": {}}
    jobs = [("AP1", lambda: q.ap1(con, tgt["member_id"], tgt["month"])),
            ("AP2", lambda: q.ap2(con, tgt["card_id"])),
            ("AP3", lambda: q.ap3(con, tgt["merchant_id"], tgt["day"])),
            ("AP4", lambda: q.ap4(con, tgt["member_id"])),
            ("AP5", lambda: q.ap5(con, tgt["merchant_id"])),
            ("AP6", lambda: q.ap6(con, tgt["card_id"]))]
    for name, fn in jobs:
        rows, timing = timed(fn)
        out["queries"][name] = {"rows_returned": len(rows), **timing}
    return out


def run_cassandra(data):
    from cassandra.cluster import Cluster
    from cassandra.query import BatchStatement, ConsistencyLevel
    cluster = Cluster(["127.0.0.1"], port=9042,
                      connect_timeout=10)
    session = cluster.connect()
    schema = (HERE / "model" / "schema.cql").read_text()
    body = "\n".join(ln for ln in schema.splitlines()
                     if not ln.strip().startswith("--"))
    clean = [x.strip() for x in body.split(";") if x.strip()]
    t0 = time.perf_counter()
    for stmt in clean:
        session.execute(stmt)
    session.set_keyspace("card_ledger")
    ins = {
        "transactions_by_member_month": session.prepare(
            "INSERT INTO transactions_by_member_month (member_id, month_bucket,"
            " transaction_date, txn_id, card_id, merchant_id, merchant_name,"
            " amount, currency, txn_type, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)"),
        "transactions_by_card": session.prepare(
            "INSERT INTO transactions_by_card (card_id, transaction_date, txn_id,"
            " member_id, merchant_id, merchant_name, card_number_token, amount,"
            " currency, txn_type, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)"),
        "transactions_by_merchant_day": session.prepare(
            "INSERT INTO transactions_by_merchant_day (merchant_id, day_bucket,"
            " transaction_date, txn_id, card_id, member_id, amount, currency,"
            " txn_type, status) VALUES (?,?,?,?,?,?,?,?,?,?)"),
        "statements_by_member": session.prepare(
            "INSERT INTO statements_by_member (member_id, period_start,"
            " statement_id, period_end, opening_balance, closing_balance,"
            " due_date) VALUES (?,?,?,?,?,?,?)"),
        "settlements_by_merchant": session.prepare(
            "INSERT INTO settlements_by_merchant (merchant_id, settlement_date,"
            " settlement_id, gross_amount, fees, net_amount, txn_count)"
            " VALUES (?,?,?,?,?,?,?)"),
        "auth_events_by_card": session.prepare(
            "INSERT INTO auth_events_by_card (card_id, event_time, event_id,"
            " member_id, merchant_id, amount, currency, outcome, reason)"
            " VALUES (?,?,?,?,?,?,?,?,?)"),
    }

    def flush(rows_by_table):
        for table, rows in rows_by_table:
            batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)
            for row in rows:
                batch.add(ins[table], row)
                if len(batch) >= 40:
                    session.execute(batch)
                    batch = BatchStatement(
                        consistency_level=ConsistencyLevel.ONE)
            if len(batch):
                session.execute(batch)

    tx = data["transactions"]
    for i in range(0, len(tx), 2000):
        chunk = tx[i:i + 2000]
        flush([
            ("transactions_by_member_month",
             [(t["member_id"], t["transaction_date"].strftime("%Y-%m"),
               t["transaction_date"], t["txn_id"], t["card_id"],
               t["merchant_id"], t["merchant_name"], t["amount"],
               t["currency"], t["txn_type"], t["status"]) for t in chunk]),
            ("transactions_by_card",
             [(t["card_id"], t["transaction_date"], t["txn_id"],
               t["member_id"], t["merchant_id"], t["merchant_name"],
               t["card_number_token"], t["amount"], t["currency"],
               t["txn_type"], t["status"]) for t in chunk]),
            ("transactions_by_merchant_day",
             [(t["merchant_id"], t["transaction_date"].date(),
               t["transaction_date"], t["txn_id"], t["card_id"],
               t["member_id"], t["amount"], t["currency"], t["txn_type"],
               t["status"]) for t in chunk]),
        ])
    flush([("statements_by_member",
            [(x["member_id"], x["period_start"], x["statement_id"],
              x["period_end"], x["opening_balance"], x["closing_balance"],
              x["due_date"]) for x in data["statements"]]),
           ("settlements_by_merchant",
            [(x["merchant_id"], x["settlement_date"], x["settlement_id"],
              x["gross_amount"], x["fees"], x["net_amount"], x["txn_count"])
             for x in data["settlements"]]),
           ("auth_events_by_card",
            [(e["card_id"], e["event_time"], e["event_id"], e["member_id"],
              e["merchant_id"], e["amount"], e["currency"], e["outcome"],
              e["reason"]) for e in data["auth_events"]])])
    load_s = time.perf_counter() - t0

    import queries_duckdb as q
    import uuid as _uuid
    tgt = q.pick_targets(data)
    # pick_targets stringifies ids for DuckDB; Cassandra uuid columns
    # need real UUID objects as query parameters.
    tgt = {**tgt, "member_id": _uuid.UUID(tgt["member_id"]),
           "card_id": _uuid.UUID(tgt["card_id"]),
           "merchant_id": _uuid.UUID(tgt["merchant_id"])}
    counts = {}
    for table in ins:
        counts[table] = session.execute(
            f"SELECT COUNT(*) FROM {table}").one()[0]
    sel = {
        "AP1": ("SELECT * FROM transactions_by_member_month "
                "WHERE member_id=%s AND month_bucket=%s",
                (tgt["member_id"], tgt["month"])),
        "AP2": ("SELECT * FROM transactions_by_card WHERE card_id=%s LIMIT 100",
                (tgt["card_id"],)),
        "AP3": ("SELECT * FROM transactions_by_merchant_day "
                "WHERE merchant_id=%s AND day_bucket=%s",
                (tgt["merchant_id"], tgt["day"])),
        "AP4": ("SELECT * FROM statements_by_member WHERE member_id=%s",
                (tgt["member_id"],)),
        "AP5": ("SELECT * FROM settlements_by_merchant WHERE merchant_id=%s",
                (tgt["merchant_id"],)),
        "AP6": ("SELECT * FROM auth_events_by_card WHERE card_id=%s LIMIT 200",
                (tgt["card_id"],)),
    }
    out = {"engine": "cassandra", "load_seconds": round(load_s, 3),
           "row_counts": counts,
           "targets": {k: str(v) for k, v in tgt.items()}, "queries": {}}
    for name, (cql, params) in sel.items():
        rows, timing = timed(lambda c=cql, p=params: list(session.execute(c, p)))
        out["queries"][name] = {"rows_returned": len(rows), **timing}
    cluster.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "cassandra", "duckdb"])
    ap.add_argument("--txns", type=int, default=50000)
    args = ap.parse_args()
    data = generate(n_txns=args.txns)
    print(f"dataset: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))
    result, engine_used, cass_error = None, None, None
    if args.engine in ("auto", "cassandra"):
        try:
            result = run_cassandra(data)
            engine_used = "cassandra"
        except Exception as exc:  # noqa: BLE001
            cass_error = f"{type(exc).__name__}: {exc}"
            print(f"Cassandra path failed: {cass_error}")
            if args.engine == "cassandra":
                raise
    if result is None:
        result = run_duckdb(data)
        engine_used = "duckdb"
    if cass_error:
        result["cassandra_error"] = cass_error
    path = HERE / f"results_{engine_used}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
