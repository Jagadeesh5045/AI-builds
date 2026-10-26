"""Command line for retail-warehouse.

  python -m retail_warehouse.cli generate --batch 1 --out data/raw
  python -m retail_warehouse.cli load --raw data/raw/batch1 \
      --db data/warehouse.db --batch batch1 --effective 2025-01-01
  python -m retail_warehouse.cli report --db data/warehouse.db
  python -m retail_warehouse.cli demo --work demo_run

demo runs the whole story end to end: batch 1, then batch 2 with
movers, price rises and a late correction, then prints the quality
checks and the KPI report. It is the command the README asks a
reviewer to run first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import init_db, kpi_report, quality_checks, run_load
from .sample_data import generate_batch


def cmd_generate(args: argparse.Namespace) -> int:
    dest = generate_batch(args.batch, args.out)
    print(f"wrote {dest}")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    conn = init_db(args.db)
    manifest = run_load(conn, args.raw, args.batch, args.effective)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    conn = init_db(args.db)
    checks = quality_checks(conn)
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['check']}: {check['detail']}")
    print(json.dumps(kpi_report(conn), indent=2))
    return 0 if all(c["passed"] for c in checks) else 1


def cmd_demo(args: argparse.Namespace) -> int:
    work = Path(args.work)
    raw = work / "raw"
    db = work / "warehouse.db"
    if db.exists():
        db.unlink()
    generate_batch(1, raw)
    generate_batch(2, raw)
    conn = init_db(db)
    print("== batch 1 ==")
    print(json.dumps(run_load(conn, raw / "batch1", "batch1", "2025-01-01"), indent=2))
    print("== batch 2 (incremental) ==")
    print(json.dumps(run_load(conn, raw / "batch2", "batch2", "2025-07-01"), indent=2))
    print("== re-run batch 2 (idempotency check) ==")
    print(json.dumps(run_load(conn, raw / "batch2", "batch2", "2025-07-01"), indent=2))
    checks = quality_checks(conn)
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['check']}: {check['detail']}")
    print("== KPIs (completed orders only) ==")
    print(json.dumps(kpi_report(conn), indent=2))
    return 0 if all(c["passed"] for c in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retail-warehouse")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--batch", type=int, required=True, choices=[1, 2])
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("load")
    p.add_argument("--raw", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--effective", required=True)
    p.set_defaults(fn=cmd_load)
    p = sub.add_parser("report")
    p.add_argument("--db", required=True)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("demo")
    p.add_argument("--work", default="demo_run")
    p.set_defaults(fn=cmd_demo)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
