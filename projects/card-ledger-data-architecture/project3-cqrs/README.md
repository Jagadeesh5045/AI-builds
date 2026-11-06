# Project 3: Event-Driven CQRS Billing Pipeline and Governance Pack

## What this is
The capstone slice of the Card Ledger data platform. It implements CQRS
narrowly, for statement enquiry, on top of the bounded contexts defined in
`context_map.md` (Authorisation, Ledger, Billing, Settlement, Reconciliation).

- **Write model** (`write_model.py`): transactions stored in PostgreSQL when a
  server is available, otherwise in a DuckDB file. Amounts are integer pence
  so totals reconcile exactly. RESULTS.md states which engine actually ran.
- **Events** (`events.py`): every committed transaction publishes a
  `TransactionRecorded` event. Transport is a real Kafka broker when one is
  reachable; otherwise a durable append-only JSON lines event log is used as
  an explicit fallback. The fallback is never described as Kafka, and
  RESULTS.md states which transport actually ran.
- **Read model** (`projector.py`): an idempotent projector builds the
  statement-enquiry view (per member, per period) in DuckDB. Processed event
  ids are stored, replays are dropped and counted, and invalid events land in
  a dead letter table with a machine-readable reason.
- **Verification** (`run_pipeline.py`): generates 20,000 seeded (seed 42)
  synthetic GBP transactions, pushes them through write model, events and
  projector, injects 25 malformed events, replays 1,000 events, and checks the
  read model totals equal the write model totals exactly.

## Run it
```
python3 -m venv .venv
.venv/bin/pip install duckdb kafka-python pytest psycopg2-binary
.venv/bin/python run_pipeline.py
.venv/bin/python -m pytest tests/ -q
```

## Governance pack
In `../governance/`: ADR-001 (relational vs wide column), ADR-002
(partitioning strategy), ADR-003 (adopting CQRS), `data_standards.md`,
`presentation_outline.md`, and `backup_restore_drill.py` with its recorded
result in RESULTS.md.

## Honest results
See `RESULTS.md` in this folder for the commands run, the actual event
counts, duplicates dropped, dead letters, timings, test results, the restore
drill outcome, and exactly what could not be run in this environment and why.
