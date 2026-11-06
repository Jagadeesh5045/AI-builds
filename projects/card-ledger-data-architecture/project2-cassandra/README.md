# Project 2: Card Ledger, Cassandra Wide-Column Redesign

A query-first Cassandra redesign of the card transactions read model.
The relational schema (Project 1) stays the system of record; this
project builds the denormalised per-query tables that high volume
read services would use, and verifies the design by actually running
it. See `RESULTS.md` for exactly what ran, with measured numbers, and
what did not run, with the reason.

## Start here

1. `access_patterns.md`: the read patterns were fixed FIRST, with
   their service level needs. Every table exists because a pattern
   needs it, and the patterns deliberately not supported are named.
2. `model/schema.cql`: one table per pattern, keyspace `card_ledger`.
   Local setup uses SimpleStrategy RF 1 for a single node; the file
   states in a comment that production would use
   NetworkTopologyStrategy with RF 3.
3. `model/hackolade_model.json`: the same design as a Hackolade-style
   JSON data model (entities, fields, partition and clustering keys).
4. `design_writeup.md`: partitioning and hot partition reasoning,
   replication, consistency level per pattern, an explicit
   when-NOT-to-use-Cassandra section, and a design-only comparison
   with YugabyteDB and GCP Spanner (neither was used).

## Tables

| Table | Pattern | Partition key |
| --- | --- | --- |
| transactions_by_member_month | AP1 statement view | (member_id, month_bucket) |
| transactions_by_card | AP2 card feed, newest first | (card_id) |
| transactions_by_merchant_day | AP3 settlement checks | (merchant_id, day_bucket) |
| statements_by_member | AP4 statement by member and period | (member_id) |
| settlements_by_merchant | AP5 settlements by merchant | (merchant_id) |
| auth_events_by_card | AP6 authorisation events, 90 day TTL | (card_id) |

## Code

- `synth.py`: seeded (42) synthetic dataset, GBP only, about 50,000
  transactions plus statements, settlements and authorisation events
  derived from them. Same style and IDs as the relational project.
- `load_and_bench.py`: loads the dataset and times AP1 to AP6,
  5 runs each, reporting median and min. Engines: `cassandra`
  (server at 127.0.0.1:9042 via cassandra-driver) or `duckdb`
  (identical per-query tables, used as the verification fallback).
  Outputs `results_<engine>.json`.
- `store_duckdb.py` and `queries_duckdb.py`: the DuckDB
  implementation of the same tables and query shapes.
- `tests/test_access_patterns.py`: pytest assertions that each
  per-query table answers its pattern correctly against the source
  dataset (row sets, ordering, reconciliation totals).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install duckdb cassandra-driver pytest
.venv/bin/python -m pytest tests/ -q
.venv/bin/python load_and_bench.py --engine auto
```

All data is synthetic. No real members, cards or transactions.
