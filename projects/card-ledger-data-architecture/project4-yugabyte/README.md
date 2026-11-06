# Project 4: Card Ledger on YugabyteDB

A real YugabyteDB run of the Project 1 card ledger design: the same tables,
the same constraints, the same monthly partitioned transactions, loaded with
a seeded synthetic dataset and checked with the same headline queries and
the same reconciliation pack. Personal portfolio project, synthetic data
only (seed 42, GBP), built on 6 November 2026. Nothing here is commercial
or client work.

## What ran

- YugabyteDB 2024.2.3.2-b6 (official Linux x86_64 tarball from
  downloads.yugabyte.com), single node started with yugabyted, RF 1,
  YSQL on port 5433, YCQL on port 9042.
- Database `card_ledger` with the Project 1 schema applied essentially
  unchanged (see SCHEMA_NOTES.md for the probes behind that statement and
  the small adaptations in database creation, ANALYZE and the loader).
- 50,000 transactions with consistent members, cards, merchants,
  authorisations and a balanced double-entry ledger (100,000 entries),
  loaded with COPY in about 24 seconds.
- Headline results (full record in RESULTS.md):
  - Member statement lookup: median 1.450 ms client-side over 5 runs,
    1.076 ms server-side in EXPLAIN ANALYZE, August partition only scanned.
  - Merchant daily totals: median 6.540 ms over 5 runs.
  - Reconciliation: ledger debits equal credits at 8,977,229.35; signed
    ledger total equals signed transaction total at 8,067,653.05; the full
    three-check reconciliation pack found 0 exceptions (this dataset has
    no injected mismatches, unlike Project 1 which injected 12 on purpose).

## Files

- `sql/ddl_yugabyte.sql` and `sql/indexes_yugabyte.sql`: the schema and
  indexes, carried over from Project 1.
- `generate_data.py`: seeded generator and COPY loader.
- `bench_and_reconcile.py`: the timed queries and reconciliation pack;
  writes `results.json`, which RESULTS.md quotes. `generated.json` records
  the generator's own counts and the load path used.
- `SCHEMA_NOTES.md`: every adaptation, and how the PostgreSQL monthly
  partitioning design translates to a hash-sharded distributed store.
- `distributed_design.md`: hash sharding and tablets, RF 1 on one node
  versus RF 3 in production, consistency, and when YugabyteDB fits instead
  of PostgreSQL or Cassandra, with demonstrated facts and design reasoning
  labelled separately.
- `RESULTS.md`: versions, the three startup blockers and their fixes,
  probes, row counts read back from the server, all timings, and the
  comparison caveat against Project 1.
- `.runtime/`: the extracted tarball, the cluster data directory and the
  Python venv, left in place with the node stopped, so the run can be
  restarted and re-checked.

## Honest scope

Single node, RF 1: there is no fault tolerance in what ran, and no
multi-node, failover or geo-distributed behaviour was measured. Those parts
of distributed_design.md are reasoning for a production version, not
results. The PostgreSQL comparison in RESULTS.md is explicitly not an
apples-to-apples race (different dataset sizes and measurement points).
