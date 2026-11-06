# RESULTS: What Actually Ran

Every number below was measured by code run in this environment on
6 November 2026. Nothing is estimated. Raw outputs are in
`results_cassandra.json` and `results_duckdb.json`.

Headline: **Apache Cassandra 4.1.9 really ran.** A single local node
was installed from the official tarball, the schema in
`model/schema.cql` was applied to it, the full seeded dataset was
loaded through the Python cassandra-driver, and all six access
patterns were timed against the live server. The identical design was
also run on DuckDB as the fallback verification, and the two engines
agree exactly on every row count and every query result size.

## Environment and how Cassandra was installed

The documented route failed first, and the workaround is recorded
because it is part of the result:

1. `sudo -n apt-get update -qq` followed by
   `sudo -n apt-get install -y -qq openjdk-17-jre-headless` was
   attempted. apt-get update hung with its http method processes idle
   for over 13 minutes and was killed. No Java was installed by apt.
   That is the exact blocker for the apt route in this environment.
2. Java was instead installed from official Eclipse Temurin tarballs
   downloaded from api.adoptium.net into `.runtime/`:
   - Temurin JRE 17.0.21 downloaded and ran, but Cassandra 4.1.9
     refused to start on Java 17 with the error
     `Unrecognized VM option 'UseConcMarkSweepGC'`
     (CMS was removed in Java 17; Cassandra 4.1 targets Java 8 and 11).
   - Temurin JRE 11.0.33 was then downloaded and Cassandra started
     correctly on it.
3. Cassandra: `apache-cassandra-4.1.9-bin.tar.gz` from
   archive.apache.org, extracted to `.runtime/cassandra`, started as a
   single node with `MAX_HEAP_SIZE=1024M` and `HEAP_NEWSIZE=256M`,
   using the `-R` flag because this VM session runs as root.
   Startup was confirmed in the server log
   (node state jump to NORMAL, "Starting listening for CQL clients on
   localhost/127.0.0.1:9042") and by a driver query returning
   `release_version = 4.1.9`.
4. Python: virtualenv `.venv` with pip packages duckdb, cassandra-driver
   and pytest installed successfully.

Server-side verification queries against `system_schema` confirmed:
keyspace `card_ledger` replication is SimpleStrategy with
replication_factor 1 (local only, as the schema states), all six
tables exist, and `auth_events_by_card` carries
`default_time_to_live = 7776000` (90 days).

The node was stopped after verification to free memory. The runtime,
including the loaded data directory, remains under `.runtime/` and
can be restarted with the same command shown in the server log.

## Dataset

Command: `.venv/bin/python synth.py` (seed 42, GBP, synthetic only)

| Entity | Rows |
| --- | --- |
| members | 1,500 |
| cards | 2,037 |
| merchants | 250 |
| transactions | 50,000 |
| statements | 9,000 |
| settlements | 26,851 |
| auth_events | 8,000 |

Loading the three transaction tables means the 50,000 transactions
are written three times, once per query table, for 193,851 total rows
loaded per engine.

## Tests

Command: `.venv/bin/python -m pytest tests/ -q`

Result: **6 passed in 47.36s.** The tests run against the DuckDB
implementation of the identical per-query tables on a 4,000
transaction seeded subset, and assert against the source dataset:
statement view row set and newest-first order (AP1), card feed order
and limit (AP2), merchant-day totals reconciling to the source sum
(AP3), six statements newest period first (AP4), settlement
gross minus fees equals net on every row (AP5), and authorisation
events newest first (AP6). The same query shapes and targets were
then exercised against live Cassandra by the benchmark below, which
returned identical result sizes for every pattern.

## Cassandra load and benchmark (live server)

Command: `.venv/bin/python load_and_bench.py --engine cassandra`

- Schema apply plus full load: **219.461 seconds** for 193,851 rows,
  using logged batches of 40 at consistency ONE.
- Row counts read back from the server (SELECT COUNT): transactions
  tables 50,000 each, statements 9,000, settlements 26,851,
  auth_events 8,000. Identical to the source dataset.
- Query targets were the busiest real keys in the data: member
  5dc1819c in month 2025-03, the busiest card, and merchant
  e5ea1ff5 on 2025-02-19.
- Each query ran 5 times; median and minimum of the samples:

| Pattern | Rows returned | Median ms | Min ms |
| --- | --- | --- | --- |
| AP1 statement view by member and month | 19 | 5.320 | 2.720 |
| AP2 card feed, newest first, limit 100 | 41 | 3.653 | 3.252 |
| AP3 merchant day settlement check | 7 | 2.432 | 2.299 |
| AP4 statements by member | 6 | 2.169 | 2.050 |
| AP5 settlements by merchant | 111 | 3.981 | 3.514 |
| AP6 auth events by card, limit 200 | 5 | 2.407 | 1.938 |

Two loader bugs were hit and fixed on the way, recorded for honesty:
the first run skipped CREATE KEYSPACE because the CQL parser dropped
a chunk that began with a comment (parser fixed to strip comment
lines before splitting), and the first benchmark attempt passed
string IDs to uuid columns, which the server rejected (fixed by
passing real UUID objects). Inserts are upserts, so the successful
final run re-applied the load cleanly and counts above are from that
final state.

## DuckDB fallback run (identical tables, full dataset)

Command: `.venv/bin/python load_and_bench.py --engine duckdb`

- Load: **481.19 seconds**. This is honestly slow: rows are inserted
  with executemany one by one. It is a loader weakness, not a design
  finding, and it is recorded rather than hidden.
- Row counts identical to Cassandra for all six tables.
- Query medians (5 runs each): AP1 4.569 ms, AP2 5.218 ms,
  AP3 2.169 ms, AP4 2.948 ms, AP5 5.195 ms, AP6 2.664 ms, with rows
  returned identical to Cassandra on every pattern (19, 41, 7, 6,
  111, 5).

At this dataset size a single node columnar engine and a single node
Cassandra are in the same few-millisecond range for single partition
reads, which is the expected and honest result: Cassandra's case at
this scale is the design discipline and write distribution, not a
speed win, exactly as the design writeup says.

## What could NOT be run, and the limits of what did run

- apt installation of Java: blocked, apt-get update hung (see above).
  Java came from Temurin tarballs instead.
- YugabyteDB and GCP Spanner: not installed, not run. The writeup
  comparison is design reading only, and claims nothing more.
- Hackolade itself: not installed. `model/hackolade_model.json` is a
  hand authored model document in the Hackolade style structure
  (entities, fields, partition and clustering keys), not a file
  exported from the Hackolade tool.
- Multi node behaviour: not tested. One node at RF 1 means
  replication, quorum consistency, failover and hot partition
  behaviour under real cluster load are design reasoning, verified
  only for correctness of the schema and queries, not for cluster
  performance. All consistency levels collapse to the single node
  locally.
- TTL expiry: the 90 day TTL is verified as a live table property in
  system_schema. Actual expiry of rows after 90 days was obviously
  not observed.
- Benchmark scope: timings are single client, single node, on a
  shared VM, with the first sample of several patterns visibly slower
  (46.517 ms on AP1's first Cassandra run). They show the patterns
  answer correctly and fast enough at portfolio scale; they are not
  production performance claims.
