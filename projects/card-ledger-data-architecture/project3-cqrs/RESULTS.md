# RESULTS: Project 3 CQRS Pipeline and Governance Pack

All figures below come from runs executed in this environment on 2026-11-06.
Nothing is estimated. Raw outputs are saved under `data/`.

## What actually ran

| Component | Engine or transport that ran | Notes |
|---|---|---|
| Write model (final run) | PostgreSQL (real server) | Bundled PostgreSQL binaries from the pgserver pip package, run as user nobody on 127.0.0.1 port 54329, database cqrs_write, local trust auth for this drill only |
| Write model (baseline run) | DuckDB file | Same code path, run before PostgreSQL was available; results kept as run_results_duckdb_fallback.json |
| Event transport | Durable local append-only JSON lines event log (fallback) | NOT Kafka. See the Kafka section below |
| Read model | DuckDB | statement_enquiry projection plus processed_events and dead_letters tables |

## Pipeline run: PostgreSQL write model, file event log transport

Command:
```
CQRS_PG_DSN="host=127.0.0.1 port=54329 dbname=cqrs_write user=postgres" .venv/bin/python run_pipeline.py
```
Measured result (data/run_results_postgres.json):
- Transactions generated (seed 42): 20,000
- Rows in write model: 20,000
- Events in stream: 20,025 (20,000 valid plus 25 deliberately malformed)
- Events applied to read model: 20,000
- Duplicates dropped on deliberate replay of the first 1,000 events: 1,000
- Dead letters captured: 25 (negative amount, unsupported currency EUR, missing member_id)
- Write model total: 252,111,146 pence
- Read model total: 252,111,146 pence
- Totals match: yes, exact, per member per period and in aggregate
- Timings: write 0.81s, publish 0.20s, project 0.50s, total run 2.38s

Baseline run with the DuckDB write model produced the identical totals
(252,111,146 pence both sides, totals match yes, total run 3.22s), saved as
data/run_results_duckdb_fallback.json.

## Tests

Command: `.venv/bin/python -m pytest tests/ -q`
Result: 3 passed in 1.11s
- test_totals_match: write and read totals equal on a 200 transaction slice
- test_idempotent_replay: replaying 50 events drops them as duplicates and the read total does not move
- test_dead_letter_capture: 5 injected malformed events all captured, with at least 2 distinct reasons

## Backup and restore drill

Script: `../governance/backup_restore_drill.py`
- PostgreSQL drill (the drill of record): pg_dump of cqrs_write, restore with
  psql into a fresh database cqrs_write_restored. Rows before 20,000, rows
  after 20,000. Total before 252,111,146 pence, total after 252,111,146
  pence. Match: yes. Restore time: 0.258s.
  Saved as data/restore_drill_postgres.json.
- DuckDB drill (baseline): EXPORT DATABASE to Parquet, IMPORT DATABASE into a
  fresh file. Rows 20,000 to 20,000, totals identical, restore time 0.259s.
  Saved as data/restore_drill_duckdb.json.

## Kafka: attempted, did not run

Kafka was genuinely attempted and did not start in this environment:
- `sudo apt-get install postgresql openjdk-17-jre-headless` failed: package
  lists were stale ("Unable to locate package"), and a system apt-get update
  started by a parallel process stayed running for the whole session without
  completing, holding the package lock.
- A Temurin OpenJDK 17.0.21 JRE tarball was downloaded from Adoptium and runs.
- The official Apache Kafka 3.9.1 tarball was downloaded from
  archive.apache.org, storage was formatted in KRaft mode successfully
  (format exit 0, twice), but the broker died during startup both times with
  java.lang.OutOfMemoryError: Java heap space in the controller registration
  channel, with heap set to 512M and then 384M. Port 9092 never opened. The
  VM reported only about 496MB of available memory at the time, with several
  parallel build processes running.
- Because Kafka never ran, the pipeline used the fallback transport: a durable
  append-only JSON lines event log with the identical event contract. It is
  labelled file_event_log in every result file. It is not Kafka and is never
  described as Kafka.

## PostgreSQL: how it was made to run

- apt PostgreSQL was unavailable (above). The zonky embedded binaries route
  also failed: Maven Central returned 404 for the artifact and its metadata.
- The pgserver pip package installed, but its get_server() helper failed with
  PermissionError on chown of the data directory in this sandbox.
- What worked: pgserver's bundled PostgreSQL binaries were copied to /tmp
  (the venv path is not readable by other users), initdb was run as user
  nobody with superuser postgres and local trust auth, and the server was
  started on port 54329 with data in /tmp/pgdata3. Trust auth is acceptable
  for a local synthetic-data drill only and is not a production pattern.

## What could NOT be run, honestly

- Kafka: not run (OutOfMemoryError at broker startup, above). The event
  contract, producer and consumer code exist and are exercised against the
  fallback log; the Kafka code path itself is unverified in this environment.
- Cassandra: not part of Project 3 (it is Project 2). The read model here is
  DuckDB by design.
- Managed or cloud services: none used. Everything ran locally.

## Limitations

- Single-node everything: no replication, no partitions, no failover was
  exercised. Partitioning and replication are design decisions recorded in
  ADR-002, not measured behaviour.
- The projector is a single process; projection lag under load was not
  measured beyond the end-to-end timings above.
- Synthetic data only (seeded, seed 42), 500 members, 120 merchants, GBP only.
- The PostgreSQL server and the Kafka and JRE installs live in /tmp and home
  tooling folders, not in this repository; rerunning run_pipeline.py without
  a PostgreSQL server falls back to the DuckDB write model automatically and
  says so in its results.
