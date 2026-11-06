# Project 1: Results (everything below was actually run)

Date of run: 6 November 2026. Machine: this Linux VM, 4 cores, 8 GB RAM.
All data is synthetic, generated with seed 42. No figure in this file is
estimated: each comes from a command output recorded during the run.

## How PostgreSQL was run (read this first)

- `sudo apt-get update` could not be completed in this session: the apt lists
  lock was held for over 20 minutes by a parallel process on the same VM, and
  `sudo apt-get install -y postgresql` failed with "Unable to locate package
  postgresql" because package lists were unavailable. That is the honest state
  of the apt route at run time.
- PostgreSQL was instead run as a real local server using the `pgserver` pip
  distribution, which bundles the genuine PostgreSQL server binaries. Version
  reported by the server itself:
  "PostgreSQL 16.2 on x86_64-pc-linux-gnu, compiled by gcc (GCC) 10.2.1".
- The server runs as the `nobody` user (PostgreSQL refuses to run as root)
  with data directory `/tmp/cardledger_pgdata`, over a Unix socket.
  Connection URI used for every command below:
  `postgresql://postgres:@/postgres?host=/tmp/cardledger_pgdata`
- All DDL, data loading, reconciliation and benchmarks below ran against that
  PostgreSQL 16.2 server. No SQLite or DuckDB substitute was needed for the
  data itself.
- As an extra check, every SQL file also parses cleanly under `pglast` (the
  real PostgreSQL grammar parser): ddl.sql 24 statements, indexes.sql 9,
  reconciliation.sql 10, benchmarks.sql 2, zero parse errors.

## Commands run, in order

```bash
psql "$DATABASE_URL" -f sql/ddl.sql                      # DDL_OK, 13 relations created
python generate_data.py                                  # 9.2 seconds wall clock
psql "$DATABASE_URL" -c "<member statement query, EXPLAIN ANALYZE>"  # x3, before indexes
psql "$DATABASE_URL" -f sql/indexes.sql                  # INDEXES_OK
psql "$DATABASE_URL" -c "<member statement query, EXPLAIN ANALYZE>"  # x3, after indexes
psql "$DATABASE_URL" -f queries/reconciliation.sql
pytest tests -v
```

## Row counts (generator output, verified again by the test suite)

| Table | Rows |
|---|---|
| card_members | 2,000 |
| cards | 2,613 |
| merchants | 150 |
| authorisations | 100,400 (100,000 approved, 400 declined) |
| transactions | 100,000 |
| ledger_entries | 199,992 (2 per transaction, minus 4 skipped transactions) |
| statements | 6,000 (2,000 members x 3 months) |
| settlements | 13,786 |

Partition counts: 2026-07: 33,539. 2026-08: 33,782. 2026-09: 32,679.
DEFAULT partition: 0. Sum equals the parent count of 100,000.

## Double entry balance

SUM(debit) = 18,035,967.44 and SUM(credit) = 18,035,967.44 across
ledger_entries. Exactly equal.

Statement integrity spot check: all 6,000 statements satisfy
closing_balance = opening_balance + signed transaction activity for that
member and period. 6,000 of 6,000.

## Reconciliation results

Injected by the generator (ids in injected.json): 12 mismatches in total.
Found by queries/reconciliation.sql:

| Issue type | Injected | Found |
|---|---|---|
| auth_amount_mismatch | 3 | 3 |
| missing_ledger_entries | 4 | 4 |
| settlement_amount_mismatch | 5 | 5 |
| Total | 12 | 12 |

The test suite also asserts the found source ids are exactly the injected ids
(transaction ids 24102, 50872, 86996 for authorisation mismatches;
6330, 6584, 51496, 87369 for missing ledger postings; settlement ids
2012, 2589, 3271, 8350, 12598 for settlement mismatches).
Informational reports: transactions with an unmatched authorisation: 0.
Approved authorisations that never cleared: 0.

## Benchmark: member statement lookup

Query: transactions for member 42, 1 to 31 August 2026, ordered by date
(15 rows returned). EXPLAIN ANALYZE execution times, three runs each:

| | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| Before indexes | 6.901 ms | 6.861 ms | 5.995 ms | 6.861 ms |
| After sql/indexes.sql | 0.506 ms | 0.334 ms | 0.323 ms | 0.334 ms |

After indexing the plan uses a Bitmap Index Scan on the August partition
index (member_id, transaction_date), and only the August partition is
scanned: monthly partition pruning is visible in the plan. These are single
user, local VM figures for this one query, not a load test.

## Tests

`pytest tests -v`: 4 passed in 0.90s.

- test_row_counts: PASSED
- test_double_entry_balances: PASSED
- test_partition_row_total_matches_parent: PASSED
- test_reconciliation_finds_exactly_the_injected_mismatches: PASSED

## What could NOT be done, and why

- System PostgreSQL via apt was not installed, for the lock and package list
  reason above. The pgserver route runs the same PostgreSQL 16.2 engine, so
  no functional substitute was needed; if this build is rerun on a machine
  with working apt, README.md gives the apt commands and nothing else changes.
- No Docker was available or used (none installed on the VM). The
  docker-compose style instructions sometimes used for this project were
  therefore not part of this run.
- Benchmarks are single query, single user, warm cache local figures. No
  concurrency, replication or high availability behaviour was tested or is
  claimed.
