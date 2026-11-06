# Project 4: Results (everything below was actually run)

Date of run: 6 November 2026. Machine: this Linux VM, 4 cores, 8 GB RAM,
Ubuntu 24.04. All data is synthetic, generated with seed 42. No figure in
this file is estimated: each comes from a command output recorded during
the run (the benchmark and reconciliation script writes results.json, which
this file quotes).

## The server that ran

- Official binary tarball downloaded from downloads.yugabyte.com:
  `yugabyte-2024.2.3.2-b6-linux-x86_64.tar.gz` (release 2024.2.3.2).
- Version from the binary itself (`yb-master --version`):
  "version 2024.2.3.2 build 6 revision 1d80005a21157bd42050615aeca130ff3ff140ef
  build_type RELEASE built at 18 Jun 2025 22:03:25 UTC".
- Version reported by YSQL `SELECT version()`:
  "PostgreSQL 11.2-YB-2024.2.3.2-b0 on x86_64-pc-linux-gnu, compiled by clang
  version 17.0.6".
- Started with `yugabyted start` (single node, insecure mode, local use
  only), base directory `.runtime/yb-data`, replication factor 1.
  Conservative memory settings from the yugabyted config file:
  master `memory_limit_hard_bytes=536870912` (512 MB), tserver
  `memory_limit_hard_bytes=1610612736` (1.5 GB), 2 YSQL and 2 YCQL shards
  per tserver, callhome disabled.
- YSQL reachable on 127.0.0.1:5433. YCQL reachable on 127.0.0.1:9042 and
  verified with a real query through the cassandra-driver: `system.local`
  reported release_version 3.9-SNAPSHOT, and a probe keyspace accepted a
  CREATE TABLE, an INSERT and a SELECT that read the row back (probe
  keyspace dropped afterwards). The bundled ycqlsh wrapper itself could not
  start because the system python3 lacks the `six` module; that is a wrapper
  problem on this VM, not a server problem, and the driver check above is
  the YCQL evidence used here.

## Getting it running: three blockers, all fixed, all recorded

1. **yugabyted config format.** The first start attempt failed with
   "Failed to read user config file ... Expecting value: line 1 column 1"
   because the user config file must be JSON, and the failed attempt left an
   empty conf file inside the data directory which then failed the same way.
   Fixed by writing the config as JSON into `yb-data/conf/yugabyted.conf`.
2. **Missing OS locale.** The tserver then died during bootstrap:
   its embedded initdb failed with `initdb: invalid locale name
   "en_US.UTF-8"` because this VM only had the C and POSIX locales.
   Fixed with `localedef -i en_US -f UTF-8 en_US.UTF-8`, then a fresh start.
3. **/tmp full.** The embedded PostgreSQL then crash-looped with
   `FATAL: could not write lock file "/tmp/.yb.127.0.0.1:5433/.s.PGSQL.5433.lock":
   No space left on device`. /tmp is a 512 MB tmpfs and was at 100 percent,
   filled by earlier projects' leftovers. Fixed by deleting two redundant
   installer tarballs in /tmp (162 MB freed; the extracted installs live
   elsewhere under the home directory). Other projects' live data
   directories in /tmp were not touched. With space available, the tserver's
   wrapper restarted PostgreSQL by itself and YSQL came up.

## Probes run before writing the schema (outcomes drove SCHEMA_NOTES.md)

| Probe | Outcome |
|---|---|
| CREATE TABLE ... PARTITION BY RANGE + PARTITION OF, insert, select | Works |
| CREATE INDEX on a partitioned table | Works |
| Plain INSERT of an explicit id into a GENERATED ALWAYS AS IDENTITY column | Rejected, with the OVERRIDING SYSTEM VALUE hint (same as PostgreSQL) |
| COPY FROM STDIN of explicit ids into a GENERATED ALWAYS AS IDENTITY column | Works (2 rows loaded and read back) |

## Schema applied

`sql/ddl_yugabyte.sql` (Project 1 definitions, see SCHEMA_NOTES.md) applied
with ysqlsh against database `card_ledger`: all CREATE TABLE statements
succeeded first time, including the partitioned transactions table.
`sql/indexes_yugabyte.sql` then applied all six Project 1 indexes; the three
ANALYZE statements succeeded with a beta-feature warning from YSQL.

## Load (generate_data.py, seed 42, COPY FROM STDIN path used)

Wall clock for the whole load: 24.089 seconds. Row counts below are read
back from the server after loading, not taken from the generator.

| Table | Rows read back |
|---|---|
| card_members | 2,000 |
| cards | 2,613 |
| merchants | 150 |
| authorisations | 50,200 (50,000 approved, 200 declined) |
| transactions | 50,000 |
| ledger_entries | 100,000 |
| statements | 5,997 |
| settlements | 13,411 |

Statements are derived per member-month with activity, so 5,997 rather than
a round 6,000 means three member-months had no transactions in this 50,000
row draw. Settlements are derived per merchant-day with activity.

Tablet evidence (yb-admin list_tablets): the August 2026 partition
`transactions_2026_08` has 2 tablets, hash ranges split at the midpoint of
the hash space, both led by the single tserver.

## Benchmarks (bench_and_reconcile.py, client-side wall time, 5 runs each)

Q1 member statement lookup: transactions for member 42, 1 to 31 August
2026, ordered by date. 9 rows returned.

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
|---|---|---|---|---|---|
| 57.307 ms | 1.636 ms | 1.378 ms | 1.450 ms | 1.423 ms | 1.450 ms |

Run 1 is the cold first run; runs 2 to 5 are tightly grouped. EXPLAIN
ANALYZE for the same query reported Execution Time 1.076 ms, with a Merge
Append over the partitions and an Index Scan using
`transactions_2026_08_member_id_transaction_date_idx` on the August
partition only: partition pruning is visible in the plan.

Q2 merchant daily totals: merchant 7, all 91 days, grouped by date.

| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
|---|---|---|---|---|---|
| 78.033 ms | 6.439 ms | 6.705 ms | 6.540 ms | 6.493 ms | 6.540 ms |

91 rows returned.

## Reconciliation

Aggregate check:

| Measure | Value |
|---|---|
| Ledger SUM(debit) | 8,977,229.35 |
| Ledger SUM(credit) | 8,977,229.35 |
| Debits equal credits | yes |
| Ledger signed total, account 1100 (debits minus credits) | 8,067,653.05 |
| Transactions signed total (purchases minus refunds) | 8,067,653.05 |
| Ledger matches transactions | yes, exactly |

Full Project 1 reconciliation pack (the same three INSERT-based checks:
authorisation amount vs transaction amount, ledger completeness per
transaction, settlement gross vs derived transactions): **0 exceptions of
every type, 0 total.** This dataset deliberately contains no injected
mismatches (Project 1 injected 12 into its own dataset on purpose), so zero
is the expected and correct result, and the aggregate equality above is an
independent check on the same property.

## Comparison with Project 1 PostgreSQL, with the caveat stated plainly

Project 1 (PostgreSQL 16.2, 100,000 transactions) measured the same Q1
statement lookup at a median of 0.334 ms after indexes. That figure is a
server-side EXPLAIN ANALYZE execution time. The YugabyteDB median of
1.450 ms above is a client-side wall time over 50,000 transactions, and its
comparable server-side EXPLAIN ANALYZE figure is 1.076 ms. Datasets differ
by 2x, measurement points differ, both are single-node local runs with warm
caches and one user. This is therefore not an apples-to-apples race and no
winner is claimed; the honest reading is that both engines answer this
indexed statement lookup in about a millisecond or less at this scale, and
the reasons to choose YugabyteDB are distribution and fault tolerance
(design reasoning in distributed_design.md), not this query.

## What could NOT be done, and why

- **Multi-node behaviour: not tested.** One VM, one node, RF 1. Replication,
  failover, quorum reads and geo-distribution are design reasoning only,
  labelled as such in distributed_design.md.
- **The bundled ycqlsh shell: not run**, because the VM's system python3
  lacks the `six` module its wrapper imports. YCQL was verified with the
  cassandra-driver instead, as recorded above.
- **No Docker was used** (none installed on the VM); the cluster is the
  official binary tarball run directly.
- Benchmarks are single-query, single-user, local figures. No concurrency,
  throughput or load test was run or is claimed.
