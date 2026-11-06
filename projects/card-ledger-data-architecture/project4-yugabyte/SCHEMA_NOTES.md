# Schema notes: Project 1 PostgreSQL schema on YugabyteDB (YSQL)

The schema in `sql/ddl_yugabyte.sql` is Project 1's `sql/ddl.sql` with the same
table definitions, column names, check constraints, foreign keys and the same
monthly RANGE partitioned `transactions` design. Before writing it, four
behaviours were probed against the running server (YugabyteDB 2024.2.3.2-b6)
so that every adaptation below is based on something measured, not assumed.
Probe outcomes are recorded in RESULTS.md.

## What did NOT need adapting

- **Table definitions: unchanged.** All nine tables, the unique constraints,
  the check constraints (including `net_amount = gross_amount - fees` and the
  ledger debit/credit check), the foreign keys, and the deliberate non-FK
  columns (`transactions.auth_id`, `ledger_entries.txn_id`, kept non-FK in
  Project 1 so reconciliation can detect breakage) are byte-for-byte the
  Project 1 definitions.
- **Declarative partitioning: kept verbatim.** A probe table with
  `PARTITION BY RANGE` plus a `PARTITION OF ... FOR VALUES` clause accepted an
  insert and a select, and an index created on a partitioned parent also
  worked. The full Project 1 layout (three monthly partitions plus a default
  partition) therefore applied as written.
- **`GENERATED ALWAYS AS IDENTITY`: kept verbatim.** A plain INSERT with an
  explicit id is rejected, exactly as in PostgreSQL, with the hint to use
  OVERRIDING SYSTEM VALUE. The loader uses COPY FROM STDIN, and a probe
  confirmed COPY accepts explicit identity values on this build, which is also
  how Project 1 loaded its data. So no column definition had to change.
- **Indexes: kept verbatim.** All six indexes from Project 1's indexes.sql
  applied, including on the partitioned transactions table, where they
  materialise per partition (the Q1 plan in RESULTS.md shows the index
  `transactions_2026_08_member_id_transaction_date_idx`).

## Adaptations that were needed

1. **Database creation is a separate step.** `CREATE DATABASE card_ledger` was
   run against the default `yugabyte` database first; Project 1 loaded into
   its default database. The DDL file itself is unchanged in structure
   (BEGIN, DROP IF EXISTS, CREATE, COMMIT).
2. **ANALYZE carries a beta warning in YSQL.** All three ANALYZE statements in
   the indexes file succeeded, each with the warning that analyze is a beta
   feature in YugabyteDB 2024.2. That is a warning, not an error, and the
   statistics were used by the planner in the recorded Q1 plan.
3. **Loader mechanics (generate_data.py), not table definitions:**
   - No TRUNCATE ... RESTART IDENTITY CASCADE step. The database is created
     fresh for this run and the loader guards with DELETE FROM, so no data
     from a previous run can survive.
   - No setval sequence resets after loading explicit ids. Project 1 reset
     the PostgreSQL sequences because later inserts there rely on generated
     ids. This run performs no generated-id inserts after the load, so the
     step has nothing to do; identity sequences in YSQL are DocDB sequences
     and are not reset as part of the load.
   - Dataset scaled to 50,000 transactions and 200 declined authorisations,
     and zero injected reconciliation mismatches (Project 1 injected 12 on
     purpose). The reconciliation expectation for this dataset is therefore
     zero exceptions, stated wherever the pack results are quoted.

## How the monthly partitioning design translates

On PostgreSQL, Project 1's monthly RANGE partitions are physical storage
segments of one table on one server: pruning means whole months of files are
skipped, and dropping a month is a fast metadata operation.

On YugabyteDB each monthly partition is a separate distributed table in its
own right. Inside a partition, rows are not stored in date order on one
machine; they are hash-sharded on the first primary key column (`txn_id`)
into tablets (the August partition has 2 tablets on this single node, listed
in RESULTS.md), and each tablet is a Raft-replicated unit. Two consequences:

- Partition pruning still works at the YSQL layer and is visible in the Q1
  plan: only the August partition is scanned for an August query, using its
  own (member_id, transaction_date) index. So the monthly design still pays
  off for statement lookups.
- What partitioning does not provide here is date locality. A month's rows
  are spread by transaction hash, not gathered together, because distribution
  is the point of the store. The monthly partitions earn their place as a
  lifecycle and manageability tool (drop or archive a month as one table,
  per-month index maintenance), not as a storage-locality tool. If date
  locality ever mattered more than even write distribution, the alternative
  is a single transactions table with (txn_id HASH, transaction_date) and no
  declarative partitions, letting automatic tablet splitting handle growth;
  that trade-off is discussed in distributed_design.md.

At this dataset size (50,000 transactions) all of this is design reasoning
backed by the tablet listing for one partition. Nothing about multi-node
partition behaviour was measured, because there is only one node.
