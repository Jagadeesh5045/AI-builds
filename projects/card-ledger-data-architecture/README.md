# Card Ledger Data Architecture

A personal portfolio capstone built on 6 November 2026 to design, build and verify
the data architecture for a card payments platform: card members, cards, merchants,
authorisations, transactions, ledger entries, billing statements, merchant
settlement and reconciliation. All data is synthetic (seeded generators), in GBP,
and clearly labelled synthetic. Nothing here is commercial or client work.

## What is inside

### Project 1: Relational architecture on PostgreSQL (project1-postgres/, docs/)
- Domain, conceptual, logical and physical data models (docs/models/), with a DBML
  schema, an ER diagram and a normalisation note (3NF, with 4 documented
  denormalisation decisions).
- Real PostgreSQL 16.2 build: transactions table RANGE partitioned by month,
  constraints and indexes, 100,000 synthetic transactions, 199,992 double-entry
  ledger entries (debits equal credits at 18,035,967.44), 6,000 statements verified
  against transaction activity.
- Reconciliation queries found exactly the 12 deliberately injected mismatches.
- Index benchmark on a member statement lookup: 6.861 ms before indexes, 0.334 ms
  after (medians of 3 EXPLAIN ANALYZE runs, single-user local figures).
- Tests: 4 passed. Full record: project1-postgres/RESULTS.md.

### Project 2: Query-first wide-column redesign on Cassandra (project2-cassandra/)
- Six access patterns written first, one Cassandra table per pattern, partition and
  clustering keys justified, 90-day TTL on authorisation events, and a
  Hackolade-style JSON model.
- Apache Cassandra 4.1.9 really ran as a single local node: 193,851 rows loaded
  across the six tables, row counts read back identical to source, query medians
  2.169 to 5.320 ms across the six patterns.
- The same design was benchmarked on DuckDB for comparison; at this scale the
  honest result is parity, not a Cassandra speed win.
- Single node with replication factor 1: replication, quorum and failover behaviour
  are design reasoning in design_writeup.md, not tested behaviour.
- Tests: 6 passed. Full record: project2-cassandra/RESULTS.md.

### Project 3: CQRS pipeline and governance pack (project3-cqrs/, governance/)
- Domain Driven Design bounded contexts (Authorisation, Ledger, Billing,
  Settlement, Reconciliation) with a context map.
- CQRS pipeline: PostgreSQL write model, TransactionRecorded events, DuckDB read
  model for statement enquiry. 20,000 transactions processed; write and read totals
  matched exactly at 252,111,146 pence; 1,000 replayed duplicates dropped
  (idempotency); 25 malformed events captured in a dead letter store with reasons.
- Transport honesty: the Kafka code path exists but the Kafka broker did not start
  in this environment (out of memory with parallel builds running), so the verified
  run used a durable append-only JSON-lines event log with the same event contract.
  It is labelled file_event_log everywhere, never called Kafka.
- Governance: 3 Architecture Decision Records, data standards, and a backup and
  restore drill (PostgreSQL dump restored in 0.258s, 20,000 rows, totals identical).
- Tests: 3 passed. Full record: project3-cqrs/RESULTS.md.

## Honest scope

These projects create project-level evidence for data modelling, PostgreSQL,
Cassandra, CQRS, Domain Driven Design patterns and architecture documentation.
They do not create financial services employment, enterprise governance experience
or architect job tenure, and they are never presented that way.
