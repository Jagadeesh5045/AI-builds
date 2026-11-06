# Design Writeup: Card Ledger on Cassandra

## Starting point

Cassandra modelling runs backwards compared with relational modelling.
The relational schema (Project 1) models the data once, in third normal
form, and answers questions with joins at read time. Cassandra models
the questions: access patterns are fixed first (see
`access_patterns.md`), then one denormalised table is built per
pattern, and the data is written several times, once per table. Storage
is cheap, joins at read time are not possible, so duplication is the
price of predictable single partition reads.

The relational store remains the system of record. These tables are
read models, rebuilt from it, which is exactly the CQRS split used in
Project 3.

## Partitioning strategy

Every partition key was chosen against three tests:

1. It matches the equality filters of its access pattern exactly.
2. It bounds partition size, so no partition grows forever.
3. It spreads load, so write and read traffic distributes across nodes
   by the partition key hash instead of piling onto one key.

Per table:

- `transactions_by_member_month`, key `(member_id, month_bucket)`:
  a member's heaviest month is still one month of one person's card
  use, a small partition. Month bucketing stops a long lived member
  account becoming one giant partition, and it spreads one member's
  traffic across 12 partitions a year instead of one hot key.
- `transactions_by_card`, key `(card_id)`: cards are numerous and
  traffic per card is human scale, so load spreads naturally. Reads
  always LIMIT, so a busy card cannot return an unbounded result.
- `transactions_by_merchant_day`, key `(merchant_id, day_bucket)`:
  the day bucket is the safety valve. A national retailer's raw
  merchant partition would be the classic hot partition; splitting by
  day keeps each partition to one day of one merchant, and settlement
  jobs read exactly one such partition.
- `statements_by_member` and `settlements_by_merchant`: tiny reference
  style data, a few rows per partition per year, no hot partition risk.
- `auth_events_by_card`, key `(card_id)`: authorisation writes are the
  highest rate in the system. Keying by card spreads them across every
  card in the estate, and the 90 day TTL stops the table growing into
  an archive it was never meant to be.

No partition key uses a low cardinality value on its own (for example
currency or status), because that would funnel huge fractions of all
traffic onto a handful of partitions.

## Replication

- Local verification setup: `SimpleStrategy`, replication factor 1,
  one node. There is no redundancy locally and the schema says so in a
  comment rather than pretending otherwise.
- Production: `NetworkTopologyStrategy`, replication factor 3 per data
  centre, rack aware placement, so the loss of one node or one rack
  loses no data and reads continue.

## Consistency choices, per pattern

Consistency is a per query decision in Cassandra, not a database
setting. With production RF 3 in mind:

- AP1 statement view and AP2 card feed: `LOCAL_QUORUM` writes,
  `LOCAL_ONE` reads are acceptable for the activity feed (a transaction
  appearing a second late in a feed is tolerable), but the statement
  view uses `LOCAL_QUORUM` reads because a statement is a document of
  record for a billing period.
- AP3 settlement checks: `LOCAL_QUORUM` reads. Reconciliation against
  money movement must not run on a possibly stale replica view.
- AP4 statements and AP5 settlements: `LOCAL_QUORUM` reads and writes.
  Balances and settlement totals are the numbers members and merchants
  dispute, so freshness wins over the small latency saving.
- AP6 authorisation events: `LOCAL_ONE` writes and reads are
  acceptable. Velocity rules work on approximate recent data, and the
  write path must never block on quorum during a payment.

Locally, with RF 1, every level collapses to the single node, which
the loader records in RESULTS.md rather than glossing over.

## When NOT to use Cassandra

This design is deliberately incomplete, and naming the exclusions is
part of the design:

- Ad hoc and analytical queries. "Show me all refunds over GBP 500 by
  MCC last quarter" cannot be answered without a full scan. Analytics
  belongs in DuckDB or a warehouse, which is how Projects 1 and 3 use
  it.
- Anything needing joins, transactions across entities, or strong
  relational integrity. Account opening, ledger double entry and
  balance accounting stay relational, where foreign keys and ACID
  transactions exist.
- Low volume data. Statements and settlements are included here only
  to complete the read model picture; at their real volumes a
  relational table would serve them just as well, and a design that
  puts everything in Cassandra regardless of fit is a worse design.
- Workloads where the questions are not known in advance. Cassandra
  punishes unknown questions with new tables and backfills. If the
  product team cannot name the reads, do not choose Cassandra yet.
- Small systems. One PostgreSQL instance with good indexes answers
  every pattern in this project at portfolio scale. Cassandra earns
  its operational cost only when write volume or multi region needs
  outgrow a single relational writer.

## YugabyteDB and GCP Spanner positioning (comparison only)

Design comparison only: neither system was used in this project and
nothing here claims hands on experience with them.

- YugabyteDB keeps a PostgreSQL compatible SQL surface (YSQL) while
  distributing storage and Raft based replication underneath. For a
  team whose patterns are still partly relational, it offers
  horizontal scale without giving up SQL and joins, at the cost of a
  younger ecosystem than PostgreSQL itself.
- GCP Spanner is a fully managed, globally distributed relational
  database with strong external consistency across regions and
  standard SQL. It removes most operational burden and suits globally
  consistent transactional data, at managed service pricing and with
  Google Cloud lock in.
- Against both, Cassandra's edge is very high write throughput,
  multi data centre active active writes, and no single primary. Its
  cost is the query-first discipline in this document: no joins, no ad
  hoc SQL, one table per question. If a workload needs both global
  scale and relational freedom, Spanner or YugabyteDB is usually the
  better first answer; if it is an append heavy, pattern known,
  write dominated feed like authorisation events, Cassandra remains
  the natural fit.
