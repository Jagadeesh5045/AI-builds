# Card Ledger: Physical Model (PostgreSQL)

Physical choices for PostgreSQL. The executable form of this model is
`project1-postgres/sql/ddl.sql`. All data is synthetic, currency GBP.

## Storage choices

| Decision | Choice | Reason |
|---|---|---|
| Money type | numeric(12,2) | Exact decimal arithmetic; float is never acceptable for money |
| Primary keys | bigint generated always as identity | Simple, index friendly, no natural key leakage |
| Transactions table | RANGE partitioned by transaction_date, monthly partitions | Statement and settlement workloads are date driven; monthly partitions allow partition pruning and cheap archival of closed months |
| Partitions created | 2026-07, 2026-08, 2026-09 plus a DEFAULT partition | Three months of synthetic data, and a safe landing place for out of range rows |
| Partition key in PK | transactions primary key is (txn_id, transaction_date) | PostgreSQL requires the partition key in any unique or primary key on a partitioned table |
| Authorisation link | transactions.auth_id is a plain column, not a foreign key | A foreign key cannot reference a non partitioned parent cleanly across this design and, more importantly, reconciliation must be able to detect a transaction whose authorisation is missing: a hard FK would make that error impossible rather than detectable. Referential safety is instead checked by the reconciliation query pack. This is a deliberate, documented trade off |
| Timestamps | authorised_at and detected_at use timestamp | Sufficient for a single region portfolio build |

## Indexes (sql/indexes.sql, applied separately for benchmarking)

| Index | Definition | Serves |
|---|---|---|
| idx_txn_member_date | transactions (member_id, transaction_date) | Member statement lookup: one member, one date range |
| idx_txn_merchant_date | transactions (merchant_id, transaction_date) | Merchant daily totals and settlement checks |
| idx_txn_auth | transactions (auth_id) | Authorisation reconciliation |
| idx_ledger_txn | ledger_entries (txn_id) | Ledger completeness check per transaction |
| idx_cards_member | cards (member_id) | Member to card traversal |
| idx_auth_card_time | authorisations (card_id, authorised_at) | Authorisation history per card |

Indexes on the partitioned transactions table are created on the parent and
propagate to every partition automatically.

## Constraint strategy

- Primary keys and foreign keys (except transactions.auth_id, above) enforce
  structure at write time.
- Check constraints: amounts positive on authorisations and transactions,
  exactly one of debit and credit non zero per ledger line, settlements
  net_amount equals gross_amount minus fees, statement closing balance equals
  opening balance plus a value the application must keep consistent
  (balance arithmetic across tables is verified by reconciliation, not by a
  cross table check constraint, which PostgreSQL does not support).
- Unique constraints: cards.card_number_token, card_members.email,
  statements (member_id, period_start), settlements (merchant_id, settlement_date).

## What is deliberately not claimed

This is a single instance PostgreSQL build for a portfolio. It does not claim
high availability, replication, point in time recovery or production load
figures. Benchmark numbers in RESULTS.md are single user, local VM figures for
one stated query, before and after one stated index.
