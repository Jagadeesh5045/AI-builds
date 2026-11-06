# Access Patterns First: Card Ledger on Cassandra

Rule used throughout: no table exists unless a named read pattern needs it.
Every pattern below was written before any schema, and the Cassandra schema
in `model/schema.cql` is derived from this list, one table per pattern.
Cassandra has no joins and no ad hoc queries, so each pattern gets a
denormalised table shaped exactly like its query.

Source of truth: the relational design (Project 1) stays the system of
record. These Cassandra tables are read models fed from it. Amounts are
GBP. All data is synthetic.

## AP1: Statement view, transactions by member within one month

- Caller: member app and statement service.
- Need: given a member and a calendar month, return that month's
  transactions in date order to build or display a statement.
- Service level need: single partition read, p95 target under 50 ms at
  read time, month is the natural billing boundary so it must never
  require scanning across months.
- Derived table: `transactions_by_member_month`
  Partition key `(member_id, month_bucket)`, clustering
  `transaction_date DESC, txn_id`.

## AP2: Transactions by card, newest first

- Caller: member app card activity feed and fraud review.
- Need: given a card, return the most recent transactions first,
  typically the first 50 to 200 rows.
- Service level need: single partition read with a LIMIT, newest first
  ordering must come from clustering order, never from sorting in the
  application.
- Derived table: `transactions_by_card`
  Partition key `(card_id)`, clustering `transaction_date DESC, txn_id`.

## AP3: Transactions by merchant per day, for settlement checks

- Caller: settlement operations job.
- Need: given a merchant and a day, list every transaction that should
  be inside that day's settlement so totals can be reconciled against
  the settlement record.
- Service level need: one partition per merchant per day, complete scan
  of that partition is acceptable because it runs as a batch check, not
  an interactive read.
- Derived table: `transactions_by_merchant_day`
  Partition key `(merchant_id, day_bucket)`, clustering
  `transaction_date ASC, txn_id`.

## AP4: Statement by member and period

- Caller: member app statements screen and support tooling.
- Need: given a member, list statements newest period first, and fetch
  one statement's opening balance, closing balance and due date.
- Service level need: single partition read, small rows, must be
  strongly fresh because a member disputing a balance cannot be shown a
  stale closing figure without explanation (see consistency choices in
  the design writeup).
- Derived table: `statements_by_member`
  Partition key `(member_id)`, clustering `period_start DESC`.

## AP5: Settlements by merchant, newest first

- Caller: merchant portal and finance operations.
- Need: given a merchant, list settlements newest first with gross,
  fees, net and transaction count.
- Service level need: single partition read, low volume, freshness
  matters more than raw speed.
- Derived table: `settlements_by_merchant`
  Partition key `(merchant_id)`, clustering `settlement_date DESC`.

## AP6: Recent authorisation events by card (ephemeral)

- Caller: fraud rules and card controls.
- Need: given a card, read the last few hours or days of authorisation
  attempts, including declines, to evaluate velocity rules.
- Service level need: very high write rate, reads only ever look at
  recent events, old events have no value after 90 days.
- Derived table: `auth_events_by_card`
  Partition key `(card_id)`, clustering `event_time DESC, event_id`,
  with a table TTL of 90 days so expired events delete themselves.

## Patterns deliberately NOT supported

- "All transactions above GBP 500 across every member": no table,
  because it would need a full cluster scan. It belongs in the
  analytical store (DuckDB in Project 1 and 3), not in Cassandra.
- "Transactions by MCC category": no current caller. Adding a table per
  hypothetical question is how Cassandra models rot, so it is excluded
  until a real service needs it.
- Any join: transaction tables carry `merchant_name` and
  `card_number_token` as copied display fields so reads never join.
