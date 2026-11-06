-- Card Ledger: Cassandra read models (Project 2)
-- Derived query-first from access_patterns.md, one table per pattern.
-- Local single-node use only as configured below.
-- PRODUCTION NOTE: production would use NetworkTopologyStrategy with
-- replication_factor 3 per data centre, multiple racks, and nodes sized
-- for the write rate. SimpleStrategy with RF 1 here is only so a single
-- local node can run the design end to end for verification.

CREATE KEYSPACE IF NOT EXISTS card_ledger
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
  AND durable_writes = true;

USE card_ledger;

-- AP1: statement view. Partition (member_id, month_bucket) keeps one
-- month of one member together and bounded in size, so no partition
-- grows without limit and no single heavy member creates a hot spot
-- across the whole cluster: load spreads across their 12 monthly
-- partitions per year.
CREATE TABLE IF NOT EXISTS transactions_by_member_month (
  member_id        uuid,
  month_bucket     text,          -- 'YYYY-MM' of transaction_date
  transaction_date timestamp,
  txn_id           uuid,
  card_id          uuid,
  merchant_id      uuid,
  merchant_name    text,          -- copied display field, avoids a join
  amount           decimal,
  currency         text,
  txn_type         text,
  status           text,
  PRIMARY KEY ((member_id, month_bucket), transaction_date, txn_id)
) WITH CLUSTERING ORDER BY (transaction_date DESC, txn_id ASC);

-- AP2: card activity feed, newest first. Partition is one card, which
-- is the natural access boundary. A single card can be busy but its
-- partition is read with a LIMIT and cards number in the thousands, so
-- traffic spreads across many partitions.
CREATE TABLE IF NOT EXISTS transactions_by_card (
  card_id          uuid,
  transaction_date timestamp,
  txn_id           uuid,
  member_id        uuid,
  merchant_id      uuid,
  merchant_name    text,
  card_number_token text,         -- copied display field, avoids a join
  amount           decimal,
  currency         text,
  txn_type         text,
  status           text,
  PRIMARY KEY ((card_id), transaction_date, txn_id)
) WITH CLUSTERING ORDER BY (transaction_date DESC, txn_id ASC);

-- AP3: settlement checks. Partition (merchant_id, day_bucket) matches
-- exactly how settlements are cut: one merchant, one day. Even the
-- largest merchant's daily partition stays small enough for a batch
-- reconciliation scan.
CREATE TABLE IF NOT EXISTS transactions_by_merchant_day (
  merchant_id      uuid,
  day_bucket       date,
  transaction_date timestamp,
  txn_id           uuid,
  card_id          uuid,
  member_id        uuid,
  amount           decimal,
  currency         text,
  txn_type         text,
  status           text,
  PRIMARY KEY ((merchant_id, day_bucket), transaction_date, txn_id)
) WITH CLUSTERING ORDER BY (transaction_date ASC, txn_id ASC);

-- AP4: statements for a member, newest period first. One partition per
-- member, at most a handful of rows per year, so this partition is
-- always tiny.
CREATE TABLE IF NOT EXISTS statements_by_member (
  member_id       uuid,
  period_start    date,
  statement_id    uuid,
  period_end      date,
  opening_balance decimal,
  closing_balance decimal,
  due_date        date,
  PRIMARY KEY ((member_id), period_start)
) WITH CLUSTERING ORDER BY (period_start DESC);

-- AP5: settlements for a merchant, newest first. One partition per
-- merchant, one row per settlement day, always small.
CREATE TABLE IF NOT EXISTS settlements_by_merchant (
  merchant_id     uuid,
  settlement_date date,
  settlement_id   uuid,
  gross_amount    decimal,
  fees            decimal,
  net_amount      decimal,
  txn_count       int,
  PRIMARY KEY ((merchant_id), settlement_date)
) WITH CLUSTERING ORDER BY (settlement_date DESC);

-- AP6: authorisation events, ephemeral and write heavy. Partition per
-- card keeps velocity checks to one partition. The 90 day TTL
-- (7,776,000 seconds) deletes old events automatically so the table
-- never becomes an unbounded archive.
CREATE TABLE IF NOT EXISTS auth_events_by_card (
  card_id     uuid,
  event_time  timestamp,
  event_id    uuid,
  member_id   uuid,
  merchant_id uuid,
  amount      decimal,
  currency    text,
  outcome     text,               -- APPROVED or DECLINED
  reason      text,
  PRIMARY KEY ((card_id), event_time, event_id)
) WITH CLUSTERING ORDER BY (event_time DESC, event_id ASC)
  AND default_time_to_live = 7776000;
