-- Card Ledger: indexes for YugabyteDB (YSQL), carried over from Project 1
-- (project1-postgres/sql/indexes.sql) unchanged. Applied before loading, so
-- the benchmark figures in RESULTS.md are with-index figures.

CREATE INDEX IF NOT EXISTS idx_txn_member_date   ON transactions (member_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_merchant_date ON transactions (merchant_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_auth          ON transactions (auth_id);
CREATE INDEX IF NOT EXISTS idx_ledger_txn        ON ledger_entries (txn_id);
CREATE INDEX IF NOT EXISTS idx_cards_member      ON cards (member_id);
CREATE INDEX IF NOT EXISTS idx_auth_card_time    ON authorisations (card_id, authorised_at);

ANALYZE transactions;
ANALYZE ledger_entries;
ANALYZE authorisations;
