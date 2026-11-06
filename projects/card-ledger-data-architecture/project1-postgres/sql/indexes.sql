-- Card Ledger: indexes. Applied AFTER the before-index benchmark, so RESULTS.md
-- can compare the same member statement query with and without them.
-- Indexes on the partitioned transactions table propagate to all partitions.

CREATE INDEX IF NOT EXISTS idx_txn_member_date   ON transactions (member_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_merchant_date ON transactions (merchant_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_auth          ON transactions (auth_id);
CREATE INDEX IF NOT EXISTS idx_ledger_txn        ON ledger_entries (txn_id);
CREATE INDEX IF NOT EXISTS idx_cards_member      ON cards (member_id);
CREATE INDEX IF NOT EXISTS idx_auth_card_time    ON authorisations (card_id, authorised_at);

ANALYZE transactions;
ANALYZE ledger_entries;
ANALYZE authorisations;
