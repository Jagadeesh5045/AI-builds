-- Card Ledger: reconciliation pack.
-- Run with: psql "$DATABASE_URL" -f queries/reconciliation.sql
-- Populates reconciliation_exceptions from three independent checks, then
-- reports counts, then runs an informational unmatched authorisation report
-- (report only, not inserted as exceptions).

BEGIN;
DELETE FROM reconciliation_exceptions;

-- Check 1: authorisation amount versus cleared transaction amount.
INSERT INTO reconciliation_exceptions
    (source_table, source_id, issue_type, expected_amount, actual_amount)
SELECT 'transactions', t.txn_id, 'auth_amount_mismatch', a.amount, t.amount
FROM transactions t
JOIN authorisations a ON a.auth_id = t.auth_id
WHERE a.amount <> t.amount;

-- Check 2: ledger completeness. For every transaction, the member receivable
-- account (1100) net posting (debits minus credits) must equal the signed
-- transaction amount (purchase positive, refund negative).
INSERT INTO reconciliation_exceptions
    (source_table, source_id, issue_type, expected_amount, actual_amount)
SELECT 'transactions', t.txn_id, 'missing_ledger_entries',
       CASE WHEN t.txn_type = 'purchase' THEN t.amount ELSE -t.amount END,
       COALESCE(le.ledger_signed, 0)
FROM transactions t
LEFT JOIN (
    SELECT txn_id, SUM(debit - credit) AS ledger_signed
    FROM ledger_entries
    WHERE account_code = '1100'
    GROUP BY txn_id
) le ON le.txn_id = t.txn_id
WHERE COALESCE(le.ledger_signed, 0)
      <> CASE WHEN t.txn_type = 'purchase' THEN t.amount ELSE -t.amount END;

-- Check 3: settlement gross versus the transactions it was derived from.
INSERT INTO reconciliation_exceptions
    (source_table, source_id, issue_type, expected_amount, actual_amount)
SELECT 'settlements', s.settlement_id, 'settlement_amount_mismatch',
       tx.gross_from_txns, s.gross_amount
FROM settlements s
JOIN (
    SELECT merchant_id, transaction_date,
           SUM(CASE WHEN txn_type = 'purchase' THEN amount ELSE -amount END)
               AS gross_from_txns
    FROM transactions
    GROUP BY merchant_id, transaction_date
) tx ON tx.merchant_id = s.merchant_id AND tx.transaction_date = s.settlement_date
WHERE s.gross_amount <> tx.gross_from_txns;

COMMIT;

-- Summary: exceptions found, by type.
SELECT issue_type, COUNT(*) AS exceptions_found
FROM reconciliation_exceptions
GROUP BY issue_type
ORDER BY issue_type;

SELECT COUNT(*) AS total_exceptions FROM reconciliation_exceptions;

-- Informational report only (not exceptions): transactions pointing at an
-- authorisation row that does not exist, and approved authorisations that
-- never cleared into a transaction.
SELECT COUNT(*) AS txns_with_unmatched_authorisation
FROM transactions t
WHERE t.auth_id IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM authorisations a WHERE a.auth_id = t.auth_id);

SELECT COUNT(*) AS approved_auths_never_cleared
FROM authorisations a
WHERE a.auth_status = 'approved'
  AND NOT EXISTS (SELECT 1 FROM transactions t WHERE t.auth_id = a.auth_id);
