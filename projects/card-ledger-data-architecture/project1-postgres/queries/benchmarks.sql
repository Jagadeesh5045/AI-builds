-- Card Ledger: benchmark queries.
-- The member statement lookup is the benchmarked query in RESULTS.md: run it
-- with EXPLAIN ANALYZE before sql/indexes.sql is applied, then again after.
-- Member 42 and August 2026 are fixed so both runs are identical.

-- Q1 (benchmarked): member statement lookup, one member, one month.
EXPLAIN (ANALYZE, BUFFERS)
SELECT txn_id, transaction_date, merchant_id, txn_type, amount
FROM transactions
WHERE member_id = 42
  AND transaction_date >= DATE '2026-08-01'
  AND transaction_date <  DATE '2026-09-01'
ORDER BY transaction_date;

-- Q2: merchant daily totals for one merchant across the whole period.
SELECT transaction_date,
       COUNT(*) AS txn_count,
       SUM(CASE WHEN txn_type = 'purchase' THEN amount ELSE -amount END)
           AS signed_total
FROM transactions
WHERE merchant_id = 7
GROUP BY transaction_date
ORDER BY transaction_date;
