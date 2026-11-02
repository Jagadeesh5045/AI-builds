-- SaaS KPI board: named SQL queries.
-- These are the exact queries saas_kpi/kpis.py runs (a test keeps them in sync).
-- Each query runs against the SQLite warehouse built by main.py.

-- name: monthly_kpis
WITH m AS (
            SELECT month, COUNT(*) AS active_customers, SUM(mrr) AS mrr
            FROM subscription_snapshots GROUP BY month
        ),
        new_c AS (
            SELECT signup_month AS month, COUNT(*) AS new_customers
            FROM customers GROUP BY signup_month
        ),
        churned AS (
            SELECT churn_month AS month, COUNT(*) AS churned_customers
            FROM customers WHERE churn_month IS NOT NULL GROUP BY churn_month
        )
        SELECT m.month, m.active_customers, ROUND(m.mrr, 2) AS mrr,
               ROUND(m.mrr * 12, 2) AS arr_run_rate,
               ROUND(m.mrr / m.active_customers, 2) AS arpu,
               COALESCE(n.new_customers, 0) AS new_customers,
               COALESCE(c.churned_customers, 0) AS churned_customers
        FROM m
        LEFT JOIN new_c n ON n.month = m.month
        LEFT JOIN churned c ON c.month = m.month
        ORDER BY m.month;

-- name: cohort_retention
SELECT c.signup_month AS cohort, s.month AS month,
               COUNT(DISTINCT s.customer_id) AS retained,
               (SELECT COUNT(*) FROM customers c2 WHERE c2.signup_month = c.signup_month) AS cohort_size,
               ROUND(SUM(s.mrr), 2) AS retained_mrr
        FROM customers c
        JOIN subscription_snapshots s ON s.customer_id = c.customer_id
        GROUP BY c.signup_month, s.month
        ORDER BY c.signup_month, s.month;

-- name: plan_mix
SELECT month, plan, COUNT(*) AS customers, ROUND(SUM(mrr), 2) AS mrr
        FROM subscription_snapshots
        WHERE month = (SELECT MAX(month) FROM subscription_snapshots)
        GROUP BY month, plan ORDER BY mrr DESC;

-- name: channel_economics
SELECT c.channel,
               COUNT(DISTINCT c.customer_id) AS customers,
               ROUND(AVG(c.cac), 2) AS avg_cac,
               ROUND(SUM(i.amount), 2) AS revenue_collected,
               ROUND(AVG(s.mrr), 2) AS avg_mrr
        FROM customers c
        JOIN subscription_snapshots s ON s.customer_id = c.customer_id
        JOIN invoices i ON i.customer_id = c.customer_id AND i.month = s.month
        GROUP BY c.channel ORDER BY revenue_collected DESC;

-- name: failed_payments
SELECT month, COUNT(*) AS invoices, SUM(1 - paid) AS failed,
               ROUND(100.0 * SUM(1 - paid) / COUNT(*), 2) AS failed_pct
        FROM invoices GROUP BY month ORDER BY month;

