-- 02_funnel.sql
-- Session-level funnel: how many sessions reach each step, and the
-- step-to-step conversion. Uses the same gaps-and-islands sessionisation
-- as 01, then conditional aggregation (SUM of CASE) to count reach per step.
-- A session counts for a step if it contains at least one event of that type.
WITH ordered AS (
    SELECT
        user_id,
        ts,
        event_type,
        LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) AS prev_ts
    FROM events
),
flagged AS (
    SELECT
        *,
        CASE
            WHEN prev_ts IS NULL THEN 1
            WHEN CAST(strftime('%s', ts) AS INTEGER)
               - CAST(strftime('%s', prev_ts) AS INTEGER) > 1800 THEN 1
            ELSE 0
        END AS is_new_session
    FROM ordered
),
sessioned AS (
    SELECT
        *,
        SUM(is_new_session) OVER (
            PARTITION BY user_id ORDER BY ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_seq
    FROM flagged
),
per_session AS (
    SELECT
        user_id,
        session_seq,
        MAX(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END)         AS r_view,
        MAX(CASE WHEN event_type = 'search' THEN 1 ELSE 0 END)       AS r_search,
        MAX(CASE WHEN event_type = 'product_view' THEN 1 ELSE 0 END) AS r_product,
        MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END)  AS r_cart,
        MAX(CASE WHEN event_type = 'checkout' THEN 1 ELSE 0 END)     AS r_checkout,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END)     AS r_purchase
    FROM sessioned
    GROUP BY user_id, session_seq
)
SELECT
    COUNT(*)                                   AS sessions,
    SUM(r_view)                                AS reached_view,
    SUM(r_search)                              AS reached_search,
    SUM(r_product)                             AS reached_product,
    SUM(r_cart)                                AS reached_cart,
    SUM(r_checkout)                            AS reached_checkout,
    SUM(r_purchase)                            AS reached_purchase,
    ROUND(100.0 * SUM(r_purchase) / COUNT(*), 2) AS session_conversion_pct
FROM per_session;
