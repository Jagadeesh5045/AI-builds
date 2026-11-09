-- 04_cohort_retention.sql
-- Purchase cohort retention. A user's cohort is the calendar month of their
-- first purchase. month_offset is the number of whole months between the
-- cohort month and the activity month, computed from year and month parts
-- so December to January counts as 1, not 89 or -11.
WITH first_purchase AS (
    SELECT
        user_id,
        strftime('%Y-%m', MIN(ts)) AS cohort_month
    FROM events
    WHERE event_type = 'purchase'
    GROUP BY user_id
),
activity AS (
    SELECT DISTINCT
        e.user_id,
        f.cohort_month,
        (CAST(strftime('%Y', e.ts) AS INTEGER)
            - CAST(substr(f.cohort_month, 1, 4) AS INTEGER)) * 12
        + (CAST(strftime('%m', e.ts) AS INTEGER)
            - CAST(substr(f.cohort_month, 6, 2) AS INTEGER)) AS month_offset
    FROM events e
    JOIN first_purchase f ON f.user_id = e.user_id
    WHERE e.event_type = 'purchase'
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_users
    FROM first_purchase
    GROUP BY cohort_month
)
SELECT
    a.cohort_month,
    s.cohort_users,
    a.month_offset,
    COUNT(DISTINCT a.user_id) AS active_users,
    ROUND(100.0 * COUNT(DISTINCT a.user_id) / s.cohort_users, 2)
        AS retention_pct
FROM activity a
JOIN cohort_sizes s ON s.cohort_month = a.cohort_month
GROUP BY a.cohort_month, a.month_offset
ORDER BY a.cohort_month, a.month_offset;
