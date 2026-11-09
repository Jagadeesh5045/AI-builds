-- 03_daily_windows.sql
-- Daily revenue with window functions: a 7-day moving average using an
-- explicit ROWS frame, a running total, and day-on-day change via LAG.
-- The moving average is only meaningful because 05_calendar_spine.sql
-- proves the daily series has no missing dates; on a gappy series a ROWS
-- frame would silently average over the wrong days.
WITH daily AS (
    SELECT
        date(ts) AS day,
        COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases,
        SUM(revenue_pence) AS revenue_pence
    FROM events
    GROUP BY date(ts)
)
SELECT
    day,
    purchases,
    revenue_pence,
    ROUND(AVG(revenue_pence) OVER (
        ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2)
        AS revenue_7d_avg_pence,
    SUM(revenue_pence) OVER (ORDER BY day) AS revenue_running_total_pence,
    revenue_pence - LAG(revenue_pence) OVER (ORDER BY day)
        AS revenue_dod_change_pence
FROM daily
ORDER BY day;
