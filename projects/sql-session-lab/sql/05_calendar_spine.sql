-- 05_calendar_spine.sql
-- Gap-free daily series from a recursive CTE calendar spine.
-- The spine walks one day at a time from the first to the last event date,
-- then LEFT JOINs the real daily aggregates, so days with no activity show
-- up as explicit zeros instead of silently disappearing (which would break
-- every moving average and day-on-day comparison downstream).
WITH RECURSIVE spine(day) AS (
    SELECT (SELECT MIN(date(ts)) FROM events)
    UNION ALL
    SELECT date(day, '+1 day')
    FROM spine
    WHERE day < (SELECT MAX(date(ts)) FROM events)
),
daily AS (
    SELECT
        date(ts) AS day,
        COUNT(*) AS events,
        COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases,
        SUM(revenue_pence) AS revenue_pence
    FROM events
    GROUP BY date(ts)
)
SELECT
    s.day,
    COALESCE(d.events, 0)        AS events,
    COALESCE(d.purchases, 0)     AS purchases,
    COALESCE(d.revenue_pence, 0) AS revenue_pence
FROM spine s
LEFT JOIN daily d ON d.day = s.day
ORDER BY s.day;
