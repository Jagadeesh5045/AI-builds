-- 01_sessionize.sql
-- Sessionisation with the gaps-and-islands pattern.
-- A new session starts when the gap since the user's previous event is
-- strictly more than 30 minutes (or there is no previous event). Flagging
-- those rows and taking a cumulative SUM over the flag turns each island of
-- activity into one session number per user.
-- Dialect: SQLite (window functions). The gap test uses epoch-second
-- integer arithmetic on purpose: a julianday() float difference puts an
-- exactly-30-minute gap a hair over 30 and splits sessions that should
-- stay together (the test suite has a fixture for exactly this).
WITH ordered AS (
    SELECT
        event_id,
        user_id,
        ts,
        event_type,
        revenue_pence,
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
)
SELECT
    user_id || '-' || session_seq        AS session_id,
    user_id,
    MIN(ts)                              AS session_start,
    MAX(ts)                              AS session_end,
    COUNT(*)                             AS events_in_session,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
    SUM(revenue_pence)                   AS revenue_pence
FROM sessioned
GROUP BY user_id, session_seq
ORDER BY user_id, session_seq;
