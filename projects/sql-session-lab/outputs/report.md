# sql-session-lab: run report

Dataset: 6,965 events from 800 users, 320 of them purchased. Total revenue £30,755.34.

## Sessions (01_sessionize.sql, 30-minute timeout)
- Sessions found: 2,207
- The generator plants a pause of over 30 minutes inside about 12% of browsing sessions; each pause is a split this query has to find, so the SQL session count is higher than the number of visits the generator started.

## Funnel (02_funnel.sql)
- Sessions: 2,207; reached view 2,207, search 1,622, product 1,451, cart 752, checkout 518, purchase 406.
- Session conversion (view to purchase): 18.4%.

## Daily revenue (03_daily_windows.sql)
- Days in series: 92. Running total ends at £30,755.34.
- Peak revenue day: 2026-10-22 at £1,286.83.

## Cohort retention (04_cohort_retention.sql)
- Cohort rows (cohort month x month offset): 6.

## Calendar spine (05_calendar_spine.sql)
- Spine days: 92 (first 2026-08-01, last 2026-10-31), zero-event days filled: 0.

Every number above comes straight from the SQL files in sql/, executed by sql_session_lab/runner.py. CSV extracts sit next to this report in outputs/.
