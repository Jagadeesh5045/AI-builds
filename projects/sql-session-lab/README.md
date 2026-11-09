# sql-session-lab

Advanced SQL analytics over a raw clickstream: sessionisation, a session
funnel, daily revenue window functions, purchase cohort retention, and a
recursive calendar spine. The SQL files are the product. Python only
generates the data, runs the queries, and tests the answers.

## The problem

Product analytics usually starts from an events table that has no session
column, missing days, and timestamps as text. The questions a team actually
asks (how many sessions converted, what does the funnel look like, is
revenue trending up, do buyers come back next month) all need patterns that
basic `GROUP BY` cannot express: gaps-and-islands sessionisation, window
frames, and recursive date generation. This lab implements each pattern as
one self-contained, commented `.sql` file and proves every result with
tests, including the edge cases that quietly corrupt real dashboards.

## The dataset

A seeded generator (`sql_session_lab/generate.py`, standard library only)
builds the clickstream for a fictional UK outdoor store: 800 users over
92 days (2026-08-01 to 2026-10-31), arriving through 5 channels on 3 device
types, browsing 12 products and moving through a
view, search, product view, add to cart, checkout, purchase funnel with
realistic drop-off. About 12% of browsing sessions contain a pause of over
30 minutes, planted so the sessionisation query has real splits to find.
Same seed, same dataset, same revenue to the penny.

## Results (from `python run_demo.py`, seed 42)

- 6,965 events, 800 users, 320 purchasers, total revenue £30,755.34
- Sessions after 30-minute-timeout sessionisation: 2,207
- Funnel: 2,207 sessions reached view, 1,622 search, 1,451 product view,
  752 cart, 518 checkout, 406 purchase. Session conversion 18.4%
- Peak revenue day: 2026-10-22 at £1,286.83
- Calendar spine: 92 days, first 2026-08-01, last 2026-10-31, no gaps
- Revenue reconciles three ways: the daily window query's running total,
  the spine's revenue sum, and the session query's revenue sum all equal
  the generator's purchase total exactly
- 9 tests, all passing

## The five queries

| File | Pattern | Question it answers |
|---|---|---|
| `sql/01_sessionize.sql` | Gaps-and-islands: `LAG` flag + cumulative `SUM` | How many sessions were there, and what happened in each? |
| `sql/02_funnel.sql` | Sessionisation + conditional aggregation | Where do sessions drop out of the funnel? |
| `sql/03_daily_windows.sql` | `ROWS BETWEEN` frames, `LAG`, running totals | Is daily revenue trending up, and by how much? |
| `sql/04_cohort_retention.sql` | First-purchase cohorts, month-offset arithmetic | Do buyers come back in later months? |
| `sql/05_calendar_spine.sql` | Recursive CTE + `LEFT JOIN` | Which days had no activity, so averages stay honest? |

## One bug worth keeping

The first version of the gap test used `julianday()` float arithmetic. An
exactly-30-minute gap came out a fraction over 30 and split a session that
should have stayed together. The fixture test caught it (a user with events
exactly 30 minutes apart must stay in one session). The shipped queries use
epoch-second integer arithmetic instead. Boundary conditions are where
analytics code lies to you; this is why the fixtures exist.

## How to run

```bash
pip install -r requirements.txt   # runtime is stdlib only; pytest for tests
python run_demo.py                # writes outputs/report.md + CSV extracts
pytest                            # 9 tests
```

`outputs/01_sessionize.csv` (one row per session) is generated locally but
not committed; the other four CSV extracts and `report.md` are in
`outputs/` in this repo.

## Tech stack

Python (standard library: `sqlite3`, `random`, `csv`), SQL (SQLite window
functions, recursive CTEs, `FILTER` clauses), pytest. The SQL patterns port
to Postgres, BigQuery and Snowflake with dialect swaps noted in the file
headers.

## Layout

```
sql/                  the five queries, one question per file
sql_session_lab/      generate.py (seeded dataset), runner.py (executes sql/)
run_demo.py           end-to-end entry point
tests/                fixture tests with exact expected values + invariants
outputs/              report.md and CSV extracts from the verified run
```
