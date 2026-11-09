"""Tests for sql-session-lab.

Two layers: a tiny hand-built fixture where every expected value is known
exactly (session splits, the 30-minute boundary, moving averages, cohort
offsets), and the full generated dataset, checked for the invariants any
correct run must hold (monotonic funnel, gap-free spine, revenue that
reconciles three different ways).
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_session_lab.generate import SCHEMA, generate
from sql_session_lab.runner import run_all, run_query

FIXTURE = [
    # user A: two events 10 min apart, then one 45 min later -> 2 sessions
    ("A", "2026-08-01 09:00:00", "view", None, 0),
    ("A", "2026-08-01 09:10:00", "purchase", "P01", 1000),
    ("A", "2026-08-01 09:55:00", "view", None, 0),
    # user B: exactly 30 min apart -> still 1 session (rule is > 30)
    ("B", "2026-08-01 10:00:00", "view", None, 0),
    ("B", "2026-08-01 10:30:00", "purchase", "P02", 2000),
    # user C: 31 min apart -> 2 sessions
    ("C", "2026-08-02 11:00:00", "view", None, 0),
    ("C", "2026-08-02 11:31:00", "view", None, 0),
    # user A again next month: cohort offset check for 04
    ("A", "2026-09-05 08:00:00", "purchase", "P03", 3000),
]


def fixture_db(tmp_path):
    db = str(tmp_path / "fixture.db")
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO events (user_id, ts, event_type, product_id,"
        " revenue_pence, channel, device) VALUES (?,?,?,?,?,'direct','desktop')",
        FIXTURE,
    )
    con.commit()
    con.close()
    return db


def full_db(tmp_path):
    db = str(tmp_path / "full.db")
    summary = generate(db)
    return db, summary


def test_session_splits_on_gap_over_30_min(tmp_path):
    con = sqlite3.connect(fixture_db(tmp_path))
    _, rows = run_query(con, "01_sessionize")
    con.close()
    by_user = {}
    for r in rows:
        by_user[r[1]] = by_user.get(r[1], 0) + 1
    assert by_user == {"A": 3, "B": 1, "C": 2}  # A: Aug pair split + Sep visit


def test_exactly_30_minutes_stays_in_session(tmp_path):
    con = sqlite3.connect(fixture_db(tmp_path))
    _, rows = run_query(con, "01_sessionize")
    con.close()
    b_sessions = [r for r in rows if r[1] == "B"]
    assert len(b_sessions) == 1 and b_sessions[0][4] == 2


def test_spine_fills_missing_days(tmp_path):
    con = sqlite3.connect(fixture_db(tmp_path))
    _, rows = run_query(con, "05_calendar_spine")
    con.close()
    # 2026-08-01 .. 2026-09-05 inclusive = 36 days, none missing
    assert len(rows) == 36
    assert rows[0][0] == "2026-08-01" and rows[-1][0] == "2026-09-05"
    zero_days = [r for r in rows if r[1] == 0]
    assert len(zero_days) == 33  # only 3 days carry events in the fixture


def test_moving_average_first_row_and_running_total(tmp_path):
    con = sqlite3.connect(fixture_db(tmp_path))
    _, rows = run_query(con, "03_daily_windows")
    con.close()
    # daily revenue: 08-01 = 3000, 08-02 = 0 (no purchase), 09-05 = 3000
    by_day = {r[0]: r for r in rows}
    assert by_day["2026-08-01"][3] == 3000.0   # 7d avg of a single day
    assert by_day["2026-09-05"][4] == 6000     # running total reconciles


def test_cohort_offsets_cross_months(tmp_path):
    con = sqlite3.connect(fixture_db(tmp_path))
    _, rows = run_query(con, "04_cohort_retention")
    con.close()
    aug = {(r[0], r[2]): r for r in rows}
    # cohort 2026-08 has users A and B; A buys again in Sep -> offset 1
    assert aug[("2026-08", 0)][3] == 2
    assert aug[("2026-08", 1)][3] == 1
    assert aug[("2026-08", 1)][4] == 50.0


def test_generated_dataset_is_deterministic(tmp_path):
    _, s1 = full_db(tmp_path)
    db2 = str(tmp_path / "second.db")
    s2 = generate(db2)
    assert s1 == s2


def test_funnel_is_monotonic_on_full_data(tmp_path):
    db, _ = full_db(tmp_path)
    results = run_all(db, str(tmp_path / "out"))
    f = results["02_funnel"][1][0]
    steps = list(f[1:7])
    assert steps == sorted(steps, reverse=True)
    assert f[0] > 0 and 0 < f[7] < 100


def test_revenue_reconciles_across_queries(tmp_path):
    db, summary = full_db(tmp_path)
    results = run_all(db, str(tmp_path / "out"))
    spine_total = sum(r[3] for r in results["05_calendar_spine"][1])
    daily_total = results["03_daily_windows"][1][-1][4]
    session_total = sum(r[6] for r in results["01_sessionize"][1])
    assert spine_total == daily_total == session_total == summary["revenue_pence"]


def test_spine_is_gap_free_on_full_data(tmp_path):
    db, _ = full_db(tmp_path)
    results = run_all(db, str(tmp_path / "out"))
    spine = results["05_calendar_spine"][1]
    assert len(spine) == 92  # 2026-08-01 .. 2026-10-31
    assert sum(r[1] for r in spine) > 0
