"""End-to-end demo: generate the dataset, run all five SQL queries, and
write outputs/report.md plus one CSV per query.

Usage:  python run_demo.py
"""
from __future__ import annotations

from pathlib import Path

from sql_session_lab.generate import generate
from sql_session_lab.runner import run_all

ROOT = Path(__file__).resolve().parent
DB = ROOT / "outputs" / "events.db"
OUT = ROOT / "outputs"


def gbp(pence: float) -> str:
    return f"£{pence / 100:,.2f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = generate(str(DB))
    results = run_all(str(DB), str(OUT))

    _, sessions = results["01_sessionize"]
    _, funnel_rows = results["02_funnel"]
    f = funnel_rows[0]
    funnel = dict(zip(
        ["sessions", "view", "search", "product", "cart", "checkout",
         "purchase", "conv_pct"], f))
    _, daily = results["03_daily_windows"]
    _, cohorts = results["04_cohort_retention"]
    _, spine = results["05_calendar_spine"]
    zero_days = sum(1 for r in spine if r[1] == 0)
    total_revenue = daily[-1][4] if daily else 0
    peak_day = max(daily, key=lambda r: r[2]) if daily else None

    lines = [
        "# sql-session-lab: run report",
        "",
        f"Dataset: {summary['events']:,} events from {summary['users']} users, "
        f"{summary['purchasers']} of them purchased. Total revenue "
        f"{gbp(summary['revenue_pence'])}.",
        "",
        "## Sessions (01_sessionize.sql, 30-minute timeout)",
        f"- Sessions found: {len(sessions):,}",
        f"- The generator plants a pause of over 30 minutes inside about "
        f"12% of browsing sessions; each pause is a split this query has "
        f"to find, so the SQL session count is higher than the number of "
        f"visits the generator started.",
        "",
        "## Funnel (02_funnel.sql)",
        f"- Sessions: {funnel['sessions']:,}; reached view {funnel['view']:,}, "
        f"search {funnel['search']:,}, product {funnel['product']:,}, "
        f"cart {funnel['cart']:,}, checkout {funnel['checkout']:,}, "
        f"purchase {funnel['purchase']:,}.",
        f"- Session conversion (view to purchase): {funnel['conv_pct']}%.",
        "",
        "## Daily revenue (03_daily_windows.sql)",
        f"- Days in series: {len(daily)}. Running total ends at "
        f"{gbp(total_revenue)}.",
        (f"- Peak revenue day: {peak_day[0]} at {gbp(peak_day[2])}."
         if peak_day else "- No daily rows."),
        "",
        "## Cohort retention (04_cohort_retention.sql)",
        f"- Cohort rows (cohort month x month offset): {len(cohorts)}.",
        "",
        "## Calendar spine (05_calendar_spine.sql)",
        f"- Spine days: {len(spine)} (first {spine[0][0]}, last "
        f"{spine[-1][0]}), zero-event days filled: {zero_days}.",
        "",
        "Every number above comes straight from the SQL files in sql/, "
        "executed by sql_session_lab/runner.py. CSV extracts sit next to "
        "this report in outputs/.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Report written to {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
