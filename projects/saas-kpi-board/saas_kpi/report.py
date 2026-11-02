"""Build the analyst report (Markdown) and CSV extracts from KPI results."""
from __future__ import annotations


def _money(v: float) -> str:
    return f"£{v:,.2f}"


def build_report(monthly, cohorts, plans, channels, retention, failed) -> str:
    latest = monthly[-1]
    lines = []
    lines.append("# SaaS KPI board: monthly report")
    lines.append("")
    lines.append(f"Latest month: **{latest['month']}**")
    lines.append("")
    lines.append("## Headline KPIs")
    lines.append("")
    lines.append(f"- MRR: **{_money(latest['mrr'])}** (ARR run rate {_money(latest['arr_run_rate'])})")
    lines.append(f"- Active customers: **{latest['active_customers']}**")
    lines.append(f"- ARPU: **{_money(latest['arpu'])}** per month")
    if latest["mrr_growth_pct"] is not None:
        lines.append(f"- MRR growth vs last month: **{latest['mrr_growth_pct']:.2f}%**")
    if latest["logo_churn_rate"] is not None:
        lines.append(f"- Logo churn rate: **{100 * latest['logo_churn_rate']:.2f}%**")
    if retention:
        lines.append(
            f"- Revenue retention at {retention['lag_months']} months "
            f"({retention['base_month']} cohort of {retention['customers_in_base']} customers): "
            f"**GRR {retention['grr_pct']}%**, **NRR {retention['nrr_pct']}%**"
        )
    lines.append("")
    lines.append("## Monthly trend")
    lines.append("")
    lines.append("| Month | Active | New | Churned | MRR | ARPU | Logo churn % | MRR growth % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in monthly:
        lc = "" if r["logo_churn_rate"] is None else f"{100 * r['logo_churn_rate']:.2f}"
        mg = "" if r["mrr_growth_pct"] is None else f"{r['mrr_growth_pct']:.2f}"
        lines.append(
            f"| {r['month']} | {r['active_customers']} | {r['new_customers']} | "
            f"{r['churned_customers']} | {_money(r['mrr'])} | {_money(r['arpu'])} | {lc} | {mg} |"
        )
    lines.append("")
    lines.append(f"## Plan mix ({latest['month']})")
    lines.append("")
    lines.append("| Plan | Customers | MRR | MRR share % |")
    lines.append("|---|---:|---:|---:|")
    for r in plans:
        lines.append(f"| {r['plan']} | {r['customers']} | {_money(r['mrr'])} | {r['mrr_share_pct']} |")
    lines.append("")
    lines.append("## Channel economics")
    lines.append("")
    lines.append("| Channel | Customers | Avg CAC | Avg MRR | LTV | Payback (months) | LTV:CAC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in channels:
        ltv = "" if r["ltv"] is None else _money(r["ltv"])
        lines.append(
            f"| {r['channel']} | {r['customers']} | {_money(r['avg_cac'])} | "
            f"{_money(r['avg_mrr'])} | {ltv} | {r['cac_payback_months']} | {r['ltv_cac_ratio']} |"
        )
    lines.append("")
    lines.append("## Analyst notes")
    lines.append("")
    best = max(channels, key=lambda r: (r["ltv_cac_ratio"] or 0))
    worst = min(channels, key=lambda r: (r["ltv_cac_ratio"] or 999))
    lines.append(
        f"- Best LTV:CAC is {best['channel']} at {best['ltv_cac_ratio']}x. "
        f"Weakest is {worst['channel']} at {worst['ltv_cac_ratio']}x. Spend follows value only if "
        "the second number gets fixed or the budget moves."
    )
    total_failed = sum(r["failed"] for r in failed)
    total_inv = sum(r["invoices"] for r in failed)
    lines.append(
        f"- Failed payments: {total_failed} of {total_inv} invoices "
        f"({100.0 * total_failed / total_inv:.2f}%). Small, steady, and worth a dunning email before it is worth a meeting."
    )
    lines.append(
        "- Cohort detail and the full monthly table are in the CSV extracts next to this report."
    )
    lines.append("")
    return "\n".join(lines)
