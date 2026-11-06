"""Grounded narrative briefing.

The briefing is generated only from the computed metrics dict, and a
separate verifier extracts every number from the finished text and
checks it against the metrics. If a number in the text does not come
from the analytics layer, the verifier reports it. This is the same
groundedness idea I used in my RAG evaluation work, applied to a
management report: a fluent paragraph is not evidence.
"""
import re


def _fmt(x):
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def generate_briefing(overall, medians, gap, model_metrics):
    worst = max(medians, key=medians.get)
    best = min(medians, key=medians.get)
    lines = [
        "# Waiting list briefing (synthetic data)",
        "",
        f"Across {_fmt(overall['n_referrals'])} synthetic elective referrals, the median wait was "
        f"{_fmt(overall['median_wait_days'])} days and the mean wait was {_fmt(overall['mean_wait_days'])} days. "
        f"{_fmt(overall['breach_n'])} referrals waited longer than 18 weeks, a breach rate of {_fmt(overall['breach_pct'])}%.",
        "",
        f"The longest median wait was in {worst} at {_fmt(medians[worst])} days. "
        f"The shortest was in {best} at {_fmt(medians[best])} days.",
        "",
        f"Patients in the most deprived quintile waited a median of {_fmt(gap['median_wait_q5'])} days, "
        f"compared with {_fmt(gap['median_wait_q1'])} days in the least deprived quintile, "
        f"a gap of {_fmt(gap['median_gap_days'])} days. Their breach rates were "
        f"{_fmt(gap['breach_pct_q5'])}% and {_fmt(gap['breach_pct_q1'])}%.",
        "",
        f"A gradient boosting model trained to flag breach risk at referral scored a ROC-AUC of "
        f"{_fmt(model_metrics['roc_auc'])} on a held-out test set of {_fmt(model_metrics['n_test'])} referrals. "
        f"The strongest single signal was {model_metrics['top_feature']}.",
        "",
        "All figures above were computed from the synthetic warehouse in this repo. "
        "No real patient data was used.",
    ]
    return "\n".join(lines)


def allowed_numbers(overall, medians, gap, model_metrics):
    allowed = set()
    for src in (overall, medians, gap, model_metrics):
        for v in src.values():
            if isinstance(v, dict):
                for vv in v.values():
                    allowed.add(_fmt(float(vv)) if isinstance(vv, (int, float)) else str(vv))
            elif isinstance(v, (int, float)):
                allowed.add(_fmt(float(v)) if isinstance(v, float) else str(v))
    allowed.add("18")  # the 18-week standard, a constant named in the text
    return allowed


def verify_briefing(text, overall, medians, gap, model_metrics):
    allowed = allowed_numbers(overall, medians, gap, model_metrics)
    found = re.findall(r"\d+(?:\.\d+)?", text)
    return sorted({n for n in found if n not in allowed})
