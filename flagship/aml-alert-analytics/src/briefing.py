"""Grounded briefing: every number in the text is re-checked.

The briefing is written only from the analytics and model
dictionaries. After it is drafted, a verifier extracts every number
from every claim and checks it against the source values. A claim
with even one number that cannot be traced back is counted as
ungrounded, and a briefing with any ungrounded claims is not fit
to share.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
TOP_DECILE_PCT = 10


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def generate_briefing(analytics: dict, model_metrics: dict, etl_counts: dict) -> dict:
    """Draft the briefing claims from source numbers only."""
    rule = analytics["riskiest_rule"]
    tier = analytics["by_risk_tier"]
    channel = analytics["riskiest_channel"]
    bands = analytics["by_amount_band"]

    claims = [
        f"Of {analytics['total_alerts']} alerts in the warehouse, "
        f"{analytics['total_suspicious']} were confirmed suspicious, "
        f"an overall confirmed-suspicious rate of {_pct(analytics['overall_rate'])}.",

        f"The {rule['rule']} rule had the highest confirmed-suspicious rate "
        f"at {_pct(rule['rate'])} across {rule['alerts']} alerts.",

        f"High risk tier customers were confirmed suspicious at "
        f"{_pct(tier['high']['rate'])}, compared with {_pct(tier['low']['rate'])} "
        f"for low risk tier customers.",

        f"The {channel['channel']} channel had the highest "
        f"confirmed-suspicious rate by channel, at {_pct(channel['rate'])}.",

        f"Alerts for amounts of 50000 or more were confirmed suspicious at "
        f"{_pct(bands['50k_plus']['rate'])}, compared with "
        f"{_pct(bands['under_1k']['rate'])} for alerts under 1000.",

        f"Median transaction amount was {analytics['median_amount_suspicious']:.2f} "
        f"for confirmed-suspicious alerts and "
        f"{analytics['median_amount_not_suspicious']:.2f} for alerts that "
        f"were not confirmed suspicious.",

        f"The ETL stage quarantined {etl_counts['quarantined_rows']} of "
        f"{etl_counts['raw_rows']} raw records, loading "
        f"{etl_counts['warehouse_rows']} clean records into the warehouse.",

        f"Analysts closed {analytics['by_disposition']['false_positive']['alerts']} "
        f"alerts as false positives and confirmed "
        f"{analytics['by_disposition']['confirmed_suspicious']['alerts']} "
        f"alerts as suspicious.",

        f"The gradient boosting model achieved a test ROC-AUC of "
        f"{model_metrics['roc_auc']:.3f} on {model_metrics['test_alerts']} "
        f"held-out test alerts.",

        f"The highest-risk 10% of test alerts captured "
        f"{_pct(model_metrics['top_decile_capture_rate'])} of all "
        f"confirmed-suspicious cases in the test set.",
    ]
    return {"claims": claims, "text": " ".join(claims)}


def _collect_numbers(obj, sink: set) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        sink.add(round(float(obj), 4))
        sink.add(round(float(obj) * 100, 4))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _collect_numbers(key, sink)
            _collect_numbers(value, sink)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            _collect_numbers(item, sink)
    elif isinstance(obj, str):
        for token in _NUMBER_RE.findall(obj):
            sink.add(float(token.replace(",", "")))


def source_number_pool(analytics: dict, model_metrics: dict, etl_counts: dict) -> set:
    """Every number a briefing is allowed to use, in any quoted form."""
    pool: set = set()
    for source in (analytics, model_metrics, etl_counts):
        _collect_numbers(source, pool)
    pool.update({float(TOP_DECILE_PCT), 1000.0, 50000.0})
    rounded = set()
    for value in pool:
        for places in (0, 1, 2, 3):
            rounded.add(round(value, places))
    return pool | rounded


def verify_briefing(briefing: dict, analytics: dict, model_metrics: dict,
                    etl_counts: dict) -> int:
    """Count claims containing a number that no source value supports."""
    pool = source_number_pool(analytics, model_metrics, etl_counts)
    ungrounded = 0
    for claim in briefing["claims"]:
        numbers = [float(t.replace(",", "")) for t in _NUMBER_RE.findall(claim)]
        if any(not any(abs(number - allowed) < 0.011 for allowed in pool)
               for number in numbers):
            ungrounded += 1
    return ungrounded
