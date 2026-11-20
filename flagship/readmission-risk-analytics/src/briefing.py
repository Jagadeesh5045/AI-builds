"""Grounded briefing: every number in the text is re-checked.

The briefing is written only from the analytics and model dictionaries.
After it is drafted, a verifier extracts every number from every claim
and checks it against the source values. A claim with even one number
that cannot be traced back is counted as ungrounded - and a briefing
with any ungrounded claims is not fit to share.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
WINDOW_DAYS = 30
TOP_DECILE_PCT = 10


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def generate_briefing(analytics: dict, model_metrics: dict, etl_counts: dict) -> dict:
    """Draft the briefing claims from source numbers only."""
    riskiest = analytics["riskiest_condition"]
    prior = analytics["by_prior_admissions_band"]
    destinations = analytics["by_discharge_destination"]
    riskiest_destination = max(destinations.items(), key=lambda item: item[1]["rate"])
    deprivation = analytics["by_deprivation_quintile"]
    admission = analytics["by_admission_type"]

    claims = [
        f"Of {analytics['total_discharges']} discharged patients in the warehouse, "
        f"{analytics['total_readmissions']} were readmitted within 30 days, "
        f"an overall readmission rate of {_pct(analytics['overall_rate'])}.",

        f"{riskiest['condition'].capitalize()} had the highest readmission rate "
        f"at {_pct(riskiest['rate'])} across {riskiest['discharges']} discharges.",

        f"Patients with no prior admissions had a readmission rate of "
        f"{_pct(prior['0']['rate'])}, compared with {_pct(prior['3+']['rate'])} "
        f"for patients with 3 or more prior admissions.",

        f"Emergency admissions were readmitted at {_pct(admission['emergency']['rate'])}, "
        f"compared with {_pct(admission['elective']['rate'])} for elective admissions.",

        f"Patients discharged to {riskiest_destination[0]} had the highest "
        f"readmission rate by discharge destination, at "
        f"{_pct(riskiest_destination[1]['rate'])}.",

        f"Patients in the most deprived quintile were readmitted at "
        f"{_pct(deprivation['1']['rate'])}, compared with "
        f"{_pct(deprivation['5']['rate'])} in the least deprived quintile.",

        f"Median length of stay was {analytics['median_los_readmitted']} days "
        f"for readmitted patients and {analytics['median_los_not_readmitted']} "
        f"days for patients who were not readmitted.",

        f"The ETL stage quarantined {etl_counts['quarantined_rows']} of "
        f"{etl_counts['raw_rows']} raw records, loading "
        f"{etl_counts['warehouse_rows']} clean records into the warehouse.",

        f"The gradient boosting model achieved a test ROC-AUC of "
        f"{model_metrics['roc_auc']:.3f} on {model_metrics['test_patients']} "
        f"held-out test patients.",

        f"The highest-risk 10% of test patients captured "
        f"{_pct(model_metrics['top_decile_capture_rate'])} of all readmissions "
        f"in the test set.",
    ]
    return {"claims": claims, "text": " ".join(claims)}


def _collect_numbers(obj, sink: set) -> None:
    """Collect every numeric value (and numbers inside strings) in obj."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        sink.add(round(float(obj), 4))
        # Rates are also legitimately quoted as percentages.
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
    pool.update({float(WINDOW_DAYS), float(TOP_DECILE_PCT)})
    # Percentages are quoted to one decimal place and the AUC to three,
    # so rounded forms of every source value are grounded too.
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
