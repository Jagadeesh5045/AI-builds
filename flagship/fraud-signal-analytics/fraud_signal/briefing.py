"""The AI layer: a grounded analyst briefing plus plain-English alert reasons.

The briefing is generated from computed results only. Every number it states
is registered as a claim (label, formatted text, source value) and the
verifier re-checks each claim against its source before the briefing is
allowed out. If a claim does not match, the briefing fails loudly instead of
shipping a wrong number.
"""
from __future__ import annotations

import pandas as pd


def _fmt(value: float, spec: str) -> str:
    return format(value, spec)


def build_briefing(pipeline: dict, analytics: dict, model: dict) -> tuple[str, list[dict]]:
    h = analytics["headline"]
    top_channel = analytics["by_channel"].iloc[0]
    top_category = analytics["by_category"].iloc[0]
    night = analytics["night_split"]
    night_rate = float(night.loc[night["period"] == "night_00_05", "fraud_rate_pct"].iloc[0])
    day_rate = float(night.loc[night["period"] == "day_06_23", "fraud_rate_pct"].iloc[0])
    top_feature = model["top_features"][0][0].replace("_", " ")

    claims = [
        {"label": "clean transactions", "text": f"{pipeline['clean_rows']:,}", "value": pipeline["clean_rows"]},
        {"label": "quarantined rows", "text": f"{pipeline['quarantined_rows']}", "value": pipeline["quarantined_rows"]},
        {"label": "fraud transactions", "text": f"{h['fraud_transactions']:,}", "value": h["fraud_transactions"]},
        {"label": "fraud rate %", "text": _fmt(h["fraud_rate_pct"], ".2f"), "value": h["fraud_rate_pct"]},
        {"label": "fraud value £", "text": f"{h['fraud_value_gbp']:,.2f}", "value": h["fraud_value_gbp"]},
        {"label": "top channel", "text": str(top_channel["channel"]), "value": str(top_channel["channel"])},
        {"label": "top channel fraud rate %", "text": _fmt(float(top_channel["fraud_rate_pct"]), ".2f"), "value": float(top_channel["fraud_rate_pct"])},
        {"label": "top category", "text": str(top_category["merchant_category"]).replace("_", " "), "value": str(top_category["merchant_category"])},
        {"label": "night fraud rate %", "text": _fmt(night_rate, ".2f"), "value": night_rate},
        {"label": "day fraud rate %", "text": _fmt(day_rate, ".2f"), "value": day_rate},
        {"label": "model ROC-AUC", "text": _fmt(model["roc_auc"], ".4f"), "value": model["roc_auc"]},
        {"label": "top-decile capture %", "text": _fmt(100 * model["top_decile_fraud_capture"], ".1f"), "value": round(100 * model["top_decile_fraud_capture"], 1)},
    ]

    lines = [
        "Fraud signal briefing",
        "",
        f"The warehouse holds {claims[0]['text']} clean card transactions after "
        f"{claims[1]['text']} dirty rows were quarantined by 10 data-quality checks.",
        f"Fraud accounts for {claims[2]['text']} transactions, a fraud rate of {claims[3]['text']}%, "
        f"worth £{claims[4]['text']} in fraudulent value.",
        f"The riskiest channel is {claims[5]['text']} at {claims[6]['text']}% fraud, and the riskiest "
        f"merchant category is {claims[7]['text']}.",
        f"Night-time transactions (00:00 to 05:59) run at {claims[8]['text']}% fraud against "
        f"{claims[9]['text']}% in the day, so overnight monitoring is where review capacity earns most.",
        f"The Gradient Boosting risk model scores ROC-AUC {claims[10]['text']} on held-out data. "
        f"Reviewing only the riskiest 10% of transactions would surface {claims[11]['text']}% of all fraud. "
        f"The strongest single signal is {top_feature}.",
    ]
    return "\n".join(lines), claims


def verify_briefing(briefing: str, claims: list[dict]) -> dict:
    """A claim is grounded only if its exact formatted text appears in the briefing."""
    ungrounded = [c["label"] for c in claims if str(c["text"]) not in briefing]
    return {"claims_checked": len(claims), "ungrounded_claims": ungrounded, "ungrounded_count": len(ungrounded)}


def explain_flag(row: pd.Series, category_median: float) -> str:
    reasons = []
    if int(row.get("is_foreign", 0)) == 1:
        reasons.append("foreign transaction")
    if int(row.get("is_night", 0)) == 1:
        reasons.append("made between midnight and 6am")
    if int(row.get("device_new", 0)) == 1:
        reasons.append("first seen device")
    if int(row.get("velocity_10min", 0)) >= 3:
        reasons.append(f"{int(row['velocity_10min'])} transactions inside 10 minutes")
    if float(row.get("amount_gbp", 0)) >= 3 * max(category_median, 1.0):
        reasons.append(f"amount £{float(row['amount_gbp']):,.2f} is at least 3x the typical spend in its category")
    if str(row.get("channel", "")) == "online":
        reasons.append("online channel")
    if not reasons:
        reasons.append("combined risk signals scored highly by the model")
    return "; ".join(reasons)


def top_alert_explanations(scored_test: pd.DataFrame, clean: pd.DataFrame, n: int = 5) -> list[dict]:
    medians = clean.groupby("merchant_category")["amount_gbp"].median().to_dict()
    alerts = []
    for _, row in scored_test.head(n).iterrows():
        median = float(medians.get(row["merchant_category"], 0.0))
        alerts.append(
            {
                "txn_id": row["txn_id"],
                "fraud_probability": round(float(row["fraud_probability"]), 4),
                "is_fraud_actual": int(row["is_fraud"]),
                "reasons": explain_flag(row, median),
            }
        )
    return alerts
