"""Synthetic AML transaction-monitoring alert generation.

The generator builds plausible alert records from scratch with a
hand-written risk model: an alert is more likely to be confirmed
suspicious when the amount is large, the customer is in a high risk
tier, the rule is structuring or a sanctions match, the customer has
had prior alerts, the channel is wire or correspondent banking, or the
account is very new. That structure is deliberate, it gives the
downstream model something real to learn, without copying any real
customer or transaction record.

Exactly ``CORRUPT_ROWS`` deliberately damaged records are then
injected, each broken in exactly one way, so the ETL stage has a
known, countable job to do.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pandas as pd

SEED = 42
CLEAN_ROWS = 10000
CORRUPT_ROWS = 75

ALERT_RULES = [
    "large_transaction",
    "structuring",
    "high_risk_country",
    "rapid_movement",
    "dormant_account",
    "sanctions_match",
]
_RULE_WEIGHTS = [0.30, 0.18, 0.16, 0.16, 0.12, 0.08]

CHANNELS = ["online", "mobile", "branch", "wire", "atm", "correspondent"]
_CHANNEL_WEIGHTS = [0.28, 0.22, 0.18, 0.14, 0.10, 0.08]

RISK_TIERS = ["low", "medium", "high"]
_TIER_WEIGHTS = [0.52, 0.33, 0.15]

_RULE_RISK = {
    "sanctions_match": 1.10,
    "structuring": 0.85,
    "high_risk_country": 0.55,
    "rapid_movement": 0.45,
    "large_transaction": 0.25,
    "dormant_account": 0.20,
}
_TIER_RISK = {"low": 0.0, "medium": 0.45, "high": 1.05}
_CHANNEL_RISK = {
    "correspondent": 0.55,
    "wire": 0.40,
    "online": 0.05,
    "mobile": 0.0,
    "branch": 0.0,
    "atm": -0.10,
}

COLUMNS = [
    "alert_id",
    "customer_id",
    "alert_date",
    "transaction_amount",
    "channel",
    "alert_rule",
    "customer_risk_tier",
    "prior_alerts",
    "account_age_days",
    "confirmed_suspicious",
    "disposition",
]

_BASE_DATE = date(2026, 1, 1)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _one_record(rng: random.Random, seq: int) -> dict:
    tier = rng.choices(RISK_TIERS, weights=_TIER_WEIGHTS)[0]
    rule = rng.choices(ALERT_RULES, weights=_RULE_WEIGHTS)[0]
    channel = rng.choices(CHANNELS, weights=_CHANNEL_WEIGHTS)[0]

    # Amounts are log-normal: most alerts are modest, a tail is large.
    # High tier customers and large-transaction rules skew larger.
    centre = 8.0
    if rule == "large_transaction":
        centre = 10.2
    if tier == "high":
        centre += 0.4
    amount = round(min(950000.0, max(15.0, rng.lognormvariate(centre, 1.05))), 2)

    prior = rng.choices([0, 1, 2, 3, 4, 5, 6], weights=[0.42, 0.26, 0.15, 0.08, 0.05, 0.03, 0.01])[0]
    if tier == "high":
        prior = min(6, prior + rng.choice([0, 0, 1, 1, 2]))
    account_age = max(5, int(rng.gauss(1400, 900)))
    account_age = min(account_age, 7300)
    alert_day = _BASE_DATE + timedelta(days=rng.randrange(0, 273))

    logit = -3.55
    logit += _RULE_RISK[rule]
    logit += _TIER_RISK[tier]
    logit += _CHANNEL_RISK[channel]
    logit += 0.38 * prior
    logit += 0.55 * max(0.0, (math.log(amount) - 9.0))
    logit += 0.45 if account_age < 180 else 0.0
    logit += 0.20 if account_age < 365 else 0.0

    confirmed = 1 if rng.random() < _sigmoid(logit) else 0
    return {
        "alert_id": f"AML-{seq:05d}",
        "customer_id": f"CUST-{rng.randrange(1, 4200):05d}",
        "alert_date": alert_day.isoformat(),
        "transaction_amount": amount,
        "channel": channel,
        "alert_rule": rule,
        "customer_risk_tier": tier,
        "prior_alerts": prior,
        "account_age_days": account_age,
        "confirmed_suspicious": confirmed,
        "disposition": "confirmed_suspicious" if confirmed else "false_positive",
    }


def _corrupt(record: dict, kind: int, clean_first_id: str) -> dict:
    """Break one field of an otherwise valid record."""
    row = dict(record)
    if kind == 0:
        row["customer_id"] = None
    elif kind == 1:
        row["customer_id"] = ""
    elif kind == 2:
        row["alert_rule"] = ""
    elif kind == 3:
        row["transaction_amount"] = -500.00
    elif kind == 4:
        row["transaction_amount"] = 0
    elif kind == 5:
        row["transaction_amount"] = 99999999.00
    elif kind == 6:
        row["alert_date"] = "2026-13-45"
    elif kind == 7:
        row["alert_date"] = "not-a-date"
    elif kind == 8:
        row["alert_id"] = clean_first_id
    elif kind == 9:
        row["prior_alerts"] = -1
    elif kind == 10:
        row["prior_alerts"] = 999
    elif kind == 11:
        row["account_age_days"] = -5
    elif kind == 12:
        row["account_age_days"] = 99999
    elif kind == 13:
        row["channel"] = "pigeon"
    elif kind == 14:
        row["customer_risk_tier"] = "extreme"
    return row


def generate_raw(seed: int = SEED, clean_rows: int = CLEAN_ROWS,
                 corrupt_rows: int = CORRUPT_ROWS) -> pd.DataFrame:
    """Return the raw feed: ``clean_rows`` valid records plus exactly
    ``corrupt_rows`` damaged ones, shuffled deterministically."""
    rng = random.Random(seed)
    clean = [_one_record(rng, seq) for seq in range(1, clean_rows + 1)]

    damaged = []
    kinds = 15
    for i in range(corrupt_rows):
        base = _one_record(rng, clean_rows + 1 + i)
        damaged.append(_corrupt(base, i % kinds, clean[0]["alert_id"]))

    frame = pd.DataFrame(clean + damaged, columns=COLUMNS)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
