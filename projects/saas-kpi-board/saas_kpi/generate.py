"""Seeded synthetic SaaS subscription data.

The generator simulates a B2B SaaS product month by month. Every customer
arrives through an acquisition channel, starts on a plan, may upgrade, and
may churn. Churn risk depends on plan and channel, which is what makes the
downstream KPI work interesting: cheap channels are not cheap if their
customers leave in month two.

Data is synthetic and seeded, so every number in the report is reproducible.
It is shaped like the exports an analyst actually gets from a billing system:
one customer table, one monthly subscription snapshot, one invoice per
active customer per month.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

PLANS = {"Starter": 29.0, "Growth": 79.0, "Scale": 199.0}
PLAN_ORDER = ["Starter", "Growth", "Scale"]
PLAN_WEIGHTS = [0.52, 0.34, 0.14]
# Monthly churn probability by plan, before the channel multiplier.
PLAN_CHURN = {"Starter": 0.055, "Growth": 0.038, "Scale": 0.022}

CHANNELS = ["Organic", "Referral", "Paid Search", "Partner", "Outbound"]
CHANNEL_WEIGHTS = [0.30, 0.15, 0.30, 0.15, 0.10]
CHANNEL_CAC = {
    "Organic": 45.0,
    "Referral": 30.0,
    "Paid Search": 140.0,
    "Partner": 95.0,
    "Outbound": 180.0,
}
CHANNEL_CHURN_MULT = {
    "Organic": 1.00,
    "Referral": 0.80,
    "Paid Search": 1.25,
    "Partner": 0.90,
    "Outbound": 1.20,
}
COUNTRIES = ["United Kingdom", "Ireland", "Germany", "Netherlands", "France", "United States"]
COUNTRY_WEIGHTS = [0.55, 0.08, 0.12, 0.09, 0.06, 0.10]

MONTHS = [f"{y}-{m:02d}" for y, m in [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 7)]]
GROSS_MARGIN = 0.72


def month_add(month: str, offset: int) -> str:
    y, m = int(month[:4]), int(month[5:7])
    idx = (y * 12 + (m - 1)) + offset
    return f"{idx // 12}-{idx % 12 + 1:02d}"


@dataclass
class Customer:
    customer_id: int
    signup_month: str
    channel: str
    country: str
    plan: str
    cac: float
    churn_month: str | None = None  # first month NOT active; None = still active
    plan_history: list = field(default_factory=list)  # (from_month, plan)


@dataclass
class Dataset:
    customers: list
    snapshots: list  # dicts: customer_id, month, plan, mrr
    invoices: list   # dicts: customer_id, month, amount, paid


def generate(seed: int = 42) -> Dataset:
    rng = random.Random(seed)
    customers: list[Customer] = []
    snapshots: list[dict] = []
    invoices: list[dict] = []
    active: list[Customer] = []
    next_id = 1

    for i, month in enumerate(MONTHS):
        # Signups grow about 4% a month with noise, weekday-style wobble included.
        expected = 30 * (1.04 ** i)
        new_count = max(8, int(round(rng.gauss(expected, expected * 0.18))))
        for _ in range(new_count):
            plan = rng.choices(PLAN_ORDER, weights=PLAN_WEIGHTS)[0]
            channel = rng.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0]
            cust = Customer(
                customer_id=next_id,
                signup_month=month,
                channel=channel,
                country=rng.choices(COUNTRIES, weights=COUNTRY_WEIGHTS)[0],
                plan=plan,
                cac=CHANNEL_CAC[channel] * rng.uniform(0.85, 1.15),
                plan_history=[(month, plan)],
            )
            next_id += 1
            customers.append(cust)
            active.append(cust)

        survivors: list[Customer] = []
        for cust in active:
            # Snapshot first: everyone active at the start of the month is billed.
            mrr = PLANS[cust.plan]
            snapshots.append({
                "customer_id": cust.customer_id, "month": month,
                "plan": cust.plan, "mrr": mrr,
            })
            invoices.append({
                "customer_id": cust.customer_id, "month": month,
                "amount": mrr, "paid": 0 if rng.random() < 0.02 else 1,
            })
            # Tenure protects a little: month-one churn is the worst.
            months_active = MONTHS.index(month) - MONTHS.index(cust.signup_month) + 1
            tenure_factor = 1.0 if months_active <= 1 else (0.85 if months_active <= 3 else 0.7)
            p_churn = PLAN_CHURN[cust.plan] * CHANNEL_CHURN_MULT[cust.channel] * tenure_factor
            if rng.random() < p_churn:
                cust.churn_month = month_add(month, 1)
            else:
                # Upgrades happen at month end and apply from next month.
                if cust.plan == "Starter" and rng.random() < 0.03:
                    cust.plan = "Growth"
                    cust.plan_history.append((month_add(month, 1), "Growth"))
                elif cust.plan == "Growth" and rng.random() < 0.02:
                    cust.plan = "Scale"
                    cust.plan_history.append((month_add(month, 1), "Scale"))
                survivors.append(cust)
        active = survivors

    return Dataset(customers=customers, snapshots=snapshots, invoices=invoices)
