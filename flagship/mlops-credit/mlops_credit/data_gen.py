"""Synthetic credit dataset generator.

Seeded and reproducible: the same seed always produces the same CSV.
The default label is drawn from a logistic model over the applicant
features, so the signal is learnable but not trivially separable.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "income",
    "loan_amount",
    "credit_history_years",
    "employment_years",
    "debt_to_income",
    "late_payments",
    "age_band",
]

AGE_BANDS = ["18-25", "26-35", "36-50", "51+"]


def generate_credit_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    income = np.clip(rng.lognormal(mean=10.6, sigma=0.55, size=n_samples), 12000, 180000)
    loan_amount = np.clip(rng.lognormal(mean=9.6, sigma=0.7, size=n_samples), 1000, 90000)
    credit_history_years = np.clip(rng.normal(9.0, 5.0, size=n_samples), 0, 35)
    employment_years = np.clip(rng.normal(6.0, 4.5, size=n_samples), 0, 40)
    debt_to_income = np.clip(rng.beta(2.0, 5.0, size=n_samples) * 0.9, 0.02, 0.9)
    late_payments = np.clip(rng.poisson(0.8, size=n_samples), 0, 12)
    age_band = rng.choice(AGE_BANDS, size=n_samples, p=[0.18, 0.34, 0.32, 0.16])

    # Log-odds of default: higher debt burden, bigger loans relative to
    # income and more late payments raise risk; longer history and stable
    # employment lower it.
    logit = (
        -2.1
        + 1.9 * debt_to_income
        + 0.9 * (loan_amount / income)
        + 0.45 * late_payments
        - 0.06 * credit_history_years
        - 0.05 * employment_years
        + np.where(age_band == "18-25", 0.35, 0.0)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    default = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "income": np.round(income, 2),
            "loan_amount": np.round(loan_amount, 2),
            "credit_history_years": np.round(credit_history_years, 1),
            "employment_years": np.round(employment_years, 1),
            "debt_to_income": np.round(debt_to_income, 3),
            "late_payments": late_payments,
            "age_band": age_band,
            "default": default,
        }
    )


def make_drifted_batch(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Simulate a drifted production batch: incomes fall and late payments rise.

    This mimics an economic downturn reaching the scoring population.
    """
    rng = np.random.default_rng(seed)
    drifted = df.copy()
    drifted["income"] = np.round(drifted["income"] * 0.70, 2)
    drifted["late_payments"] = np.clip(
        drifted["late_payments"] + rng.poisson(2.0, size=len(drifted)), 0, 15
    )
    return drifted
