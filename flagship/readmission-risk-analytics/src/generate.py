"""Synthetic hospital discharge data generation.

The generator builds clinically plausible discharge records from scratch
with a hand-written risk model: readmission is *more likely* when a
patient has had prior admissions, carries more comorbidities, arrived as
an emergency, stayed longer, or was discharged with heart failure or
COPD. That structure is deliberate - it gives the downstream model
something real to learn, without copying any real patient record.

Exactly ``CORRUPT_ROWS`` deliberately damaged records are then injected,
each broken in exactly one way, so the ETL stage has a known, countable
job to do.
"""

from __future__ import annotations

import math
import random

import pandas as pd

SEED = 42
CLEAN_ROWS = 8000
CORRUPT_ROWS = 75

CONDITIONS = [
    "heart failure",
    "COPD",
    "pneumonia",
    "diabetes",
    "hip fracture",
    "stroke",
    "sepsis",
    "AMI",
]
_CONDITION_WEIGHTS = [0.16, 0.15, 0.14, 0.13, 0.10, 0.11, 0.10, 0.11]

AGE_BANDS = ["18-44", "45-64", "65-74", "75-84", "85+"]
_AGE_WEIGHTS = [0.18, 0.27, 0.24, 0.20, 0.11]

DESTINATIONS = ["home", "home with care", "care home", "rehab"]
_DESTINATION_WEIGHTS = [0.62, 0.19, 0.08, 0.11]

# Extra log-odds of a 30-day readmission, by primary condition.
_CONDITION_RISK = {
    "heart failure": 0.60,
    "COPD": 0.55,
    "sepsis": 0.35,
    "pneumonia": 0.30,
    "stroke": 0.25,
    "AMI": 0.20,
    "diabetes": 0.15,
    "hip fracture": 0.10,
}
_AGE_RISK = {"18-44": 0.0, "45-64": 0.05, "65-74": 0.15, "75-84": 0.30, "85+": 0.45}
_DESTINATION_RISK = {"home": 0.0, "rehab": 0.05, "home with care": 0.20, "care home": 0.25}

COLUMNS = [
    "discharge_id",
    "age_band",
    "sex",
    "primary_condition",
    "admission_type",
    "length_of_stay_days",
    "prior_admissions",
    "comorbidity_count",
    "discharge_destination",
    "deprivation_quintile",
    "readmitted_30d",
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _one_record(rng: random.Random, seq: int) -> dict:
    age_band = rng.choices(AGE_BANDS, weights=_AGE_WEIGHTS)[0]
    condition = rng.choices(CONDITIONS, weights=_CONDITION_WEIGHTS)[0]
    admission_type = rng.choices(["emergency", "elective"], weights=[0.68, 0.32])[0]

    # Older patients and emergency cases tend to carry more history.
    prior = rng.choices([0, 1, 2, 3, 4, 5], weights=[0.38, 0.27, 0.17, 0.10, 0.05, 0.03])[0]
    base_comorbidity = {"18-44": 1.0, "45-64": 2.0, "65-74": 3.0, "75-84": 3.6, "85+": 4.0}[age_band]
    comorbidity = max(0, min(8, int(round(rng.gauss(base_comorbidity, 1.4)))))

    stay_centre = 4.0 + (3.0 if admission_type == "emergency" else 0.0)
    if condition in ("hip fracture", "stroke"):
        stay_centre += 4.0
    if condition in ("heart failure", "sepsis"):
        stay_centre += 2.0
    los = max(1, min(30, int(round(rng.gauss(stay_centre, 3.0)))))

    destination = rng.choices(DESTINATIONS, weights=_DESTINATION_WEIGHTS)[0]
    deprivation = rng.choices([1, 2, 3, 4, 5], weights=[0.24, 0.22, 0.20, 0.18, 0.16])[0]

    logit = -4.10
    logit += 0.45 * prior
    logit += 0.22 * comorbidity
    logit += 0.50 if admission_type == "emergency" else 0.0
    logit += 0.04 * max(0, los - 5)
    logit += _CONDITION_RISK[condition]
    logit += _AGE_RISK[age_band]
    logit += _DESTINATION_RISK[destination]
    logit += 0.25 if deprivation == 1 else (0.12 if deprivation == 2 else 0.0)

    return {
        "discharge_id": f"DIS-{seq:05d}",
        "age_band": age_band,
        "sex": rng.choices(["F", "M"], weights=[0.52, 0.48])[0],
        "primary_condition": condition,
        "admission_type": admission_type,
        "length_of_stay_days": los,
        "prior_admissions": prior,
        "comorbidity_count": comorbidity,
        "discharge_destination": destination,
        "deprivation_quintile": deprivation,
        "readmitted_30d": 1 if rng.random() < _sigmoid(logit) else 0,
    }


def _corrupt(record: dict, kind: int, clean_first_id: str) -> dict:
    """Break one field of an otherwise valid record."""
    row = dict(record)
    if kind == 0:
        row["primary_condition"] = None
    elif kind == 1:
        row["discharge_destination"] = ""
    elif kind == 2:
        row["age_band"] = "120+"
    elif kind == 3:
        row["length_of_stay_days"] = -4
    elif kind == 4:
        row["length_of_stay_days"] = 999
    elif kind == 5:
        row["primary_condition"] = "UNKNOWN_CONDITION"
    elif kind == 6:
        row["discharge_id"] = clean_first_id  # duplicate of a clean record
    elif kind == 7:
        row["prior_admissions"] = -1
    elif kind == 8:
        row["prior_admissions"] = 99
    elif kind == 9:
        row["comorbidity_count"] = -2
    elif kind == 10:
        row["deprivation_quintile"] = 0
    elif kind == 11:
        row["deprivation_quintile"] = 9
    elif kind == 12:
        row["admission_type"] = "walk-in"
    elif kind == 13:
        row["readmitted_30d"] = 2
    elif kind == 14:
        row["sex"] = "X"
    return row


def generate_raw(seed: int = SEED, clean_rows: int = CLEAN_ROWS,
                 corrupt_rows: int = CORRUPT_ROWS) -> pd.DataFrame:
    """Return the raw feed: ``clean_rows`` valid records plus exactly
    ``corrupt_rows`` damaged ones, shuffled deterministically."""
    rng = random.Random(seed)
    clean = [_one_record(rng, seq) for seq in range(1, clean_rows + 1)]

    damaged = []
    kinds = 15  # the fifteen breakage patterns in _corrupt
    for i in range(corrupt_rows):
        base = _one_record(rng, clean_rows + 1 + i)
        damaged.append(_corrupt(base, i % kinds, clean[0]["discharge_id"]))

    frame = pd.DataFrame(clean + damaged, columns=COLUMNS)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
