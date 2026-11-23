"""Deterministic synthetic hospital readmission dataset.

The generator uses a known ground-truth risk function so the trained model
can be judged honestly. The ground truth contains two effects a plain
linear model cannot represent exactly: an interaction between prior
admissions and comorbidity count, and an extra risk step after age 75.
That mismatch is deliberate. It is what creates real miscalibration in
subgroups, which the calibration and slice stages then have to find.

Data is synthetic and generated in code. No patient data is used.
"""

import numpy as np

FEATURES = [
    "age",
    "prior_admissions",
    "length_of_stay_days",
    "comorbidity_count",
    "medication_count",
    "follow_up_booked",
    "weekend_discharge",
]


def generate_readmissions(n=3000, seed=42):
    """Return (X, y, feature_names) with X as a float array of raw features."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(64, 15, n), 18, 97)
    prior = rng.poisson(0.9, n).astype(float)
    los = np.clip(rng.gamma(2.2, 2.4, n), 1, 30)
    comorb = rng.poisson(2.0, n).astype(float)
    meds = np.clip(np.round(comorb * 1.8 + rng.normal(3.0, 1.6, n)), 0, 25)
    follow = (rng.random(n) < 0.62).astype(float)
    weekend = (rng.random(n) < 0.28).astype(float)

    logit = (
        -3.35
        + 0.022 * (age - 60)
        + 0.52 * prior
        + 0.10 * los
        + 0.24 * comorb
        + 0.05 * meds
        - 0.85 * follow
        + 0.30 * weekend
        + 0.16 * prior * comorb
        + 0.55 * (age > 75)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = np.column_stack([age, prior, los, comorb, meds, follow, weekend])
    return X, y, list(FEATURES)


def stratified_split(y, seed=7, fracs=(0.6, 0.2, 0.2)):
    """Deterministic stratified train/val/test index split."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        a = int(round(n * fracs[0]))
        b = a + int(round(n * fracs[1]))
        parts[0].append(idx[:a])
        parts[1].append(idx[a:b])
        parts[2].append(idx[b:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)
