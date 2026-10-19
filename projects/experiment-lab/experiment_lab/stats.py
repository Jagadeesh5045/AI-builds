"""Core experimentation statistics.

Frequentist tests, confidence intervals, power analysis, Bayesian A/B,
CUPED variance reduction, and a sequential probability ratio test (SPRT)
for honest early stopping. All functions are pure and unit-tested.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _normal_critical(alpha: float) -> float:
    return stats.norm.ppf(1 - alpha / 2)


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Two-sided two-proportion z-test. x1/n1 = control, x2/n2 = treatment."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = _normal_critical(0.05)
    diff = p2 - p1
    return {
        "p_control": p1,
        "p_treatment": p2,
        "lift_pp": diff * 100,
        "lift_relative": (diff / p1 * 100) if p1 > 0 else float("nan"),
        "z": z,
        "p_value": p_value,
        "ci_low_pp": (diff - zc * se_unpooled) * 100,
        "ci_high_pp": (diff + zc * se_unpooled) * 100,
        "significant": p_value < 0.05,
    }


def welch_t_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's t-test for a continuous metric (e.g. revenue per user)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t, p_value = stats.ttest_ind(b, a, equal_var=False)
    diff = b.mean() - a.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    zc = _normal_critical(0.05)
    return {
        "mean_control": float(a.mean()),
        "mean_treatment": float(b.mean()),
        "diff": float(diff),
        "t": float(t),
        "p_value": float(p_value),
        "ci_low": float(diff - zc * se),
        "ci_high": float(diff + zc * se),
        "significant": bool(p_value < 0.05),
    }


def srm_check(n1: int, n2: int) -> dict:
    """Sample-ratio-mismatch check: chi-square vs the expected 50/50 split."""
    total = n1 + n2
    expected = total / 2
    chi2 = ((n1 - expected) ** 2 + (n2 - expected) ** 2) / expected
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    return {"chi2": float(chi2), "p_value": p_value, "srm_detected": bool(p_value < 0.01)}


def required_n_per_group(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Sample size per group for a two-proportion test (normal approximation)."""
    z_alpha = _normal_critical(alpha)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n /= (p2 - p1) ** 2
    return int(math.ceil(n))


def min_detectable_effect(p: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift detectable (percentage points) at given size/power."""
    z_alpha = _normal_critical(alpha)
    z_beta = stats.norm.ppf(power)
    se = math.sqrt(2 * p * (1 - p) / n_per_group)
    return (z_alpha + z_beta) * se * 100


def bayesian_ab(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    n_samples: int = 100_000,
    seed: int = 7,
) -> dict:
    """Beta-binomial Bayesian A/B: P(treatment > control), expected lift, 95% CrI."""
    rng = np.random.default_rng(seed)
    post_a = rng.beta(prior_a + x1, prior_b + n1 - x1, n_samples)
    post_b = rng.beta(prior_a + x2, prior_b + n2 - x2, n_samples)
    lift = (post_b - post_a) * 100
    return {
        "prob_treatment_better": float(np.mean(post_b > post_a)),
        "expected_lift_pp": float(lift.mean()),
        "cri_low_pp": float(np.percentile(lift, 2.5)),
        "cri_high_pp": float(np.percentile(lift, 97.5)),
        "posterior_a": post_a,
        "posterior_b": post_b,
        "lift_samples": lift,
    }


def cuped(y: np.ndarray, x: np.ndarray) -> dict:
    """CUPED: regress the metric on a pre-experiment covariate to cut variance.

    Returns theta, the adjusted metric, and the variance reduction ratio.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    var_x = x.var(ddof=1)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / var_x) if var_x > 0 else 0.0
    y_adj = y - theta * (x - x.mean())
    var_raw, var_adj = y.var(ddof=1), y_adj.var(ddof=1)
    return {
        "theta": theta,
        "y_adjusted": y_adj,
        "variance_reduction": float(1 - var_adj / var_raw) if var_raw > 0 else 0.0,
    }


def sprt_bernoulli(
    outcomes_control: np.ndarray,
    outcomes_treatment: np.ndarray,
    p0: float,
    delta: float,
    alpha: float = 0.05,
    beta: float = 0.2,
) -> dict:
    """Sequential Probability Ratio Test for a Bernoulli lift.

    H0: treatment converts at p0 (no effect). H1: it converts at p0 + delta.
    Walks the paired observations in order and stops the moment the evidence
    crosses a boundary, the honest alternative to dashboard peeking.
    """
    a = np.asarray(outcomes_control, dtype=float)
    b = np.asarray(outcomes_treatment, dtype=float)
    p1 = p0 + delta
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))

    llr, path = 0.0, []
    decision, stopped_at = "inconclusive (max n reached)", len(a)
    n_pairs = min(len(a), len(b))
    for i in range(n_pairs):
        for x, p_h1 in ((a[i], p0), (b[i], p1)):
            llr += x * math.log(p_h1 / p0) + (1 - x) * math.log((1 - p_h1) / (1 - p0))
        path.append(llr)
        if llr >= upper:
            decision, stopped_at = "reject H0 (significant lift)", i + 1
            break
        if llr <= lower:
            decision, stopped_at = "accept H0 (no lift)", i + 1
            break
    return {
        "decision": decision,
        "stopped_at_pair": int(stopped_at),
        "total_pairs": int(n_pairs),
        "upper_boundary": upper,
        "lower_boundary": lower,
        "final_llr": float(llr),
        "llr_path": np.array(path),
    }
