"""Unit tests for experiment-lab statistics. Run with: pytest tests/ -q"""

import math

import numpy as np
import pytest

from experiment_lab import stats


def test_ztest_equal_groups_not_significant():
    r = stats.two_proportion_ztest(50, 1000, 50, 1000)
    assert r["p_value"] > 0.9
    assert not r["significant"]
    assert abs(r["lift_pp"]) < 1e-9


def test_ztest_clear_winner_significant():
    r = stats.two_proportion_ztest(50, 1000, 100, 1000)
    assert r["p_value"] < 0.001
    assert r["significant"]
    assert r["ci_low_pp"] > 0  # CI excludes zero


def test_ztest_ci_contains_true_diff_on_simulation():
    rng = np.random.default_rng(0)
    c = (rng.random(5000) < 0.05).sum()
    t = (rng.random(5000) < 0.07).sum()
    r = stats.two_proportion_ztest(int(c), 5000, int(t), 5000)
    assert r["ci_low_pp"] <= 2.0 <= r["ci_high_pp"]


def test_welch_t_test_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 2000)
    b = rng.normal(11, 2, 2000)
    r = stats.welch_t_test(a, b)
    assert r["significant"]
    assert 0.5 < r["diff"] < 1.5


def test_srm_check_flags_imbalanced_split():
    ok = stats.srm_check(2000, 2000)
    assert not ok["srm_detected"]
    bad = stats.srm_check(2400, 1600)
    assert bad["srm_detected"]


def test_power_larger_effect_needs_fewer_users():
    small = stats.required_n_per_group(0.05, 0.06)
    large = stats.required_n_per_group(0.05, 0.10)
    assert large < small
    assert small > 0


def test_mde_shrinks_with_sample_size():
    assert stats.min_detectable_effect(0.05, 50000) < stats.min_detectable_effect(0.05, 1000)


def test_bayesian_strong_evidence():
    r = stats.bayesian_ab(50, 1000, 120, 1000, n_samples=20000)
    assert r["prob_treatment_better"] > 0.99
    assert r["expected_lift_pp"] > 0
    assert r["cri_low_pp"] > 0


def test_bayesian_no_difference_is_uncertain():
    r = stats.bayesian_ab(50, 1000, 52, 1000, n_samples=20000)
    assert 0.3 < r["prob_treatment_better"] < 0.7


def test_cuped_reduces_variance_with_correlated_covariate():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 5000)
    y = 3 * x + rng.normal(0, 1, 5000)
    r = stats.cuped(y, x)
    assert r["variance_reduction"] > 0.5


def test_cuped_no_reduction_with_noise_covariate():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 5000)
    x = rng.normal(0, 1, 5000)
    r = stats.cuped(y, x)
    assert abs(r["variance_reduction"]) < 0.05


def test_sprt_stops_early_on_strong_effect():
    rng = np.random.default_rng(4)
    c = (rng.random(3000) < 0.05).astype(float)
    t = (rng.random(3000) < 0.10).astype(float)
    r = stats.sprt_bernoulli(c, t, p0=0.05, delta=0.05)
    assert r["decision"].startswith("reject H0")
    assert r["stopped_at_pair"] < r["total_pairs"]


def test_sprt_accepts_h0_when_no_effect():
    rng = np.random.default_rng(5)
    c = (rng.random(6000) < 0.05).astype(float)
    t = (rng.random(6000) < 0.05).astype(float)
    r = stats.sprt_bernoulli(c, t, p0=0.05, delta=0.05)
    assert r["decision"].startswith("accept H0")
