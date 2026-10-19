"""End-to-end experiment analysis: runs every check and writes a report.

Pipeline order mirrors how a data scientist should read an experiment:
1. guardrails (sample ratio mismatch) before any peeking at outcomes
2. primary metric: conversion rate, frequentist test + confidence interval
3. peeking audit: what a naive daily dashboard would have claimed
4. CUPED: variance reduction on revenue per user via the pre-period covariate
5. Bayesian read: probability treatment is better, expected lift, credible interval
6. SPRT: the honest sequential test for early stopping
7. power context: MDE at this sample size, n needed for 80% power
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment_lab import stats


def daily_peeking_audit(df: pd.DataFrame) -> dict:
    """Re-run the primary test on cumulative data after each day.

    Shows which days a naive dashboard-peeker would have declared victory.
    """
    days, naive_sig_days = [], []
    for day in sorted(df["day"].unique()):
        sub = df[df["day"] <= day]
        c = sub[sub["group"] == "control"]
        t = sub[sub["group"] == "treatment"]
        res = stats.two_proportion_ztest(c["converted"].sum(), len(c), t["converted"].sum(), len(t))
        days.append({"day": int(day), "p_value": res["p_value"], "lift_pp": res["lift_pp"]})
        if res["significant"]:
            naive_sig_days.append(int(day))
    return {"daily": days, "naive_significant_days": naive_sig_days}


def run_full_analysis(df: pd.DataFrame) -> dict:
    """Run the full analysis suite. Returns a results dict (JSON-serialisable parts)."""
    control = df[df["group"] == "control"].reset_index(drop=True)
    treatment = df[df["group"] == "treatment"].reset_index(drop=True)

    x1, n1 = int(control["converted"].sum()), len(control)
    x2, n2 = int(treatment["converted"].sum()), len(treatment)

    results: dict = {"n_control": n1, "n_treatment": n2}

    # 1. guardrails first
    results["srm"] = stats.srm_check(n1, n2)

    # 2. primary metric: conversion rate
    results["primary"] = stats.two_proportion_ztest(x1, n1, x2, n2)

    # 3. peeking audit
    results["peeking"] = daily_peeking_audit(df)

    # 4. CUPED with the pre-period covariate.
    #    (a) revenue per user (all users): zero-inflated, covariate helps a little
    #    (b) average order value (converters only): continuous, covariate helps a lot
    cuped_c = stats.cuped(control["revenue"].to_numpy(), control["pre_spend"].to_numpy())
    cuped_t = stats.cuped(treatment["revenue"].to_numpy(), treatment["pre_spend"].to_numpy())
    raw = stats.welch_t_test(control["revenue"].to_numpy(), treatment["revenue"].to_numpy())
    adj = stats.welch_t_test(cuped_c["y_adjusted"], cuped_t["y_adjusted"])
    conv_c = control[control["converted"] == 1]
    conv_t = treatment[treatment["converted"] == 1]
    aov_raw = stats.welch_t_test(conv_c["revenue"].to_numpy(), conv_t["revenue"].to_numpy())
    aov_c = stats.cuped(conv_c["revenue"].to_numpy(), conv_c["pre_spend"].to_numpy())
    aov_t = stats.cuped(conv_t["revenue"].to_numpy(), conv_t["pre_spend"].to_numpy())
    aov_adj = stats.welch_t_test(aov_c["y_adjusted"], aov_t["y_adjusted"])
    results["cuped"] = {
        "rpu_variance_reduction_control": cuped_c["variance_reduction"],
        "rpu_variance_reduction_treatment": cuped_t["variance_reduction"],
        "raw_revenue_test": raw,
        "adjusted_revenue_test": adj,
        "aov_variance_reduction_control": aov_c["variance_reduction"],
        "aov_variance_reduction_treatment": aov_t["variance_reduction"],
        "aov_raw_test": aov_raw,
        "aov_adjusted_test": aov_adj,
        "aov_ci_width_raw": aov_raw["ci_high"] - aov_raw["ci_low"],
        "aov_ci_width_adjusted": aov_adj["ci_high"] - aov_adj["ci_low"],
    }

    # 5. Bayesian read on conversion
    bayes = stats.bayesian_ab(x1, n1, x2, n2)
    results["bayesian"] = {k: v for k, v in bayes.items() if k not in ("posterior_a", "posterior_b", "lift_samples")}
    results["_bayes_samples"] = bayes  # kept for visualisation, stripped before JSON

    # 6. SPRT sequential test on conversion outcomes, ordered by day
    c_ord = control.sort_values("day")["converted"].to_numpy()
    t_ord = treatment.sort_values("day")["converted"].to_numpy()
    sprt = stats.sprt_bernoulli(c_ord, t_ord, p0=results["primary"]["p_control"], delta=0.018)
    results["sprt"] = {k: v for k, v in sprt.items() if k != "llr_path"}
    results["_sprt_path"] = sprt["llr_path"]

    # 7. power context
    p_bar = (x1 + x2) / (n1 + n2)
    results["power"] = {
        "mde_pp": stats.min_detectable_effect(p_bar, min(n1, n2)),
        "required_n_per_group_for_observed_lift": stats.required_n_per_group(
            results["primary"]["p_control"], results["primary"]["p_treatment"]
        ),
    }
    return results


def write_report(results: dict, path: str) -> None:
    """Write a human-readable markdown report from a results dict."""
    p, srm, pk = results["primary"], results["srm"], results["peeking"]
    cu, ba, sp, pw = results["cuped"], results["bayesian"], results["sprt"], results["power"]

    verdict = (
        "SHIP IT: the lift is statistically significant at the fixed horizon."
        if p["significant"]
        else "DO NOT SHIP: no significant lift at the fixed horizon."
    )
    lines = [
        "# Experiment report: checkout redesign A/B test",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Sample",
        f"- Control: {results['n_control']:,} users, conversion {p['p_control']*100:.2f}%",
        f"- Treatment: {results['n_treatment']:,} users, conversion {p['p_treatment']*100:.2f}%",
        f"- Sample ratio mismatch check: chi2={srm['chi2']:.2f}, p={srm['p_value']:.3f} "
        f"({'FLAGGED' if srm['srm_detected'] else 'clean, randomisation looks healthy'})",
        "",
        "## Primary metric: conversion rate",
        f"- Lift: {p['lift_pp']:+.2f} percentage points ({p['lift_relative']:+.1f}% relative)",
        f"- 95% CI: [{p['ci_low_pp']:+.2f}, {p['ci_high_pp']:+.2f}] pp",
        f"- Two-proportion z-test: z={p['z']:.2f}, p={p['p_value']:.4f} "
        f"({'significant' if p['significant'] else 'not significant'})",
        "",
        "## Peeking audit (what a naive dashboard would have said)",
        f"- Days where the running p-value dipped under 0.05: {pk['naive_significant_days'] or 'none'}",
        "- Peeking inflates the false-positive rate; only the fixed-horizon test above counts.",
        "",
        "## CUPED: pre-period spend as a variance-reduction covariate",
        f"- Revenue per user: variance reduction {cu['rpu_variance_reduction_control']*100:.1f}% / "
        f"{cu['rpu_variance_reduction_treatment']*100:.1f}% (zero-inflated metric, small win, as expected)",
        f"- Average order value (converters): variance reduction "
        f"{cu['aov_variance_reduction_control']*100:.1f}% / {cu['aov_variance_reduction_treatment']*100:.1f}%",
        f"- AOV 95% CI width: GBP {cu['aov_ci_width_raw']:.2f} raw -> GBP {cu['aov_ci_width_adjusted']:.2f} CUPED-adjusted",
        f"- Adjusted AOV lift: GBP {cu['aov_adjusted_test']['diff']:+.2f} "
        f"(p={cu['aov_adjusted_test']['p_value']:.4f})",
        "",
        "## Bayesian read",
        f"- P(treatment better than control): {ba['prob_treatment_better']*100:.1f}%",
        f"- Expected lift: {ba['expected_lift_pp']:+.2f} pp, 95% credible interval "
        f"[{ba['cri_low_pp']:+.2f}, {ba['cri_high_pp']:+.2f}] pp",
        "",
        "## Sequential test (SPRT, honest early stopping)",
        f"- Decision: {sp['decision']}",
        f"- Stopped after {sp['stopped_at_pair']:,} of {sp['total_pairs']:,} observation pairs",
        "",
        "## Power context",
        f"- Minimum detectable effect at this size: {pw['mde_pp']:.2f} pp",
        f"- Sample per group needed for 80% power at the observed lift: "
        f"{pw['required_n_per_group_for_observed_lift']:,}",
        "",
        "_Generated by experiment-lab. Dataset seed 42, fully reproducible._",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
