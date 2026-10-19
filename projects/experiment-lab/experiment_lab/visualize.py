"""Figures for the experiment report: daily trends, Bayesian posteriors, CUPED."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_daily_conversion(df, peeking: dict, path: str) -> None:
    daily = df.groupby(["day", "group"])["converted"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, color in (("control", "#64748b"), ("treatment", "#2563eb")):
        ax.plot(daily.index, daily[col] * 100, marker="o", label=col, color=color)
    for d in peeking["naive_significant_days"]:
        ax.axvline(d, color="#ef4444", linestyle="--", alpha=0.5)
    ax.set_xlabel("Experiment day")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Daily conversion by group (red dashes: naive peeking 'wins')")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bayesian_posteriors(bayes_samples: dict, path: str) -> None:
    pa, pb = bayes_samples["posterior_a"], bayes_samples["posterior_b"]
    lift = bayes_samples["lift_samples"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    xs = np.linspace(0.03, 0.10, 300)
    for samples, color, label in ((pa, "#64748b", "control"), (pb, "#2563eb", "treatment")):
        hist, bins = np.histogram(samples, bins=60, range=(0.03, 0.10), density=True)
        axes[0].plot((bins[:-1] + bins[1:]) / 2, hist, color=color, label=label)
    axes[0].set_xlabel("Conversion rate")
    axes[0].set_title("Posterior conversion rates")
    axes[0].legend()
    axes[1].hist(lift, bins=60, color="#2563eb", alpha=0.75)
    axes[1].axvline(np.percentile(lift, 2.5), color="#ef4444", linestyle="--")
    axes[1].axvline(np.percentile(lift, 97.5), color="#ef4444", linestyle="--")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Lift (percentage points)")
    axes[1].set_title("Posterior lift with 95% credible interval")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cuped_ci(results: dict, path: str) -> None:
    cu = results["cuped"]
    raw, adj = cu["aov_raw_test"], cu["aov_adjusted_test"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Raw AOV", "CUPED-adjusted AOV"]
    diffs = [raw["diff"], adj["diff"]]
    errs = [[d - t["ci_low"] for d, t in zip(diffs, (raw, adj))],
            [t["ci_high"] - d for d, t in zip(diffs, (raw, adj))]]
    ax.bar(labels, diffs, yerr=errs, capsize=8, color=["#94a3b8", "#2563eb"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("AOV lift per converter (GBP)")
    ax.set_title("CUPED shrinks the confidence interval")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
