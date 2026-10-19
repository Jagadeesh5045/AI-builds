"""Entry point: generate data, run the full analysis, write report + figures.

Usage:
    python run_analysis.py
"""

from __future__ import annotations

import os

from experiment_lab import analysis, generate, visualize

BASE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)

    df = generate.generate_experiment()
    data_path = os.path.join(BASE, "data", "experiment.csv")
    generate.save_experiment(df, data_path)
    print(f"dataset: {len(df):,} users -> {data_path}")

    results = analysis.run_full_analysis(df)
    p = results["primary"]
    print(f"control {p['p_control']*100:.2f}% vs treatment {p['p_treatment']*100:.2f}%")
    print(f"lift {p['lift_pp']:+.2f}pp, 95% CI [{p['ci_low_pp']:+.2f}, {p['ci_high_pp']:+.2f}], p={p['p_value']:.4f}")
    print(f"bayesian P(treatment better) = {results['bayesian']['prob_treatment_better']*100:.1f}%")
    print(f"CUPED variance reduction (revenue/user): {results['cuped']['rpu_variance_reduction_control']*100:.1f}% / "
          f"{results['cuped']['rpu_variance_reduction_treatment']*100:.1f}%")
    print(f"CUPED variance reduction (AOV): {results['cuped']['aov_variance_reduction_control']*100:.1f}% / "
          f"{results['cuped']['aov_variance_reduction_treatment']*100:.1f}%")
    print(f"SPRT: {results['sprt']['decision']} at pair {results['sprt']['stopped_at_pair']:,}")
    print(f"peeking audit: naive-significant days = {results['peeking']['naive_significant_days']}")

    report_path = os.path.join(BASE, "report.md")
    analysis.write_report(results, report_path)
    print(f"report -> {report_path}")

    visualize.plot_daily_conversion(df, results["peeking"], os.path.join(BASE, "figures", "daily_conversion.png"))
    visualize.plot_bayesian_posteriors(results["_bayes_samples"], os.path.join(BASE, "figures", "bayesian_posteriors.png"))
    visualize.plot_cuped_ci(results, os.path.join(BASE, "figures", "cuped_ci.png"))
    print("figures -> figures/")


if __name__ == "__main__":
    main()
