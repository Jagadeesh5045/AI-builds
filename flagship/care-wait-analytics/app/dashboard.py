"""Optional Streamlit dashboard. Run scripts/run_demo.py first, then:
streamlit run app/dashboard.py
Reads outputs/report.json only, so the dashboard can never show a
number the pipeline did not compute.
"""
import json
from pathlib import Path

import streamlit as st

report = json.loads(
    (Path(__file__).resolve().parents[1] / "outputs" / "report.json").read_text(encoding="utf-8")
)
st.title("Care Wait Analytics (synthetic data)")
o = report["overall"]
st.metric("Referrals", o["n_referrals"])
st.metric("Median wait (days)", o["median_wait_days"])
st.metric("18-week breach rate (%)", o["breach_pct"])
st.subheader("By specialty")
st.dataframe(report["by_specialty"])
st.subheader("By referral month")
st.dataframe(report["by_month"])
st.subheader("Breach-risk model")
st.json(report["model"])
st.subheader("Briefing")
st.markdown(Path(__file__).resolve().parents[1].joinpath("outputs", "briefing.md").read_text(encoding="utf-8"))
