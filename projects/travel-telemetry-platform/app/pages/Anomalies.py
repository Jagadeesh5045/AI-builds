import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Anomalies page: rolling median plus MAD modified z-score flags."""
import plotly.express as px
import streamlit as st

from analysis.anomaly import detect_anomalies
from app.common import date_filter, load_telemetry, sync_query_params

st.set_page_config(page_title="Anomalies", layout="wide")
st.session_state["active_page"] = "Anomalies"
sync_query_params()
st.title("Anomalies")

df = date_filter(load_telemetry())
threshold = st.slider("Modified z-score threshold", min_value=2.5, max_value=6.0, value=3.5, step=0.25)
st.session_state["anomaly_threshold"] = threshold

metric = st.selectbox("Metric", ["crash_rate_pct", "booking_completions", "dau", "revenue_gbp", "anr_rate_pct"])
flagged = detect_anomalies(load_telemetry(), metric, threshold=threshold)

st.metric(f"Flagged points in full series: {metric}", len(flagged))
st.dataframe(flagged, use_container_width=True)

full = load_telemetry()
fig = px.line(full, x="date", y=metric, title=f"{metric} with flagged anomalies")
if not flagged.empty:
    fig.add_scatter(x=flagged["date"], y=flagged["value"], mode="markers", marker={"color": "red", "size": 10}, name="flagged")
st.plotly_chart(fig, use_container_width=True)
