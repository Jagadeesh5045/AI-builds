import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""App Health domain page, including the matplotlib/seaborn figure."""
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st

from analysis.kpis import app_health_kpis
from app.common import date_filter, load_telemetry, sync_query_params

st.set_page_config(page_title="App Health", layout="wide")
st.session_state["active_page"] = "App Health"
sync_query_params()
st.title("App Health")

df = date_filter(load_telemetry())
kpis = app_health_kpis(df)
cols = st.columns(4)
cols[0].metric("Mean crash rate %", f"{kpis['mean_crash_rate_pct']:.2f}")
cols[1].metric("Max crash rate %", f"{kpis['max_crash_rate_pct']:.2f}")
cols[2].metric("Store rating", f"{kpis['latest_store_rating']:.2f}")
cols[3].metric("Regression failures", f"{kpis['total_regression_failures']:,}")

st.plotly_chart(px.line(df, x="date", y=["crash_rate_pct", "anr_rate_pct"], title="Crash and ANR rates"), use_container_width=True)

st.subheader("Metric correlations (matplotlib / seaborn)")
numeric = df[["crash_rate_pct", "anr_rate_pct", "store_rating", "dau", "booking_completions", "regression_failures"]]
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
