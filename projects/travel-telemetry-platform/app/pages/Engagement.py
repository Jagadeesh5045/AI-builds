import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Engagement domain page."""
import plotly.express as px
import streamlit as st

from analysis.kpis import engagement_kpis
from app.common import date_filter, load_telemetry, sync_query_params

st.set_page_config(page_title="Engagement", layout="wide")
st.session_state["active_page"] = "Engagement"
sync_query_params()
st.title("Engagement")

df = date_filter(load_telemetry())
kpis = engagement_kpis(df)
cols = st.columns(3)
cols[0].metric("Mean DAU", f"{kpis['mean_dau']:,.0f}")
cols[1].metric("Peak DAU", f"{kpis['peak_dau']:,}")
cols[2].metric("Sessions per user", f"{kpis['mean_sessions_per_user']:.2f}")

st.plotly_chart(px.line(df, x="date", y="dau", title="Daily active users"), use_container_width=True)
st.plotly_chart(px.line(df, x="date", y=["search_users", "itinerary_views"], title="Feature usage"), use_container_width=True)
