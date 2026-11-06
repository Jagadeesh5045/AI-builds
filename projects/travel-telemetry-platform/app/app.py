import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Travel Telemetry Platform: leadership dashboard entry point.

SYNTHETIC DATA: every number in this app comes from the seeded synthetic
telemetry generator. It demonstrates the platform; it is not real app data.
"""
import plotly.express as px
import streamlit as st

from analysis.kpis import domain_kpis
from app.common import date_filter, load_telemetry, sync_query_params

st.set_page_config(page_title="Travel Telemetry Platform", layout="wide")
st.session_state["active_page"] = "Bookings"
sync_query_params()

st.title("Travel Telemetry Platform")
st.caption("Production intelligence for the travel mobile app. Synthetic demonstration data.")

df = date_filter(load_telemetry())
kpis = domain_kpis(df)

cols = st.columns(4)
cols[0].metric("Booking completions", f"{kpis['bookings']['booking_completions']:,}")
cols[1].metric("Revenue (GBP)", f"{kpis['bookings']['revenue_gbp']:,.0f}")
cols[2].metric("Mean DAU", f"{kpis['engagement']['mean_dau']:,.0f}")
cols[3].metric("Mean crash rate %", f"{kpis['app_health']['mean_crash_rate_pct']:.2f}")

st.subheader("Bookings and revenue trend")
fig = px.line(df, x="date", y=["booking_completions", "booking_starts"], labels={"value": "count", "variable": "series"})
st.plotly_chart(fig, use_container_width=True)
st.page_link("pages/Anomalies.py", label="View detected anomalies")
