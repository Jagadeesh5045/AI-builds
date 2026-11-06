import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Bookings domain page."""
import plotly.express as px
import streamlit as st

from analysis.kpis import bookings_kpis
from app.common import date_filter, load_telemetry, sync_query_params

st.set_page_config(page_title="Bookings", layout="wide")
st.session_state["active_page"] = "Bookings"
sync_query_params()
st.title("Bookings")

df = date_filter(load_telemetry())
kpis = bookings_kpis(df)
cols = st.columns(4)
cols[0].metric("Sessions", f"{kpis['total_sessions']:,}")
cols[1].metric("Booking starts", f"{kpis['booking_starts']:,}")
cols[2].metric("Completions", f"{kpis['booking_completions']:,}")
cols[3].metric("Conversion rate", f"{kpis['booking_conversion_rate']:.1%}")

st.plotly_chart(px.bar(df, x="date", y="booking_completions", title="Daily booking completions"), use_container_width=True)
st.plotly_chart(px.line(df, x="date", y="revenue_gbp", title="Daily revenue (GBP)"), use_container_width=True)
