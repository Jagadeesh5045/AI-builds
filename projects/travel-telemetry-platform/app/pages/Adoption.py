import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Adoption domain page."""
import plotly.express as px
import streamlit as st

from analysis.kpis import adoption_kpis
from app.common import date_filter, load_telemetry, sync_query_params

st.set_page_config(page_title="Adoption", layout="wide")
st.session_state["active_page"] = "Adoption"
sync_query_params()
st.title("Adoption")

df = date_filter(load_telemetry())
kpis = adoption_kpis(df)
cols = st.columns(4)
cols[0].metric("New installs", f"{kpis['new_installs']:,}")
cols[1].metric("Activations", f"{kpis['activations']:,}")
cols[2].metric("Activation rate", f"{kpis['activation_rate']:.1%}")
cols[3].metric("iOS share", f"{kpis['ios_share']:.1%}")

st.plotly_chart(
    px.area(df, x="date", y=["ios_installs", "android_installs"], title="Installs by platform"),
    use_container_width=True,
)
st.plotly_chart(px.line(df, x="date", y=["registrations", "activations"], title="Registration to activation"), use_container_width=True)
