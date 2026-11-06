"""Shared data access and filter state for the Streamlit app.

UI layer rule: pages render and collect filter choices only. All KPI and
anomaly logic lives in the analysis layer; pages pass the filtered frame in
and render what comes back.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_data
def load_telemetry() -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "data" / "telemetry_daily.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        from ingestion.generate_telemetry import generate_telemetry

        df, _ = generate_telemetry()
    return df.sort_values("date").reset_index(drop=True)


def sync_query_params() -> None:
    """Persist the active domain page in the URL query params for navigation."""
    page = st.session_state.get("active_page", "Bookings")
    if st.query_params.get("page") != page:
        st.query_params["page"] = page


def date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Shared sidebar date-range filter, stored in session state."""
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    if "date_range" not in st.session_state:
        st.session_state["date_range"] = (min_date, max_date)
    chosen = st.sidebar.date_input(
        "Date range (shared across pages)",
        value=st.session_state["date_range"],
        min_value=min_date,
        max_value=max_date,
        key="date_range",
    )
    if isinstance(chosen, tuple) and len(chosen) == 2:
        start, end = pd.Timestamp(chosen[0]), pd.Timestamp(chosen[1])
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    return df
