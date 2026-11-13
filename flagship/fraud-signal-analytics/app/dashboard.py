"""Optional Streamlit dashboard over the fraud warehouse.

Run the demo first (python run_demo.py) so outputs/fraud_warehouse.db exists,
then: streamlit run app/dashboard.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB = Path(__file__).resolve().parent.parent / "outputs" / "fraud_warehouse.db"

st.set_page_config(page_title="Fraud Signal Analytics", layout="wide")
st.title("Fraud Signal Analytics")
st.caption("Card transaction pipeline, SQL analytics and a Gradient Boosting fraud-risk model.")

if not DB.exists():
    st.warning("No warehouse yet. Run: python run_demo.py")
    st.stop()

with sqlite3.connect(DB) as conn:
    headline = pd.read_sql_query(
        "SELECT COUNT(*) AS transactions, SUM(is_fraud) AS fraud, "
        "ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct, "
        "ROUND(SUM(amount_gbp), 2) AS total_value FROM transactions",
        conn,
    ).iloc[0]
    by_channel = pd.read_sql_query(
        "SELECT channel, ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct "
        "FROM transactions GROUP BY channel ORDER BY fraud_rate_pct DESC",
        conn,
    )
    by_category = pd.read_sql_query(
        "SELECT merchant_category, ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct "
        "FROM transactions GROUP BY merchant_category ORDER BY fraud_rate_pct DESC",
        conn,
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean transactions", f"{int(headline['transactions']):,}")
c2.metric("Fraud transactions", f"{int(headline['fraud']):,}")
c3.metric("Fraud rate", f"{headline['fraud_rate_pct']}%")
c4.metric("Total value", f"£{headline['total_value']:,.0f}")

st.subheader("Fraud rate by channel")
st.bar_chart(by_channel.set_index("channel"))
st.subheader("Fraud rate by merchant category")
st.bar_chart(by_category.set_index("merchant_category"))
