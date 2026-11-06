"""KPI computation per domain. Pure functions on DataFrames, no I/O."""
from __future__ import annotations

import pandas as pd


def bookings_kpis(df: pd.DataFrame) -> dict:
    starts = float(df["booking_starts"].sum())
    completions = float(df["booking_completions"].sum())
    return {
        "total_sessions": int(df["sessions"].sum()),
        "booking_starts": int(starts),
        "booking_completions": int(completions),
        "booking_conversion_rate": round(completions / starts, 4) if starts else 0.0,
        "revenue_gbp": round(float(df["revenue_gbp"].sum()), 2),
        "revenue_per_completion": round(float(df["revenue_gbp"].sum()) / completions, 2) if completions else 0.0,
    }


def adoption_kpis(df: pd.DataFrame) -> dict:
    installs = float(df["new_installs"].sum())
    return {
        "new_installs": int(installs),
        "registrations": int(df["registrations"].sum()),
        "activations": int(df["activations"].sum()),
        "activation_rate": round(float(df["activations"].sum()) / installs, 4) if installs else 0.0,
        "ios_share": round(float(df["ios_installs"].sum()) / installs, 4) if installs else 0.0,
        "android_share": round(float(df["android_installs"].sum()) / installs, 4) if installs else 0.0,
    }


def engagement_kpis(df: pd.DataFrame) -> dict:
    return {
        "mean_dau": round(float(df["dau"].mean()), 1),
        "peak_dau": int(df["dau"].max()),
        "mean_sessions_per_user": round(float(df["sessions_per_user"].mean()), 3),
        "search_penetration": round(float(df["search_users"].sum()) / float(df["dau"].sum()), 4),
    }


def app_health_kpis(df: pd.DataFrame) -> dict:
    return {
        "mean_crash_rate_pct": round(float(df["crash_rate_pct"].mean()), 3),
        "max_crash_rate_pct": round(float(df["crash_rate_pct"].max()), 3),
        "mean_anr_rate_pct": round(float(df["anr_rate_pct"].mean()), 3),
        "latest_store_rating": float(df.sort_values("date")["store_rating"].iloc[-1]),
        "defect_backlog_change": int(df["defects_opened"].sum() - df["defects_closed"].sum()),
        "total_regression_failures": int(df["regression_failures"].sum()),
    }


def domain_kpis(df: pd.DataFrame) -> dict[str, dict]:
    return {
        "bookings": bookings_kpis(df),
        "adoption": adoption_kpis(df),
        "engagement": engagement_kpis(df),
        "app_health": app_health_kpis(df),
    }
