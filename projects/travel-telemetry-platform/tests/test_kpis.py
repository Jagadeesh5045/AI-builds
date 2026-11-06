import pandas as pd

from analysis.kpis import adoption_kpis, bookings_kpis, domain_kpis


def _frame():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "sessions": [100, 200],
            "booking_starts": [40, 60],
            "booking_completions": [20, 30],
            "revenue_gbp": [4000.0, 6000.0],
            "new_installs": [1000, 1000],
            "registrations": [600, 600],
            "activations": [500, 500],
            "ios_installs": [600, 400],
            "android_installs": [400, 600],
            "dau": [300, 500],
            "sessions_per_user": [1.5, 1.7],
            "search_users": [120, 200],
            "itinerary_views": [80, 100],
            "crash_rate_pct": [0.5, 1.5],
            "anr_rate_pct": [0.2, 0.4],
            "store_rating": [4.5, 4.6],
            "defects_opened": [10, 12],
            "defects_closed": [9, 11],
            "regression_failures": [2, 3],
        }
    )


def test_bookings_kpis_hand_checked():
    kpis = bookings_kpis(_frame())
    assert kpis["booking_completions"] == 50
    assert kpis["booking_conversion_rate"] == 0.5
    assert kpis["revenue_gbp"] == 10000.0
    assert kpis["revenue_per_completion"] == 200.0


def test_adoption_kpis_hand_checked():
    kpis = adoption_kpis(_frame())
    assert kpis["activation_rate"] == 0.5
    assert kpis["ios_share"] == 0.5


def test_domain_kpis_keys(telemetry):
    df, _ = telemetry
    result = domain_kpis(df)
    assert set(result) == {"bookings", "adoption", "engagement", "app_health"}
