from pathlib import Path

from ingestion.generate_telemetry import DOMAINS, generate_telemetry
from ingestion.rest_ingestor import fetch_store_metrics
from ingestion.sql_ingestor import run_kpi_queries, transpile_snowflake_queries

HERE = Path(__file__).resolve().parent.parent / "ingestion"


def test_generator_shape_and_ground_truth():
    df, truth = generate_telemetry()
    assert len(df) == 180
    all_columns = {c for cols in DOMAINS.values() for c in cols}
    assert all_columns.issubset(set(df.columns))
    assert sum(len(v) for v in truth.values()) == 5  # 3 crash spikes + 2 booking drops


def test_rest_stub_ingestor():
    df = fetch_store_metrics(HERE / "api_stub.json")
    assert len(df) == 5
    assert "negative_review_share" in df.columns


def test_snowflake_queries_transpile_and_run(telemetry):
    df, _ = telemetry
    transpiled = transpile_snowflake_queries()
    assert set(transpiled) == {"weekly_bookings", "monthly_activation"}
    assert all("snowflake" not in sql.lower() for sql in transpiled.values())
    results = run_kpi_queries(df)
    assert len(results["monthly_activation"]) == 6  # Jan to Jun 2026
    assert len(results["weekly_bookings"]) >= 25
    total_completions = int(results["weekly_bookings"]["completions"].sum())
    assert total_completions == int(df["booking_completions"].sum())
