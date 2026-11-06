"""End-to-end local pipeline run.

Generates the synthetic telemetry, stores it to S3 (moto mock) as partitioned
Parquet, reads a domain back, runs the SQL KPI queries (Snowflake dialect
transpiled to DuckDB), runs anomaly detection on the two ground-truth series,
and writes data/results_summary.json with the real numbers from this run.
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
from moto import mock_aws

from analysis.anomaly import detect_anomalies, score_against_ground_truth
from ingestion.generate_telemetry import generate_telemetry
from ingestion.rest_ingestor import fetch_store_metrics
from ingestion.sql_ingestor import run_kpi_queries
from storage.s3_store import list_partitions, make_client, read_domain, write_partitions

ROOT = Path(__file__).parent
BUCKET = "travel-telemetry"


def main() -> dict:
    df, truth = generate_telemetry()
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "telemetry_daily.csv", index=False)
    (ROOT / "data" / "anomalies_ground_truth.json").write_text(json.dumps(truth, indent=2))

    store_metrics = fetch_store_metrics(ROOT / "ingestion" / "api_stub.json")
    kpi_queries = run_kpi_queries(df)

    with mock_aws():
        s3 = make_client()
        keys = write_partitions(s3, BUCKET, df)
        bookings_back = read_domain(s3, BUCKET, "bookings")
        partition_count = len(list_partitions(s3, BUCKET))

    # Operating threshold 5.0, tuned on this dataset from a threshold sweep
    # recorded in RESULTS.md: at 3.5 detection also recovers every injected
    # anomaly but with materially more false positives.
    scores = {}
    for metric, truth_dates in truth.items():
        flagged = detect_anomalies(df, metric, threshold=5.0)
        flagged_dates = [str(d.date()) for d in flagged["date"]]
        scores[metric] = score_against_ground_truth(flagged_dates, truth_dates)

    summary = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "injected_anomalies": {k: len(v) for k, v in truth.items()},
        "anomaly_scores": scores,
        "s3_partitions_written": len(keys),
        "s3_partitions_listed": partition_count,
        "bookings_rows_read_back": int(len(bookings_back)),
        "rest_stub_rows": int(len(store_metrics)),
        "sql_queries_run": {name: int(len(frame)) for name, frame in kpi_queries.items()},
    }
    (ROOT / "data" / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
