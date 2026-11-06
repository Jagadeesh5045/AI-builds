"""S3 storage: telemetry written as Parquet, partitioned by domain and date.

Locally this runs against moto, an in-process mock of AWS S3, through the real
boto3 client and real Parquet bytes (pyarrow). The partitioning layout and the
read-back path are exactly what would run against a real bucket; the bucket
itself is mocked, and the README says so.
"""
from __future__ import annotations

import io
import os

import boto3
import pandas as pd

DOMAIN_COLUMNS = {
    "bookings": ["sessions", "booking_starts", "booking_completions", "revenue_gbp"],
    "adoption": ["new_installs", "registrations", "activations", "ios_installs", "android_installs"],
    "engagement": ["dau", "sessions_per_user", "search_users", "itinerary_views"],
    "app_health": ["crash_rate_pct", "anr_rate_pct", "store_rating", "app_version"],
    "quality": ["defects_opened", "defects_closed", "regression_failures"],
}


def make_client(endpoint_url: str | None = None):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-west-2"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "testing"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "testing"),
    )


def ensure_bucket(s3, bucket: str) -> None:
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket not in existing:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")},
        )


def write_partitions(s3, bucket: str, df: pd.DataFrame) -> list[str]:
    """Write one Parquet object per domain per date. Returns the keys written."""
    ensure_bucket(s3, bucket)
    keys: list[str] = []
    for domain, columns in DOMAIN_COLUMNS.items():
        part = df[["date"] + columns].copy()
        part["date"] = pd.to_datetime(part["date"])
        for date_value, day in part.groupby(part["date"].dt.date):
            buffer = io.BytesIO()
            day.to_parquet(buffer, engine="pyarrow", index=False)
            key = f"telemetry/domain={domain}/date={date_value.isoformat()}/data.parquet"
            s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            keys.append(key)
    return keys


def list_partitions(s3, bucket: str, domain: str | None = None) -> list[str]:
    prefix = f"telemetry/domain={domain}/" if domain else "telemetry/"
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
        if not page.get("IsTruncated"):
            return sorted(keys)
        token = page["NextContinuationToken"]


def read_partition(s3, bucket: str, key: str) -> pd.DataFrame:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body), engine="pyarrow")


def read_domain(s3, bucket: str, domain: str) -> pd.DataFrame:
    frames = [read_partition(s3, bucket, key) for key in list_partitions(s3, bucket, domain)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
