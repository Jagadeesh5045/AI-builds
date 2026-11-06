"""Storage round-trip on moto S3, JWT behaviour, OpenSearch query layer."""
import boto3
import pytest
from moto import mock_aws

from storage.auth import AuthError, issue_token, verify_token
from storage.opensearch_client import TelemetryLogSearcher, build_telemetry_query, make_client
from storage.s3_store import list_partitions, make_client, read_domain, write_partitions

BUCKET = "travel-telemetry-test"


def test_s3_parquet_round_trip(telemetry):
    df, _ = telemetry
    with mock_aws():
        s3 = make_client()
        keys = write_partitions(s3, BUCKET, df)
        assert len(keys) == len(df) * 5  # 5 domains, one object per domain per day
        listed = list_partitions(s3, BUCKET, "bookings")
        assert len(listed) == len(df)
        back = read_domain(s3, BUCKET, "bookings")
        assert len(back) == len(df)
        assert int(back["booking_completions"].sum()) == int(df["booking_completions"].sum())


def test_jwt_issue_verify_and_expiry(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-unit-tests-only-0123456789")
    token = issue_token("analyst@example.com", expires_minutes=5)
    claims = verify_token(token)
    assert claims["sub"] == "analyst@example.com"
    assert claims["iss"] == "travel-telemetry-platform"

    expired = issue_token("analyst@example.com", expires_minutes=-1)
    with pytest.raises(AuthError):
        verify_token(expired)

    with pytest.raises(AuthError):
        verify_token(token + "tampered")


def test_jwt_secret_required(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(AuthError):
        issue_token("nobody@example.com")


def test_opensearch_query_builder():
    body = build_telemetry_query(text="crash", domain="app_health", date_from="2026-01-01", date_to="2026-01-31")
    bool_q = body["query"]["bool"]
    assert bool_q["must"][0]["multi_match"]["query"] == "crash"
    assert {"term": {"domain.keyword": "app_health"}} in bool_q["filter"]
    empty = build_telemetry_query()
    assert empty["query"]["bool"]["must"] == [{"match_all": {}}]


def test_opensearch_search_with_mocked_transport():
    client = make_client()
    captured = {}

    def fake_search(index, body):
        captured["index"] = index
        captured["body"] = body
        return {"hits": {"hits": [{"_source": {"domain": "app_health", "message": "crash spike"}}]}}

    client.search = fake_search  # mocked client transport: no server involved
    searcher = TelemetryLogSearcher(client)
    hits = searcher.search(text="crash", domain="app_health")
    assert hits == [{"domain": "app_health", "message": "crash spike"}]
    assert captured["index"] == "telemetry-logs"
