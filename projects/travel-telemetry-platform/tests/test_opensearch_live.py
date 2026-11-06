"""Live OpenSearch tests: run against a real local server when reachable.

These tests are skipped cleanly when no OpenSearch server answers on
127.0.0.1:9200 (for example in CI or on a machine where the node was never
started). When the server is up they exercise the project's own client layer
(storage.opensearch_client) end to end: index creation, bulk indexing, the
domain query, the high-crash range query and the monthly aggregation.
"""
from __future__ import annotations

import pytest
from opensearchpy.helpers import bulk

from storage.opensearch_client import (
    TelemetryLogSearcher,
    ensure_index,
    make_client,
)

TEST_INDEX_DOCS = [
    {
        "_index": "telemetry-logs",
        "_id": "live-test-app-health-2026-02-10",
        "_source": {
            "@timestamp": "2026-02-10",
            "date": "2026-02-10",
            "domain": "App Health",
            "event_name": "daily_telemetry",
            "message": "App Health daily telemetry for 2026-02-10",
            "crash_rate_pct": 3.41,
            "anr_rate_pct": 0.52,
            "store_rating": 4.31,
            "app_version": "7.2.0",
        },
    },
    {
        "_index": "telemetry-logs",
        "_id": "live-test-bookings-2026-02-10",
        "_source": {
            "@timestamp": "2026-02-10",
            "date": "2026-02-10",
            "domain": "Bookings",
            "event_name": "daily_telemetry",
            "message": "Bookings daily telemetry for 2026-02-10",
            "sessions": 51234.0,
            "booking_starts": 15890.0,
            "booking_completions": 7310.0,
            "revenue_gbp": 1543210.5,
        },
    },
]


@pytest.fixture(scope="module")
def live_client():
    client = make_client(host="127.0.0.1", port=9200)
    try:
        reachable = client.ping()
    except Exception:
        reachable = False
    if not reachable:
        pytest.skip("no OpenSearch server reachable on 127.0.0.1:9200")
    ensure_index(client)
    success, errors = bulk(client, TEST_INDEX_DOCS, refresh=True, raise_on_error=False)
    assert not errors
    assert success == len(TEST_INDEX_DOCS)
    yield client
    for doc in TEST_INDEX_DOCS:
        client.delete(index="telemetry-logs", id=doc["_id"], refresh=True, ignore=[404])


def test_server_reports_version(live_client):
    info = live_client.info()
    assert info["version"]["number"]
    assert info["version"]["distribution"] == "opensearch"


def test_domain_query_finds_indexed_doc(live_client):
    searcher = TelemetryLogSearcher(live_client)
    hits = searcher.search(
        domain="Bookings", date_from="2026-02-10", date_to="2026-02-10", size=5
    )
    assert hits, "expected the indexed Bookings doc for 2026-02-10"
    assert all(hit["domain"] == "Bookings" for hit in hits)
    assert all(hit["date"] == "2026-02-10" for hit in hits)


def test_high_crash_range_query(live_client):
    searcher = TelemetryLogSearcher(live_client)
    response = searcher.search_high_crash(threshold=2.0)
    dates = {hit["_source"]["date"] for hit in response["hits"]["hits"]}
    assert "2026-02-10" in dates
    for hit in response["hits"]["hits"]:
        assert hit["_source"]["crash_rate_pct"] > 2.0


def test_monthly_average_aggregation(live_client):
    searcher = TelemetryLogSearcher(live_client)
    buckets = searcher.monthly_average()
    february = [b for b in buckets if b["month"] == "2026-02"]
    assert february, "expected a 2026-02 aggregation bucket"
    assert february[0]["doc_count"] >= 1
    assert february[0]["average"] is not None
