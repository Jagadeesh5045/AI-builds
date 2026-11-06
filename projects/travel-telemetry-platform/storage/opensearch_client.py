"""OpenSearch query layer for telemetry log search.

Implemented with the official opensearch-py client: this module builds real
OpenSearch query bodies and sends them through a real OpenSearch client
object. Honest scope: the query layer is unit-tested with the client
transport mocked, and it has also been run end to end against a real local
OpenSearch 2.19.1 single node (security disabled, loopback only): index
creation, 900 documents indexed, domain and range queries and a monthly
aggregation executed. See RESULTS_OPENSEARCH.md for the full record.
"""
from __future__ import annotations

from opensearchpy import OpenSearch

TELEMETRY_INDEX = "telemetry-logs"


def build_telemetry_query(
    text: str | None = None,
    domain: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    size: int = 50,
) -> dict:
    """Build a bool query body for telemetry log search."""
    must: list[dict] = []
    filters: list[dict] = []
    if text:
        must.append({"multi_match": {"query": text, "fields": ["message", "event_name", "domain"]}})
    if domain:
        filters.append({"term": {"domain.keyword": domain}})
    if date_from or date_to:
        range_filter: dict = {}
        if date_from:
            range_filter["gte"] = date_from
        if date_to:
            range_filter["lte"] = date_to
        filters.append({"range": {"@timestamp": range_filter}})
    body: dict = {"size": size, "query": {"bool": {"must": must or [{"match_all": {}}], "filter": filters}}}
    body["sort"] = [{"@timestamp": {"order": "desc"}}]
    return body


INDEX_MAPPING: dict = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "@timestamp": {"type": "date"},
            "date": {"type": "date"},
            "domain": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "event_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "message": {"type": "text"},
            "crash_rate_pct": {"type": "float"},
            "anr_rate_pct": {"type": "float"},
            "store_rating": {"type": "float"},
            "app_version": {"type": "keyword"},
        }
    },
}


def ensure_index(client: OpenSearch, index: str = TELEMETRY_INDEX) -> bool:
    """Create the telemetry index with INDEX_MAPPING if it does not exist.

    Returns True if the index was created by this call, False if it existed.
    """
    if client.indices.exists(index=index):
        return False
    client.indices.create(index=index, body=INDEX_MAPPING)
    return True


def build_high_crash_query(threshold: float, domain: str = "App Health", size: int = 100) -> dict:
    """Range query body: days in a domain whose crash rate exceeds a threshold."""
    return {
        "size": size,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"domain.keyword": domain}},
                    {"range": {"crash_rate_pct": {"gt": threshold}}},
                ]
            }
        },
        "sort": [{"crash_rate_pct": {"order": "desc"}}],
    }


def build_monthly_average_query(field: str = "crash_rate_pct", domain: str = "App Health") -> dict:
    """Aggregation body: monthly average of a metric within one domain."""
    return {
        "size": 0,
        "query": {"bool": {"filter": [{"term": {"domain.keyword": domain}}]}},
        "aggs": {
            "by_month": {
                "date_histogram": {"field": "@timestamp", "calendar_interval": "month"},
                "aggs": {f"avg_{field}": {"avg": {"field": field}}},
            }
        },
    }


def make_client(host: str = "localhost", port: int = 9200) -> OpenSearch:
    """Create a real opensearch-py client. Connecting is the caller's choice."""
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )


class TelemetryLogSearcher:
    """Thin wrapper: build the query, send it through the client, return hits."""

    def __init__(self, client: OpenSearch):
        self._client = client

    def search(self, **query_kwargs) -> list[dict]:
        body = build_telemetry_query(**query_kwargs)
        response = self._client.search(index=TELEMETRY_INDEX, body=body)
        return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]

    def search_raw(self, body: dict) -> dict:
        """Send a query body unchanged and return the full response."""
        return self._client.search(index=TELEMETRY_INDEX, body=body)

    def search_high_crash(self, threshold: float, domain: str = "App Health") -> dict:
        """Full response for days whose crash rate exceeds the threshold."""
        return self.search_raw(build_high_crash_query(threshold, domain=domain))

    def monthly_average(self, field: str = "crash_rate_pct", domain: str = "App Health") -> list[dict]:
        """Monthly average buckets for a metric: [{month, doc_count, average}]."""
        response = self.search_raw(build_monthly_average_query(field, domain=domain))
        buckets = response.get("aggregations", {}).get("by_month", {}).get("buckets", [])
        return [
            {
                "month": bucket["key_as_string"][:7],
                "doc_count": bucket["doc_count"],
                "average": bucket[f"avg_{field}"]["value"],
            }
            for bucket in buckets
        ]
