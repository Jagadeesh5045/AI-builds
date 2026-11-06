"""OpenSearch query layer for telemetry log search.

Implemented with the official opensearch-py client: this module builds real
OpenSearch query bodies and sends them through a real OpenSearch client
object. Honest scope: no OpenSearch server was run in this environment, so
the client and query layer are unit-tested with the client transport mocked.
Nothing here has touched a live cluster.
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
