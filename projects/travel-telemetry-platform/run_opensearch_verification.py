"""Index the synthetic telemetry dataset into a REAL local OpenSearch server
and run live queries through the project's own client layer
(storage.opensearch_client).

This is the live verification companion to RESULTS_OPENSEARCH.md. It expects a
single-node OpenSearch with security disabled on 127.0.0.1:9200 (see
RESULTS_OPENSEARCH.md for how the node was run). It prints a JSON summary of
every number it observed from server responses; nothing is estimated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from opensearchpy.helpers import bulk

from ingestion.generate_telemetry import DOMAINS, generate_telemetry
from storage.opensearch_client import (
    TELEMETRY_INDEX,
    TelemetryLogSearcher,
    build_telemetry_query,
    ensure_index,
    make_client,
)

DOMAIN_LABELS = {
    "bookings": "Bookings",
    "adoption": "Adoption",
    "engagement": "Engagement",
    "app_health": "App Health",
    "quality": "QE",
}
CRASH_THRESHOLD = 2.0


def build_documents() -> list[dict]:
    """One document per domain per day: 5 domains x 180 days = 900 documents."""
    frame, _ground_truth = generate_telemetry()
    documents: list[dict] = []
    for _, row in frame.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        for domain_key, fields in DOMAINS.items():
            label = DOMAIN_LABELS[domain_key]
            doc = {
                "@timestamp": date,
                "date": date,
                "domain": label,
                "event_name": "daily_telemetry",
                "message": f"{label} daily telemetry for {date}",
            }
            for field in fields:
                value = row[field]
                doc[field] = str(value) if field == "app_version" else float(value)
            documents.append(
                {"_index": TELEMETRY_INDEX, "_id": f"{domain_key}-{date}", "_source": doc}
            )
    return documents


def main() -> int:
    client = make_client(host="127.0.0.1", port=9200)
    if not client.ping():
        print(json.dumps({"error": "server did not answer ping on 127.0.0.1:9200"}))
        return 1

    info = client.info()
    summary: dict = {
        "server_name": info.get("name"),
        "cluster_name": info.get("cluster_name"),
        "version": info.get("version", {}).get("number"),
        "distribution": info.get("version", {}).get("distribution"),
    }

    summary["index_created"] = ensure_index(client)

    documents = build_documents()
    summary["documents_prepared"] = len(documents)
    success, errors = bulk(client, documents, refresh=True, raise_on_error=False)
    summary["bulk_success"] = success
    summary["bulk_errors"] = len(errors)

    count = client.count(index=TELEMETRY_INDEX)
    summary["server_doc_count"] = count["count"]

    searcher = TelemetryLogSearcher(client)

    bookings_raw = searcher.search_raw(build_telemetry_query(domain="Bookings", size=0))
    summary["query_domain_bookings_total_hits"] = bookings_raw["hits"]["total"]["value"]
    bookings_sample = searcher.search(domain="Bookings", size=5)
    summary["query_domain_bookings_sample_returned"] = len(bookings_sample)

    crash_raw = searcher.search_high_crash(CRASH_THRESHOLD)
    summary["query_high_crash_threshold"] = CRASH_THRESHOLD
    summary["query_high_crash_total_hits"] = crash_raw["hits"]["total"]["value"]
    summary["query_high_crash_dates"] = sorted(
        hit["_source"]["date"] for hit in crash_raw["hits"]["hits"]
    )

    summary["agg_monthly_avg_crash_rate"] = searcher.monthly_average()

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
