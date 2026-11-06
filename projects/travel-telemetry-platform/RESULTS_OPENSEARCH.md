# RESULTS: OpenSearch live run (follow-up verification, 2026-11-06)

This file records a follow-up run that upgraded the OpenSearch part of this
project from authored-only to really-run evidence. Every number below comes
from responses of a real OpenSearch server run locally in this environment.
The dataset is the project's synthetic telemetry (seed 42), labelled
synthetic everywhere.

## Server

- Distribution: official OpenSearch Linux x64 tarball,
  opensearch-2.19.1-linux-x64.tar.gz (920 MB) from artifacts.opensearch.org
  (releases/bundle/opensearch/2.19.1), extracted into
  .runtime-opensearch/ inside this project.
- Version, read from the server root response: OpenSearch 2.19.1,
  distribution "opensearch", build type tar, Lucene 9.12.1.
- The tarball bundles its own JDK, which is what ran; no system Java was used.
- Node configuration (config/opensearch.yml of the extracted distribution):
  cluster.name travel-telemetry-local, node.name os-node-1,
  discovery.type single-node, plugins.security.disabled true,
  network.host 127.0.0.1, http.port 9200, heap fixed at 512 MB
  (-Xms512m -Xmx512m in config/jvm.options), 1 shard and 0 replicas on the
  project index. Security was disabled for this local-only, loopback-bound
  node; that is never a production setting.
- Cluster health reached GREEN with 1 node and 1 data node.
- Environment workarounds, recorded honestly: OpenSearch refuses to run as
  root, and chown is not permitted in this environment, so the node ran as
  user nobody with directory permissions opened (chmod) on the runtime,
  data, logs and distribution config folders instead of ownership changes.
  The first launch attempt also failed because the distribution's own logs
  folder was not writable for the GC log; making it writable fixed startup.

## Indexing

- Index telemetry-logs created through the project's own code
  (storage.opensearch_client.ensure_index with INDEX_MAPPING: domain as text
  with a keyword subfield, @timestamp and date as dates, crash_rate_pct and
  anr_rate_pct as floats, app_version as keyword).
- Documents: one document per domain per day built by
  run_opensearch_verification.py from ingestion.generate_telemetry
  (5 domains x 180 days), indexed with the opensearch-py bulk helper through
  the project's client (storage.opensearch_client.make_client).
- Prepared: 900. Bulk success: 900. Bulk errors: 0.
- Read back from the server: GET telemetry-logs/_count returned 900.

## Live queries (all through storage.opensearch_client code)

1. Domain query (the project's build_telemetry_query plus
   TelemetryLogSearcher, term filter on domain.keyword = "Bookings"):
   total hits 180. A sample search with size 5 returned 5 documents.
2. Range query (TelemetryLogSearcher.search_high_crash, range filter
   crash_rate_pct greater than 2.0 within App Health): total hits 3.
   The returned dates are exactly the three injected crash-spike days in the
   generator ground truth: 2026-02-10, 2026-04-06, 2026-05-31. No other day
   in the 180 days exceeds 2.0 percent crash rate.
3. Aggregation (TelemetryLogSearcher.monthly_average, date_histogram by
   month with an avg sub-aggregation on crash_rate_pct, App Health only):

   | Month | App Health docs | Average crash_rate_pct |
   | --- | --- | --- |
   | 2026-01 | 31 | 0.8220 |
   | 2026-02 | 28 | 0.9062 |
   | 2026-03 | 31 | 0.8040 |
   | 2026-04 | 30 | 0.8703 |
   | 2026-05 | 31 | 0.9439 |
   | 2026-06 | 29 | 0.8476 |

   (Averages rounded to 4 decimal places from the server response values.)

## Tests

- New live test module tests/test_opensearch_live.py: 4 tests that exercise
  the project's client layer against the real server (server version,
  domain query with date filter, high-crash range query, monthly average
  aggregation). With the server running: 4 passed.
- Full project suite with the server running: 30 passed, 0 failed
  (the original 26 plus the 4 live tests).
- With the server stopped, the same live module skips cleanly: 4 skipped,
  0 failed.
- One honest correction during this work: the first version of the domain
  query test asserted a February date inside a default size-50 search, but
  the project's searcher sorts by @timestamp descending, so a default page
  holds only the newest 50 days and the assertion failed (1 failed,
  29 passed in that full-suite run). The test was fixed to use the project's
  own date-range filter instead of weakening the project's search behaviour,
  after which everything passed as recorded above.

## Shutdown and what remains

- The node was stopped cleanly at the end of the run (SIGTERM; the node log
  records "stopped" then "closed", and port 9200 no longer answers).
- The runtime stays in .runtime-opensearch/ inside this project (tarball,
  extracted distribution, data and logs). It is local verification state and
  is excluded from any future GitHub push.
- Limitations: single local node only, security disabled, tiny synthetic
  dataset. Nothing here evidences cluster sizing, replication, security
  configuration or production operation of OpenSearch.
