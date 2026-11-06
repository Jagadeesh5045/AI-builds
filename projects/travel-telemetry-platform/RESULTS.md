# RESULTS: verification run record

All numbers below come from code actually run in this environment on
2026-11-06. Nothing here is estimated or illustrative. The dataset is
synthetic (seed 42) and is labelled synthetic everywhere it appears.

## Environment

- Python 3.12.3, project virtualenv at .venv (NumPy 2.5.4 inside the venv).
- Setup note, recorded honestly: a first system-wide pip install failed
  because the system NumPy 1.26.4 conflicts with the installed pyarrow, which
  requires NumPy 2.0 or newer. Resolved by installing requirements.txt into
  the project virtualenv instead. No code changes were needed.
- No Docker, no Kubernetes, no OpenSearch server, no Snowflake account and
  no AWS account in this environment. See the could-not-run section.

## Commands run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py
.venv/bin/python -m pytest tests/ -q
```

## Dataset

- Source: ingestion/generate_telemetry.py, seed 42.
- Size: 180 daily rows, 21 columns, 2026-01-01 to 2026-06-29.
- Domains: Bookings, Adoption, Engagement, App Health, Quality (QE).
- REST stub source (ingestion/api_stub.json): 5 store-metric records ingested.

## Injected anomalies (ground truth)

5 points injected on purpose, recorded in the generator code:

- crash_rate_pct spikes (x4.2) on 2026-02-10, 2026-04-06, 2026-05-31.
- booking_completions drops (x0.48) on 2026-03-12, 2026-03-13.

## Detection results (pipeline run, operating threshold 5.0)

Method: rolling median plus MAD modified z-score; daily series scored with
the seasonal (weekday-matched) form, pool = same weekday over the prior
12 weeks; MAD computed with scipy.stats.median_abs_deviation.

| Metric | Injected | True positives | False positives | Missed |
| --- | --- | --- | --- | --- |
| crash_rate_pct | 3 | 3 | 1 (2026-01-25) | 0 |
| booking_completions | 2 | 2 | 3 (2026-01-24, 2026-03-10, 2026-04-04) | 0 |
| Total | 5 | 5 | 4 | 0 |

Threshold sweep from the same run data (why the operating point is 5.0):
at the classic 3.5 threshold detection also recovered all 5 injected points
with 0 missed, but with 12 false positives (crash 3, bookings 9). At 5.0 the
false positives fall to 4 with recall unchanged.

## Storage round-trip (moto mock S3, real boto3 and Parquet)

- Partitions written: 900 objects (5 domains x 180 days), layout
  telemetry/domain=<domain>/date=<yyyy-mm-dd>/data.parquet.
- Partitions listed back: 900.
- Bookings domain read back: 180 rows; the test suite also asserts the
  booking_completions total read back equals the generated total exactly.

## SQL ingestion (DuckDB plus sqlglot)

- Queries authored in Snowflake dialect (ingestion/queries_snowflake.sql),
  transpiled snowflake to duckdb with sqlglot and executed on DuckDB:
  weekly_bookings returned 27 rows, monthly_activation returned 6 rows.
- The test suite asserts the weekly completions total reconciles exactly to
  the generated dataset total.

## JWT auth

- Issue and verify: pass. Expired token: rejected. Tampered token: rejected.
  Missing JWT_SECRET environment variable: rejected with a clear error.
  Covered by tests in tests/test_storage_auth_opensearch.py.

## Test suite

26 passed, 0 failed (pytest, full suite). Breakdown:

- test_anomaly.py: 4 (ground-truth recovery for both series, tuned threshold
  behaviour, constant series flags nothing)
- test_baselines.py: 5 (hand-checked rolling medians, current point excluded,
  frequency windows, validation)
- test_kpis.py: 3 (hand-checked KPI values on a 2-row frame, domain keys)
- test_ingestion.py: 3 (generator shape and ground truth, REST stub,
  Snowflake transpilation and reconciliation)
- test_storage_auth_opensearch.py: 5 (S3 Parquet round-trip, JWT x3 counting
  issue/verify/expiry/tamper/missing-secret cases, OpenSearch query builder,
  OpenSearch search with mocked client)
- test_platform_artefacts.py: 4 (compose YAML, k8s required fields and secret
  reference, Dockerfile, ADRs present)
- test_app.py: 2 (Streamlit AppTest: entry point, and all five domain pages)

## Streamlit app (AppTest)

streamlit.testing.v1 AppTest ran app/app.py and all five pages (Bookings,
Adoption, Engagement, App Health, Anomalies) headlessly with no exceptions.
The entry point rendered its 4 KPI metrics. Session-state date filtering and
query-param page sync are exercised by those runs.

## What could NOT be run, and why

- Docker: not installed in this environment. Dockerfile and
  docker-compose.yml are authored artefacts, YAML-checked and reviewed only.
  The image was never built and the compose stack never started.
- Kubernetes: no cluster available. Manifests are authored and validated by
  parsing the YAML and asserting required fields in tests. Never applied.
- OpenSearch server: none was run. The opensearch-py client and query layer
  are implemented and unit-tested with the client mocked. No live cluster was
  contacted and no real search was executed against a server.
- Snowflake: no account. The Snowflake-dialect queries were validated by
  sqlglot transpilation to DuckDB and executed there. That validates the SQL
  dialect, not a Snowflake deployment.
- Real AWS S3: storage ran against moto, an in-process mock, through the
  real boto3 client with real Parquet bytes. No real bucket was touched.
- Live REST API: the REST ingestor consumed a local JSON file with the exact
  shape of the REST response. The parsing code is transport-independent; the
  HTTP call itself was not made.
