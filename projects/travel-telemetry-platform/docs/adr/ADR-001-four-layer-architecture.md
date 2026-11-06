# ADR-001: Strict four-layer architecture

Status: Accepted

## Context
The platform centralises telemetry from many sources (app telemetry, store
metrics, QE data) into leadership dashboards. Without layer discipline, dashboard
code tends to absorb ingestion quirks and statistics, which makes every change
risky and every number hard to audit.

## Decision
Four layers with one-way dependencies:
1. Ingestion lands raw data and nothing else.
2. Analysis holds all KPI, baselining and anomaly logic as pure,
   filter-agnostic functions with no UI imports.
3. Storage persists and retrieves (S3 Parquet, JWT auth, OpenSearch query layer).
4. Visualisation (Streamlit app) renders and collects filters only. No business
   logic in the UI layer; pages pass filtered frames into analysis functions.

## Consequences
Every number on a dashboard can be traced to a tested analysis function.
Layers can be tested independently (the test suite does exactly this).
The cost is more files and explicit hand-offs between layers.
