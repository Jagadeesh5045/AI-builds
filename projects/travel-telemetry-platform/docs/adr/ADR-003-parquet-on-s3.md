# ADR-003: Parquet on S3, partitioned by domain and date, for storage

Status: Accepted

## Context
Telemetry arrives daily from several domains and is read back in domain-sized
slices by dashboards and by future analysis jobs. A single monolithic file
forces full scans; a database would couple storage to one query engine.

## Decision
Store telemetry as Parquet files in S3 under
telemetry/domain=<domain>/date=<yyyy-mm-dd>/data.parquet. Parquet gives column
pruning and compression; the partition layout lets readers fetch exactly one
domain and date range. Access goes through boto3 so the same code runs against
a real bucket or the moto mock used locally.

## Consequences
Read-back is partition-scoped and engine-agnostic (pandas, DuckDB and Spark can
all read the layout). The trade-off is many small objects, acceptable at daily
grain; a compaction job would be the next step at hourly grain.
