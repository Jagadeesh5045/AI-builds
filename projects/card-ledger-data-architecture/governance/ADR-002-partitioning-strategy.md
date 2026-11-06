# ADR-002: Partitioning strategy per store, driven by the dominant query key

## Status
Accepted (portfolio architecture decision)

## Context
Transactions grow continuously and are almost always queried by time period and
by member. A single unpartitioned table degrades vacuum, index maintenance and
archival. In the wide-column read model, the partition key decides data
distribution and a wrong key creates hot partitions.

## Decision
PostgreSQL transactions table: declarative range partitioning by
transaction_date, one partition per calendar month. Hot data stays in recent
partitions; old partitions can be detached and archived without touching live
traffic. Supporting indexes are per partition, led by (member_id,
transaction_date).
Cassandra read models: partition key is (member_id, period month) so one
member-month is one partition, sized for a single statement enquiry; clustering
by transaction timestamp descending inside the partition.

## Consequences
Positive: predictable query latency for statement periods, cheap retention and
archival, no single hot partition for high-volume members because the month is
part of the key.
Negative: cross-month queries touch several partitions; partition creation must
be automated ahead of the month boundary or inserts fail; member-month sizing
must be monitored for outlier members.
