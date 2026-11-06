# ADR-001: Relational core, wide column only where the query pattern demands it

## Status
Accepted (portfolio architecture decision, Project 1 to Project 3 evidence)

## Context
The platform must serve two different shapes of work. The ledger needs joins,
cross-entity constraints, ad hoc reconciliation queries and exact totals; its
access patterns change as new regulatory and finance questions appear. Statement
enquiry and transaction history lookups have the opposite shape: a small number
of known, high-volume queries keyed by card member and time period.

## Decision
PostgreSQL (relational) is the system of record for the Ledger, Billing and
Settlement write models. A wide-column store (Cassandra, Project 2) is used only
for read models whose access patterns are known in advance and designed
query-first. No wide-column table is a source of truth for balances.

## Consequences
Positive: financial totals stay provable with joins and constraints; modelling
changes stay cheap on the write side; the expensive denormalisation is confined
to projections that can be rebuilt by replaying events.
Negative: two storage technologies to operate; read models are eventually
consistent, so user-facing views must tolerate small projection lag; every new
wide-column query pattern means a new table by design.
