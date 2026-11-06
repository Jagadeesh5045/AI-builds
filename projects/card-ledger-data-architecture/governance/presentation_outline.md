# Architecture Presentation Outline: Card Ledger Data Platform

## Slide 1: Title: Modernising the card ledger data platform
- From batch and legacy to real-time, distributed and cloud oriented
- Portfolio architecture build: models, pipelines and governance artefacts
- Presenter: Jagadeeswara Rao Padala

## Slide 2: The problem with the current estate
- Transactions flow through many systems before reaching accounts receivable
- Batch orientation delays billing, settlement and reconciliation
- No single source of truth for a transaction

## Slide 3: Business processes first
- Authorisation, clearing, posting, billing, merchant settlement
- Reconciliation as a first-class process, not an afterthought
- Data requirements derived from each process step

## Slide 4: Bounded contexts (Domain Driven Design)
- Authorisation, Ledger, Billing, Settlement, Reconciliation
- Each context owns its data and publishes domain events
- Anti-corruption layer for external scheme and merchant files

## Slide 5: The data models
- Conceptual, logical and physical models for the ledger
- PostgreSQL physical model: normalised to 3NF, documented denormalisation
- Data dictionary and naming standards govern every table

## Slide 6: Polyglot persistence, on purpose (ADR-001)
- Relational core where joins, constraints and exact totals matter
- Wide-column read models where access patterns are known and query-first
- Decision recorded as an ADR with consequences, not folklore

## Slide 7: CQRS for statement enquiry (ADR-003)
- Write model optimised for correctness, read model optimised for enquiry
- TransactionRecorded events feed an idempotent projector
- Dead letter store with reasons; nothing is dropped silently

## Slide 8: Partitioning and scale (ADR-002)
- PostgreSQL monthly range partitions on transaction_date
- Cassandra partition key: member plus period month, no hot partitions
- Retention and archival by detaching old partitions

## Slide 9: Governance and trust
- ADRs for the three load-bearing decisions
- Data standards: money in integer minor units, audit columns, key standards
- Backup and restore drill with measured restore time and totals comparison

## Slide 10: What I would do next in production
- Managed Kafka and schema registry for the event contracts
- Projection lag and dead letter alerting
- Lineage and metadata catalogue on top of the data dictionary
