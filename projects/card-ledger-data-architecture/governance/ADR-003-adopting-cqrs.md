# ADR-003: Adopt CQRS for statement enquiry, not platform-wide

## Status
Accepted (implemented in Project 3)

## Context
Statement enquiry reads (member, period, full transaction list) are frequent,
read-heavy and latency sensitive, while ledger writes are low-volume,
constraint-heavy and must never be slowed by read traffic. Scaling one model to
do both forces compromises on the write side, where correctness matters most.

## Decision
Apply CQRS narrowly: the write model is the normalised transactional store; a
TransactionRecorded event per committed transaction feeds a projector that
maintains a separate, query-shaped statement-enquiry read model. The projector
is idempotent (processed event ids stored, duplicates dropped and counted) and
invalid events go to a dead letter store with reasons, never silently dropped.
All other platform queries keep using the write model directly.

## Consequences
Positive: reads scale and are shaped exactly for the enquiry screen; the read
model can be rebuilt from the event log at any time; write latency is isolated
from read load; totals parity between write and read models is continuously
verifiable.
Negative: eventual consistency between write and read; a new moving part (the
projector) needs monitoring for lag and dead letters; developers must know which
side of the split a feature belongs on.
