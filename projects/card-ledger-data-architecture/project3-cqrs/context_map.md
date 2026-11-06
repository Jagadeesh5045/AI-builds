# Bounded Context Map: Card Ledger Data Platform

## Bounded contexts and responsibilities

### Authorisation
Owns the real-time decision on whether a card transaction may proceed: card status
checks, available credit, velocity and limit rules. Owns the `authorisations` data.
Does not own settled money movement; an approved authorisation is a promise, not a
posting.

### Ledger
Owns the double-entry record of actual money movement. Every posted transaction
creates balanced ledger entries. The Ledger is the source of truth for balances and
is append-only: corrections are new compensating entries, never edits.

### Billing
Owns the statement cycle: period opening and closing, statement generation
(statement_id, member_id, period_start, period_end, opening_balance,
closing_balance, due_date), minimum payment and due date calculation. Consumes
Ledger postings; does not create them.

### Settlement
Owns clearing and settlement with merchants and schemes: which posted transactions
are due to which merchant, settlement batches and settlement status on transactions.

### Reconciliation
Owns matching across contexts and against external files: authorisation vs posting,
posting vs settlement, ledger totals vs statement totals. Raises
`reconciliation_exceptions` for anything unmatched and never silently fixes data.

## Context map relationships

| Upstream | Downstream | Relationship |
|---|---|---|
| Authorisation | Ledger | Partnership upstream: Ledger consumes approved authorisations as a Published Language event (AuthorisationApproved). Ledger is downstream and may reject a posting that fails its own invariants. |
| Ledger | Billing | Customer-Supplier: Billing is the customer, Ledger supplies posted transactions and balances. Billing conforms to the Ledger event contract (conformist) because statement totals must match the ledger exactly. |
| Ledger | Settlement | Customer-Supplier: Settlement consumes posted transactions (conformist on the TransactionRecorded contract). |
| Settlement | Reconciliation | Published Language: settlement batch files and SettlementCompleted events. |
| All contexts | Reconciliation | Reconciliation is downstream of every context and reads their published events. It applies an Anti-Corruption Layer when consuming external scheme and merchant files, translating foreign formats into the platform model before matching, so external file quirks never leak into Ledger or Billing. |
| Billing | Card member channels (outside this platform slice) | Open Host Service: the statement-enquiry read model (Project 3 CQRS) is the published, query-optimised view of Billing data. |

## Domain events each context publishes

- Authorisation: AuthorisationApproved, AuthorisationDeclined, AuthorisationReversed
- Ledger: TransactionRecorded, LedgerEntryPosted, TransactionReversed
- Billing: StatementPeriodOpened, StatementGenerated, PaymentDueDateSet
- Settlement: SettlementBatchCreated, SettlementCompleted, SettlementFailed
- Reconciliation: ReconciliationExceptionRaised, ReconciliationExceptionCleared

## Integration style

Contexts integrate through domain events, not shared tables. Each context owns its
own store. The one deliberate exception in this portfolio slice is the CQRS read
model: Billing's statement-enquiry view is a projection of Ledger and Billing
events, rebuilt at any time by replaying the event log, and is never written to
directly by other contexts.
