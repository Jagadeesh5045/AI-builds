# Project 1: Card Ledger Relational Data Architecture (PostgreSQL)

Portfolio project. Synthetic data only, currency GBP. This project designs and
builds the relational core of a card issuer ledger: members, cards, merchants,
authorisations, cleared transactions, a double entry ledger, monthly member
statements, daily merchant settlements, and a reconciliation pack that proves
the views agree.

## Design documents

All in `../docs/models/`:

- `domain_model.md`: the business domain, flows and rules.
- `conceptual_model.md`: entities and relationships, technology free.
- `logical_model.md`: attributes and keys for all ten tables.
- `physical_model.md`: PostgreSQL storage choices, partitioning, indexes,
  constraint strategy, and what is deliberately not claimed.
- `normalisation_note.md`: how the design meets 3NF, and every
  denormalisation decision with its reason.
- `card_ledger.dbml`: the full model in DBML for dbdiagram or Hackolade import.
- `er_diagram.mmd`: Mermaid ER diagram source.

## Key design choices

- Money is `numeric(12,2)` everywhere. Never float.
- `transactions` is RANGE partitioned by month on `transaction_date`
  (2026-07, 2026-08, 2026-09, plus a DEFAULT partition), because statement and
  settlement workloads are date driven.
- `transactions.auth_id` is deliberately not a foreign key, and
  `ledger_entries.txn_id` cannot be one against a partitioned parent. Both
  links are verified by the reconciliation pack instead, so a broken link is
  detectable rather than impossible. See the physical model.
- Statements and settlements are stored derived documents, not views, so the
  reconciliation checks against them are real controls.
- The generator injects exactly 12 known mismatches (5 settlement amount,
  4 missing ledger postings, 3 authorisation amount). Reconciliation must find
  exactly those 12. The injected ids are recorded in `injected.json`.

## Layout

- `sql/ddl.sql`: all tables, constraints, partitions.
- `sql/indexes.sql`: indexes, applied separately so the statement lookup can
  be benchmarked before and after.
- `generate_data.py`: seeded (42) generator and loader, psycopg COPY based.
- `queries/reconciliation.sql`: three exception checks plus an informational
  unmatched authorisation report.
- `queries/benchmarks.sql`: member statement lookup and merchant daily totals.
- `tests/test_project1.py`: pytest suite (row counts, debits equal credits,
  partition totals, reconciliation finds exactly the injected 12).
- `RESULTS.md`: what was actually run, with measured numbers.

## Run it

The recorded run in RESULTS.md used a real PostgreSQL 16.2 server from the
`pgserver` pip distribution (apt was unavailable in that session; see
RESULTS.md for the full explanation). Either route works:

```bash
sudo apt-get install -y postgresql
sudo service postgresql start
sudo -u postgres psql -c "CREATE USER cardledger PASSWORD 'cardledger'"
sudo -u postgres psql -c "CREATE DATABASE cardledger OWNER cardledger"
export DATABASE_URL=postgresql://cardledger:cardledger@localhost:5432/cardledger
psql "$DATABASE_URL" -f sql/ddl.sql
python generate_data.py
psql "$DATABASE_URL" -f queries/reconciliation.sql
pytest tests -v
```
