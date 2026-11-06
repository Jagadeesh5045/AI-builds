# Card Ledger: Normalisation Note

## How the design meets 3NF

First Normal Form: every attribute is atomic. Amounts, dates and statuses are
single values; there are no repeating groups (a transaction's ledger lines are
separate rows in ledger_entries, not columns on transactions).

Second Normal Form: every non key attribute depends on the whole key of its
table. The only composite key is the transactions primary key
(txn_id, transaction_date); transaction attributes such as amount and status
describe the transaction identified by txn_id, and transaction_date is the
partition attribute carried in the key for PostgreSQL partitioning, not a
second business identifier being partially depended on.

Third Normal Form: no non key attribute depends on another non key attribute.

- Merchant name, MCC and country live only in merchants, never copied onto
  transactions or settlements.
- Member name and email live only in card_members, never on cards, statements
  or transactions.
- Card network and status live only in cards.
- Fees and net on settlements are functions of the settlement key
  (merchant, day), computed from that day's transactions at derivation time,
  not copied from any other stored attribute.

## Denormalisation decisions, each with its reason

1. transactions.member_id duplicates the member reachable through
   transactions.card_id then cards.member_id.
   Reason: the two highest value queries in this domain, member statement
   lookup and member balance checks, would otherwise join through cards on
   every row of the largest table. The value is written once by the generator
   from the card's owner, and the reconciliation pack can check it if a future
   build wants that control. Accepted risk: a card reissued to a different
   member would need a transaction correction process.

2. statements stores opening_balance and closing_balance instead of computing
   them on read.
   Reason: a statement is an issued legal document. Its balances must stay
   fixed even if a late correction lands in transactions afterwards; the
   difference is then a reconciliation matter, not a silent restatement.

3. settlements stores gross_amount, fees, net_amount and txn_count instead of
   computing them on read.
   Reason: same as statements. A settlement is a payment instruction. Storing
   the issued figures is what makes the settlement versus transactions check
   a real control rather than a tautology.

4. reconciliation_exceptions stores expected_amount and actual_amount.
   Reason: an exception must preserve the evidence as it was at detection
   time, even after the underlying record is fixed.

## What was NOT denormalised

No merchant or member descriptive attributes were copied onto transactions,
authorisations or settlements. Query convenience was solved with indexes
(physical model), not by duplicating descriptive data.
