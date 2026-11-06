# Card Ledger: Conceptual Model

The conceptual model names the entities and their relationships, with no data
types, keys or physical choices. All data is synthetic.

## Entities and relationships

| Entity | Relationship | Cardinality |
|---|---|---|
| Card Member | holds | 1 member to many Cards |
| Card Member | is billed on | 1 member to many Statements |
| Card Member | generates (through a card) | 1 member to many Transactions |
| Card | is used in | 1 card to many Authorisations |
| Card | is used in | 1 card to many Transactions |
| Merchant | requests | 1 merchant to many Authorisations |
| Merchant | accepts | 1 merchant to many Transactions |
| Merchant | is paid through | 1 merchant to many Settlements |
| Authorisation | may clear into | 1 authorisation to at most 1 Transaction |
| Transaction | is posted as | 1 transaction to many Ledger Entries |
| Transaction | is summarised into | many Transactions to 1 Statement (by member and month) |
| Transaction | is summarised into | many Transactions to 1 Settlement (by merchant and day) |
| Reconciliation Exception | refers to | 1 exception to 1 source record in any source table |

## Conceptual rules

- A Transaction cannot exist for a Card that does not exist, and a Card cannot
  exist without a Card Member.
- A Ledger Entry only exists as part of a balanced posting for a Transaction.
- A Statement is a derived view: it must be reproducible from Transactions.
- A Settlement is a derived view: it must be reproducible from Transactions.
- A Reconciliation Exception is a first class record, not a log line: it keeps
  the expected amount, the actual amount and the time of detection.
