# Card Ledger: Logical Model

The logical model fixes attributes and keys for every entity. Data types here
are generic (text, date, decimal); PostgreSQL specific choices live in the
physical model. All data is synthetic, currency GBP.

## card_members
| Attribute | Key | Notes |
|---|---|---|
| member_id | PK | Surrogate identifier |
| full_name | | Synthetic person name |
| email | Unique | Synthetic, example.com domain only |
| joined_date | | Date the member joined |

## cards
| Attribute | Key | Notes |
|---|---|---|
| card_id | PK | |
| member_id | FK to card_members | |
| card_number_token | Unique | Token only, never a real card number |
| network | | Card network, for example VISA or MASTERCARD |
| status | | active, blocked or closed |
| issued_date | | |

## merchants
| Attribute | Key | Notes |
|---|---|---|
| merchant_id | PK | |
| merchant_name | | Synthetic business name |
| mcc | | Merchant Category Code |
| country | | |

## authorisations
| Attribute | Key | Notes |
|---|---|---|
| auth_id | PK | |
| card_id | FK to cards | |
| merchant_id | FK to merchants | |
| amount | | Positive decimal, two places |
| currency | | GBP throughout this model |
| auth_status | | approved or declined |
| authorised_at | | Timestamp of the decision |

## transactions
| Attribute | Key | Notes |
|---|---|---|
| txn_id | PK (with transaction_date) | Part of the partition key in the physical model |
| card_id | FK to cards | |
| member_id | FK to card_members | Denormalised from cards, see normalisation note |
| merchant_id | FK to merchants | |
| auth_id | FK to authorisations, nullable | Null for transactions with no separate authorisation record |
| amount | | Positive decimal, two places |
| currency | | GBP |
| txn_type | | purchase or refund |
| status | | posted or reversed |
| transaction_date | | Calendar date of the transaction, and the partition attribute |

## ledger_entries
| Attribute | Key | Notes |
|---|---|---|
| entry_id | PK | |
| txn_id | FK to transactions | |
| account_code | | Chart of accounts code, for example 1100 member receivable, 2100 merchant payable |
| debit | | Decimal, zero or positive |
| credit | | Decimal, zero or positive; exactly one of debit and credit is non zero on a line |
| entry_date | | |

## statements
| Attribute | Key | Notes |
|---|---|---|
| statement_id | PK | |
| member_id | FK to card_members, part of unique (member_id, period_start) | |
| period_start | | First day of the billing month |
| period_end | | Last day of the billing month |
| opening_balance | | Previous period closing balance |
| closing_balance | | opening_balance plus signed period activity |
| due_date | | period_end plus 21 days |

## settlements
| Attribute | Key | Notes |
|---|---|---|
| settlement_id | PK | |
| merchant_id | FK to merchants, part of unique (merchant_id, settlement_date) | |
| settlement_date | | |
| gross_amount | | Signed sum of that merchant's transactions that day |
| fees | | 1.5 percent of that day's purchase total, rounded to two places |
| net_amount | | gross_amount minus fees |
| txn_count | | Number of transactions in the batch |

## reconciliation_exceptions
| Attribute | Key | Notes |
|---|---|---|
| exception_id | PK | |
| source_table | | Table where the problem record lives |
| source_id | | Identifier of the problem record |
| issue_type | | auth_amount_mismatch, missing_ledger_entries or settlement_amount_mismatch |
| expected_amount | | Amount the control expected |
| actual_amount | | Amount actually found |
| detected_at | | Timestamp the reconciliation run detected it |

## Derived data rule

Statements and settlements are stored (not views) because they are issued
business documents with balances that must not change silently if source data
is later corrected. Reconciliation exists to prove the stored documents still
agree with the transactions they were derived from.
