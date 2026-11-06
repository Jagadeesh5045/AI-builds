# Card Ledger: Domain Model

Status: portfolio project, synthetic data only. This document describes the problem
domain in business language, before any database design decisions.

## Domain summary

A card issuer gives payment cards to its members. Members spend at merchants.
Each spend is first authorised (a real time approval check), then cleared as a
financial transaction, then posted to a double entry ledger, then billed to the
member on a monthly statement, and settled to the merchant in daily batches.
Reconciliation is the control process that proves these views of the same money
agree with each other.

## Bounded context and core concepts

| Concept | Meaning in the domain | Key business rules |
|---|---|---|
| Card Member | A customer of the issuer who holds an account | A member can hold several cards |
| Card | A payment instrument issued to one member | Card numbers are never stored in clear; only a token is kept |
| Merchant | A business that accepts card payments | Identified by a Merchant Category Code (MCC) |
| Authorisation | A real time request to approve a spend | An authorisation can be approved or declined; approved does not always mean captured |
| Transaction | The cleared financial record of a spend or refund | Every transaction is valued in GBP in this model |
| Ledger Entry | One line of a double entry posting | Total debits must always equal total credits |
| Statement | The monthly bill presented to a member | Closing balance equals opening balance plus the signed period activity |
| Settlement | The daily amount owed to a merchant | Net equals gross minus fees |
| Reconciliation Exception | A detected disagreement between two views of the same money | Exceptions are evidence of a control working, and are stored, not deleted |

## Domain flows

1. Authorisation flow: a card is presented at a merchant, the issuer approves or
   declines, and the decision is recorded with an amount and a timestamp.
2. Clearing flow: approved authorisations become transactions. A transaction is
   either a purchase (money owed by the member) or a refund (money returned).
3. Posting flow: each transaction creates a balanced pair of ledger entries, a
   debit to the member receivable account and a credit to the merchant payable
   account (reversed for refunds).
4. Billing flow: transactions are grouped by member and month into statements
   with opening and closing balances and a payment due date.
5. Settlement flow: transactions are grouped by merchant and day; the issuer
   deducts its fee and records the net payable.
6. Reconciliation flow: independent checks compare authorisations, transactions,
   ledger totals and settlement totals, and record every disagreement as an
   exception for investigation.

## Glossary notes

Signed amount convention: purchases count positive for member balances and
merchant gross; refunds count negative. Amounts stored on transactions are
always positive; the sign is derived from the transaction type.
