-- Card Ledger: YugabyteDB (YSQL) DDL
-- Adapted from Project 1 (project1-postgres/sql/ddl.sql). Table definitions,
-- column names, constraints and the monthly RANGE partitioned transactions
-- design are carried over unchanged; see SCHEMA_NOTES.md for the probes and
-- the few adaptations that were needed. Synthetic data only, currency GBP.
-- Target database: card_ledger (created separately before this file runs).

BEGIN;

DROP TABLE IF EXISTS reconciliation_exceptions CASCADE;
DROP TABLE IF EXISTS settlements CASCADE;
DROP TABLE IF EXISTS statements CASCADE;
DROP TABLE IF EXISTS ledger_entries CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS authorisations CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;
DROP TABLE IF EXISTS cards CASCADE;
DROP TABLE IF EXISTS card_members CASCADE;

CREATE TABLE card_members (
    member_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    full_name   varchar(120) NOT NULL,
    email       varchar(200) NOT NULL UNIQUE,
    joined_date date NOT NULL
);

CREATE TABLE cards (
    card_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    member_id         bigint NOT NULL REFERENCES card_members (member_id),
    card_number_token varchar(32) NOT NULL UNIQUE,
    network           varchar(20) NOT NULL CHECK (network IN ('VISA', 'MASTERCARD', 'AMEX')),
    status            varchar(10) NOT NULL CHECK (status IN ('active', 'blocked', 'closed')),
    issued_date       date NOT NULL
);

CREATE TABLE merchants (
    merchant_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    merchant_name varchar(120) NOT NULL,
    mcc           varchar(4) NOT NULL,
    country       varchar(60) NOT NULL
);

CREATE TABLE authorisations (
    auth_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    card_id       bigint NOT NULL REFERENCES cards (card_id),
    merchant_id   bigint NOT NULL REFERENCES merchants (merchant_id),
    amount        numeric(12,2) NOT NULL CHECK (amount > 0),
    currency      char(3) NOT NULL DEFAULT 'GBP' CHECK (currency = 'GBP'),
    auth_status   varchar(10) NOT NULL CHECK (auth_status IN ('approved', 'declined')),
    authorised_at timestamp NOT NULL
);

-- Transactions: monthly RANGE partitioned on transaction_date.
-- auth_id is deliberately NOT a foreign key: reconciliation must be able to
-- detect a transaction whose authorisation is missing or inconsistent.
-- ledger_entries.txn_id is likewise not a foreign key, because a foreign key
-- cannot reference a single column of a partitioned table whose primary key
-- is (txn_id, transaction_date). Both links are verified by reconciliation.
CREATE TABLE transactions (
    txn_id           bigint GENERATED ALWAYS AS IDENTITY,
    card_id          bigint NOT NULL,
    member_id        bigint NOT NULL,
    merchant_id      bigint NOT NULL,
    auth_id          bigint,
    amount           numeric(12,2) NOT NULL CHECK (amount > 0),
    currency         char(3) NOT NULL DEFAULT 'GBP' CHECK (currency = 'GBP'),
    txn_type         varchar(10) NOT NULL CHECK (txn_type IN ('purchase', 'refund')),
    status           varchar(10) NOT NULL CHECK (status IN ('posted', 'reversed')),
    transaction_date date NOT NULL,
    PRIMARY KEY (txn_id, transaction_date)
) PARTITION BY RANGE (transaction_date);

CREATE TABLE transactions_2026_07 PARTITION OF transactions
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
CREATE TABLE transactions_2026_08 PARTITION OF transactions
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE transactions_2026_09 PARTITION OF transactions
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE transactions_default PARTITION OF transactions DEFAULT;

CREATE TABLE ledger_entries (
    entry_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    txn_id     bigint NOT NULL,
    account_code varchar(10) NOT NULL CHECK (account_code IN ('1100', '2100')),
    debit      numeric(12,2) NOT NULL DEFAULT 0 CHECK (debit >= 0),
    credit     numeric(12,2) NOT NULL DEFAULT 0 CHECK (credit >= 0),
    entry_date date NOT NULL,
    CHECK ((debit > 0 AND credit = 0) OR (credit > 0 AND debit = 0))
);

CREATE TABLE statements (
    statement_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    member_id       bigint NOT NULL REFERENCES card_members (member_id),
    period_start    date NOT NULL,
    period_end      date NOT NULL CHECK (period_end >= period_start),
    opening_balance numeric(12,2) NOT NULL,
    closing_balance numeric(12,2) NOT NULL,
    due_date        date NOT NULL,
    UNIQUE (member_id, period_start)
);

CREATE TABLE settlements (
    settlement_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    merchant_id     bigint NOT NULL REFERENCES merchants (merchant_id),
    settlement_date date NOT NULL,
    gross_amount    numeric(12,2) NOT NULL,
    fees            numeric(12,2) NOT NULL CHECK (fees >= 0),
    net_amount      numeric(12,2) NOT NULL,
    txn_count       int NOT NULL CHECK (txn_count >= 0),
    CHECK (net_amount = gross_amount - fees),
    UNIQUE (merchant_id, settlement_date)
);

CREATE TABLE reconciliation_exceptions (
    exception_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_table    varchar(40) NOT NULL,
    source_id       bigint NOT NULL,
    issue_type      varchar(40) NOT NULL CHECK (issue_type IN
        ('auth_amount_mismatch', 'missing_ledger_entries', 'settlement_amount_mismatch')),
    expected_amount numeric(12,2),
    actual_amount   numeric(12,2),
    detected_at     timestamp NOT NULL DEFAULT now()
);

COMMIT;
