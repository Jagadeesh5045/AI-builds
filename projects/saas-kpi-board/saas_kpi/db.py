"""Load a generated Dataset into SQLite with an analyst-friendly schema."""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    signup_month TEXT NOT NULL,
    channel TEXT NOT NULL,
    country TEXT NOT NULL,
    initial_plan TEXT NOT NULL,
    cac REAL NOT NULL,
    churn_month TEXT
);
CREATE TABLE subscription_snapshots (
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    month TEXT NOT NULL,
    plan TEXT NOT NULL,
    mrr REAL NOT NULL,
    PRIMARY KEY (customer_id, month)
);
CREATE TABLE invoices (
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    month TEXT NOT NULL,
    amount REAL NOT NULL,
    paid INTEGER NOT NULL CHECK (paid IN (0, 1)),
    PRIMARY KEY (customer_id, month)
);
CREATE INDEX idx_snap_month ON subscription_snapshots(month);
CREATE INDEX idx_cust_signup ON customers(signup_month);
"""


def load(conn: sqlite3.Connection, dataset) -> None:
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
        [
            (c.customer_id, c.signup_month, c.channel, c.country,
             c.plan_history[0][1], round(c.cac, 2), c.churn_month)
            for c in dataset.customers
        ],
    )
    conn.executemany(
        "INSERT INTO subscription_snapshots VALUES (?,?,?,?)",
        [(s["customer_id"], s["month"], s["plan"], s["mrr"]) for s in dataset.snapshots],
    )
    conn.executemany(
        "INSERT INTO invoices VALUES (?,?,?,?)",
        [(i["customer_id"], i["month"], i["amount"], i["paid"]) for i in dataset.invoices],
    )
    conn.commit()
