"""Incremental star-schema warehouse load.

Schema (SQLite):
  dim_customer - SCD Type 2 on city/postcode. One row per customer
      version, with valid_from / valid_to and is_current. The natural
      key is customer_id; the surrogate key is customer_sk.
  dim_product  - SCD Type 2 on unit_price, same pattern.
  dim_date     - one row per calendar date covering the orders.
  fact_orders  - one row per deduplicated order, pointing at the
      customer and product versions valid on the order date.
  quarantine   - orders that cannot be loaded (unknown customer or
      product), kept with a reason instead of being dropped silently.
  load_manifest - one row per batch load, with row counts for audit.

Load rules, in order:
  1. Deduplicate orders by order_id, keeping the row with the latest
     updated_at (so a late correction replaces the original).
  2. Upsert dimension versions. A changed customer or product closes
     the current row (valid_to = day before the new effective date)
     and opens a new current row.
  3. Load facts only for orders whose customer and product versions
     exist; everything else goes to quarantine.
  4. Re-running the same batch is idempotent: fact rows are keyed by
     order_id, so a second run replaces rather than duplicates them,
     and dimension upserts detect no-change rows.

Dates are stored as ISO text (YYYY-MM-DD) so comparisons are plain
string comparisons and the code needs no date library beyond
datetime.date for the date dimension.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_sk INTEGER PRIMARY KEY,
    customer_id TEXT NOT NULL,
    first_name TEXT, last_name TEXT, email TEXT,
    city TEXT, postcode TEXT,
    valid_from TEXT NOT NULL, valid_to TEXT,
    is_current INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_product (
    product_sk INTEGER PRIMARY KEY,
    product_id TEXT NOT NULL,
    name TEXT, category TEXT, unit_price REAL,
    valid_from TEXT NOT NULL, valid_to TEXT,
    is_current INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_date (
    date_key TEXT PRIMARY KEY,
    year INTEGER, month INTEGER, day INTEGER,
    quarter INTEGER, weekday TEXT
);
CREATE TABLE IF NOT EXISTS fact_orders (
    order_id TEXT PRIMARY KEY,
    order_date TEXT NOT NULL,
    customer_sk INTEGER NOT NULL,
    product_sk INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quarantine (
    order_id TEXT, reason TEXT, raw TEXT
);
CREATE TABLE IF NOT EXISTS load_manifest (
    batch_label TEXT PRIMARY KEY,
    source_orders INTEGER, deduped_orders INTEGER,
    fact_rows INTEGER, quarantined INTEGER,
    customers_current INTEGER, products_current INTEGER
);
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _read_sources(raw_dir: str | Path) -> tuple[list[dict], list[dict], list[dict]]:
    raw_dir = Path(raw_dir)
    customers = json.loads((raw_dir / "customers.json").read_text(encoding="utf-8"))
    with open(raw_dir / "products.csv", newline="", encoding="utf-8") as fh:
        products = list(csv.DictReader(fh))
    with open(raw_dir / "orders.csv", newline="", encoding="utf-8") as fh:
        orders = list(csv.DictReader(fh))
    return customers, products, orders


def dedupe_orders(orders: list[dict]) -> list[dict]:
    """Keep, per order_id, the row with the latest updated_at."""
    best: dict[str, dict] = {}
    for row in orders:
        cur = best.get(row["order_id"])
        if cur is None or row["updated_at"] >= cur["updated_at"]:
            best[row["order_id"]] = row
    return list(best.values())


def _upsert_customer(conn: sqlite3.Connection, cust: dict, effective: str) -> None:
    cur = conn.execute(
        "SELECT customer_sk, city, postcode FROM dim_customer "
        "WHERE customer_id = ? AND is_current = 1", (cust["customer_id"],)).fetchone()
    if cur and cur[1] == cust["city"] and cur[2] == cust["postcode"]:
        return  # no change: SCD2 needs no new version
    if cur:
        prev_day = (date.fromisoformat(effective) - timedelta(days=1)).isoformat()
        conn.execute("UPDATE dim_customer SET valid_to = ?, is_current = 0 "
                     "WHERE customer_sk = ?", (prev_day, cur[0]))
    conn.execute(
        "INSERT INTO dim_customer (customer_id, first_name, last_name, email, "
        "city, postcode, valid_from, valid_to, is_current) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 1)",
        (cust["customer_id"], cust["first_name"], cust["last_name"], cust["email"],
         cust["city"], cust["postcode"], effective))


def _upsert_product(conn: sqlite3.Connection, prod: dict) -> None:
    effective = prod["effective_date"]
    cur = conn.execute(
        "SELECT product_sk, unit_price FROM dim_product "
        "WHERE product_id = ? AND is_current = 1", (prod["product_id"],)).fetchone()
    if cur and float(cur[1]) == float(prod["unit_price"]):
        return
    if cur:
        prev_day = (date.fromisoformat(effective) - timedelta(days=1)).isoformat()
        conn.execute("UPDATE dim_product SET valid_to = ?, is_current = 0 "
                     "WHERE product_sk = ?", (prev_day, cur[0]))
    conn.execute(
        "INSERT INTO dim_product (product_id, name, category, unit_price, "
        "valid_from, valid_to, is_current) VALUES (?, ?, ?, ?, ?, NULL, 1)",
        (prod["product_id"], prod["name"], prod["category"],
         float(prod["unit_price"]), effective))


def _version_sk(conn: sqlite3.Connection, table: str, key_col: str,
                natural_col: str, natural_val: str, order_date: str) -> int | None:
    """Surrogate key of the dimension version valid on order_date.

    Falls back to the earliest version when no version covers the date
    (e.g. an order predates the first version), so the order still
    loads instead of being quarantined for a date gap.
    """
    row = conn.execute(
        f"SELECT {key_col} FROM {table} WHERE {natural_col} = ? "
        "AND valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?) "
        "ORDER BY valid_from DESC LIMIT 1",
        (natural_val, order_date, order_date)).fetchone()
    if row is None:
        row = conn.execute(
            f"SELECT {key_col} FROM {table} WHERE {natural_col} = ? "
            "ORDER BY valid_from ASC LIMIT 1", (natural_val,)).fetchone()
    return row[0] if row else None


def _build_date_dim(conn: sqlite3.Connection, dates: list[str]) -> None:
    for raw in sorted(set(dates)):
        d = date.fromisoformat(raw)
        conn.execute(
            "INSERT OR IGNORE INTO dim_date VALUES (?, ?, ?, ?, ?, ?)",
            (raw, d.year, d.month, d.day, (d.month - 1) // 3 + 1, d.strftime("%A")))


def run_load(conn: sqlite3.Connection, raw_dir: str | Path, batch_label: str,
             effective_date: str) -> dict:
    """Load one batch directory into the warehouse. Returns the manifest row."""
    customers, products, orders = _read_sources(raw_dir)
    deduped = dedupe_orders(orders)

    for cust in customers:
        _upsert_customer(conn, cust, effective_date)
    for prod in products:
        _upsert_product(conn, prod)

    fact_rows = 0
    quarantined = 0
    for order in deduped:
        cust_sk = _version_sk(conn, "dim_customer", "customer_sk",
                              "customer_id", order["customer_id"],
                              order["order_date"])
        prod_sk = _version_sk(conn, "dim_product", "product_sk",
                              "product_id", order["product_id"],
                              order["order_date"])
        if cust_sk is None or prod_sk is None:
            missing = "customer" if cust_sk is None else "product"
            conn.execute("INSERT INTO quarantine VALUES (?, ?, ?)",
                         (order["order_id"], f"unknown {missing}",
                          json.dumps(order, sort_keys=True)))
            quarantined += 1
            continue
        conn.execute(
            "INSERT INTO fact_orders VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(order_id) DO UPDATE SET order_date = excluded.order_date, "
            "customer_sk = excluded.customer_sk, product_sk = excluded.product_sk, "
            "quantity = excluded.quantity, total_amount = excluded.total_amount, "
            "status = excluded.status",
            (order["order_id"], order["order_date"], cust_sk, prod_sk,
             int(order["quantity"]), float(order["total_amount"]), order["status"]))
        fact_rows += 1

    _build_date_dim(conn, [o["order_date"] for o in deduped])
    cust_current = conn.execute(
        "SELECT COUNT(*) FROM dim_customer WHERE is_current = 1").fetchone()[0]
    prod_current = conn.execute(
        "SELECT COUNT(*) FROM dim_product WHERE is_current = 1").fetchone()[0]
    manifest = {
        "batch_label": batch_label,
        "source_orders": len(orders),
        "deduped_orders": len(deduped),
        "fact_rows": fact_rows,
        "quarantined": quarantined,
        "customers_current": cust_current,
        "products_current": prod_current,
    }
    conn.execute(
        "INSERT INTO load_manifest VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(batch_label) DO UPDATE SET source_orders = excluded.source_orders, "
        "deduped_orders = excluded.deduped_orders, fact_rows = excluded.fact_rows, "
        "quarantined = excluded.quarantined, "
        "customers_current = excluded.customers_current, "
        "products_current = excluded.products_current",
        (batch_label, manifest["source_orders"], manifest["deduped_orders"],
         fact_rows, quarantined, cust_current, prod_current))
    conn.commit()
    return manifest


def quality_checks(conn: sqlite3.Connection) -> list[dict]:
    """Checks that must all pass before a load is trusted."""
    checks = []
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    dupes = conn.execute(
        "SELECT COUNT(*) FROM (SELECT order_id FROM fact_orders "
        "GROUP BY order_id HAVING COUNT(*) > 1)").fetchone()[0]
    add("no duplicate fact orders", dupes == 0, f"duplicate order_ids: {dupes}")
    orphans = conn.execute(
        "SELECT COUNT(*) FROM fact_orders f LEFT JOIN dim_customer c "
        "ON f.customer_sk = c.customer_sk WHERE c.customer_sk IS NULL").fetchone()[0]
    add("fact customer keys resolve", orphans == 0, f"orphan rows: {orphans}")
    orphans = conn.execute(
        "SELECT COUNT(*) FROM fact_orders f LEFT JOIN dim_product p "
        "ON f.product_sk = p.product_sk WHERE p.product_sk IS NULL").fetchone()[0]
    add("fact product keys resolve", orphans == 0, f"orphan rows: {orphans}")
    multi_current = conn.execute(
        "SELECT COUNT(*) FROM (SELECT customer_id FROM dim_customer "
        "WHERE is_current = 1 GROUP BY customer_id HAVING COUNT(*) > 1)").fetchone()[0]
    add("one current customer version each", multi_current == 0,
        f"customers with >1 current row: {multi_current}")
    nulls = conn.execute(
        "SELECT COUNT(*) FROM fact_orders WHERE total_amount IS NULL "
        "OR quantity IS NULL OR order_date IS NULL").fetchone()[0]
    add("no null fact measures", nulls == 0, f"null rows: {nulls}")
    neg = conn.execute(
        "SELECT COUNT(*) FROM fact_orders WHERE total_amount < 0 OR quantity <= 0").fetchone()[0]
    add("no negative amounts or quantities", neg == 0, f"bad rows: {neg}")
    return checks


def kpi_report(conn: sqlite3.Connection) -> dict:
    """Completed-order KPIs only; cancelled orders stay in the fact table
    for audit but never inflate revenue."""
    revenue, orders = conn.execute(
        "SELECT COALESCE(SUM(total_amount), 0), COUNT(*) FROM fact_orders "
        "WHERE status = 'completed'").fetchone()
    by_month = conn.execute(
        "SELECT substr(order_date, 1, 7) AS m, COUNT(*), "
        "ROUND(SUM(total_amount), 2) FROM fact_orders WHERE status = 'completed' "
        "GROUP BY m ORDER BY m").fetchall()
    by_category = conn.execute(
        "SELECT p.category, COUNT(*), ROUND(SUM(f.total_amount), 2) "
        "FROM fact_orders f JOIN dim_product p ON f.product_sk = p.product_sk "
        "WHERE f.status = 'completed' GROUP BY p.category ORDER BY 3 DESC").fetchall()
    return {
        "completed_orders": orders,
        "completed_revenue": round(revenue, 2),
        "by_month": [{"month": m, "orders": n, "revenue": r} for m, n, r in by_month],
        "by_category": [{"category": c, "orders": n, "revenue": r}
                        for c, n, r in by_category],
    }
