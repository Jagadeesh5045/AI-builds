"""Deterministic sample source data for retail-warehouse.

Generates the messy source files a real warehouse load has to survive:
duplicate order rows, a late-arriving correction, cancelled orders,
customers who move house, products whose price changes, and a few
orders that reference customers or products that do not exist yet.

Everything is seeded, so two runs of the same batch produce identical
files. Batch 1 is the initial load; batch 2 is the incremental load
(new rows plus changes to rows batch 1 already saw).

Files written per batch (under ``<out_dir>/batch<N>/``):
  customers.json - one object per customer row
  products.csv   - one row per product row
  orders.csv     - one row per order row
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

FIRST = ["Aisha", "James", "Priya", "Mohammed", "Sophie", "Liam", "Emma",
         "Arjun", "Grace", "Oliver", "Mia", "Noah", "Ivy", "Lucas", "Zara"]
LAST = ["Khan", "Smith", "Patel", "Ahmed", "Jones", "Brown", "Taylor",
        "Sharma", "Wilson", "Davies", "Ali", "Evans", "Kaur", "Wood", "Hussain"]
CITIES = [("Birmingham", "B1"), ("London", "E1"), ("Manchester", "M1"),
          ("Leeds", "LS1"), ("Bristol", "BS1"), ("Glasgow", "G1"),
          ("Edinburgh", "EH1"), ("Cambridge", "CB1")]
PRODUCTS = [
    ("Wireless headphones", "Electronics", 49.99),
    ("Mechanical keyboard", "Electronics", 79.99),
    ("Coffee maker", "Home", 59.99),
    ("Desk lamp", "Home", 24.99),
    ("Running shoes", "Apparel", 69.99),
    ("Backpack", "Apparel", 39.99),
    ("Blender", "Home", 44.99),
    ("Smart speaker", "Electronics", 89.99),
    ("Yoga mat", "Apparel", 19.99),
    ("Monitor stand", "Electronics", 34.99),
    ("Water bottle", "Home", 14.99),
    ("Winter coat", "Apparel", 99.99),
]
STATUSES = ["completed", "completed", "completed", "completed", "cancelled"]


def _customers(rng: random.Random, start: int, count: int) -> list[dict]:
    out = []
    for i in range(start, start + count):
        first = rng.choice(FIRST)
        last = rng.choice(LAST)
        city, prefix = rng.choice(CITIES)
        out.append({
            "customer_id": f"C{i:04d}",
            "first_name": first,
            "last_name": last,
            "email": f"{first.lower()}.{last.lower()}{i}@example.co.uk",
            "city": city,
            "postcode": f"{prefix} {rng.randint(1, 9)}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",
            "signup_date": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        })
    return out


def _products() -> list[dict]:
    return [{"product_id": f"P{i + 1:03d}", "name": name, "category": cat,
             "unit_price": price, "effective_date": "2025-01-01"}
            for i, (name, cat, price) in enumerate(PRODUCTS)]


def _orders(rng: random.Random, customers: list[dict], products: list[dict],
            start: int, count: int, start_month: int) -> list[dict]:
    out = []
    for i in range(start, start + count):
        cust = rng.choice(customers)
        prod = rng.choice(products)
        qty = rng.randint(1, 3)
        month = rng.randint(start_month, start_month + 2)
        out.append({
            "order_id": f"O{i:05d}",
            "customer_id": cust["customer_id"],
            "product_id": prod["product_id"],
            "order_date": f"2025-{month:02d}-{rng.randint(1, 28):02d}",
            "quantity": qty,
            "unit_price": prod["unit_price"],
            "total_amount": round(qty * prod["unit_price"], 2),
            "status": rng.choice(STATUSES),
            "updated_at": f"2025-{month:02d}-{rng.randint(1, 28):02d}",
        })
    return out


def generate_batch(batch: int, out_dir: str | Path) -> Path:
    """Write one batch of source files. Returns the batch directory."""
    if batch not in (1, 2):
        raise ValueError("batch must be 1 or 2")
    rng = random.Random(42 + batch)
    dest = Path(out_dir) / f"batch{batch}"
    dest.mkdir(parents=True, exist_ok=True)

    # The 120 base customers are identical in both batches: batch 2 must
    # describe the same people, not a fresh random draw, or every
    # customer would spuriously look like a mover. Only the eight named
    # movers change address in batch 2.
    base = _customers(random.Random(100), 1, 120)
    if batch == 1:
        customers = base
        products = _products()
        orders = _orders(rng, customers, products, 1, 400, 1)
        # Mess on purpose: two exact duplicate rows and one orphan order.
        orders.append(dict(orders[10]))
        orders.append(dict(orders[55]))
        orders.append({"order_id": "O99901", "customer_id": "C9999",
                       "product_id": "P001", "order_date": "2025-03-10",
                       "quantity": 1, "unit_price": 49.99, "total_amount": 49.99,
                       "status": "completed", "updated_at": "2025-03-10"})
    else:
        customers = base
        # Eight existing customers move house in batch 2.
        movers = ["C0003", "C0011", "C0027", "C0035", "C0042", "C0061", "C0080", "C0104"]
        changed = []
        for cust in customers:
            cust = dict(cust)
            if cust["customer_id"] in movers:
                city, prefix = rng.choice(CITIES)
                cust["city"] = city
                cust["postcode"] = f"{prefix} {rng.randint(1, 9)}AB"
            changed.append(cust)
        customers = changed
        # Fifteen brand-new customers.
        customers += _customers(rng, 121, 15)
        # Three products get a 10% price rise in batch 2.
        products = []
        for i, (name, cat, price) in enumerate(PRODUCTS):
            new_price = round(price * 1.10, 2) if f"P{i + 1:03d}" in ("P001", "P004", "P008") else price
            eff = "2025-07-01" if f"P{i + 1:03d}" in ("P001", "P004", "P008") else "2025-01-01"
            products.append({"product_id": f"P{i + 1:03d}", "name": name,
                             "category": cat, "unit_price": new_price,
                             "effective_date": eff})
        orders = _orders(rng, customers, products, 401, 180, 7)
        # Late-arriving correction: order O00042 was undercharged in batch 1.
        orders.append({"order_id": "O00042", "customer_id": "C0019",
                       "product_id": "P002", "order_date": "2025-02-14",
                       "quantity": 2, "unit_price": 79.99, "total_amount": 159.98,
                       "status": "completed", "updated_at": "2025-07-15"})
        # Duplicate of a batch-2 order, same order_id, older updated_at.
        orders.append(dict(orders[3]))

    with open(dest / "customers.json", "w", encoding="utf-8") as fh:
        json.dump(customers, fh, indent=2)
    with open(dest / "products.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["product_id", "name", "category",
                                                "unit_price", "effective_date"])
        writer.writeheader()
        writer.writerows(products)
    with open(dest / "orders.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["order_id", "customer_id", "product_id",
                                                "order_date", "quantity", "unit_price",
                                                "total_amount", "status", "updated_at"])
        writer.writeheader()
        writer.writerows(orders)
    return dest
