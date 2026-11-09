"""Seeded clickstream generator for sql-session-lab.

Builds a realistic product-analytics event stream for a fictional UK online
store: users arrive through a channel, browse in sessions, and move through
a view -> search -> product -> cart -> checkout -> purchase funnel.

Some sessions deliberately contain a long pause (over the 30-minute session
timeout) so the sessionisation SQL has real splits to find. All randomness
comes from one seeded ``random.Random`` instance, so the dataset is fully
reproducible: same seed, same events, same revenue to the penny.

Standard library only.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta

PRODUCTS = [
    ("P01", "Trail running shoes", 7499),
    ("P02", "Merino base layer", 4200),
    ("P03", "Waterproof jacket", 12999),
    ("P04", "Insulated flask 1L", 1899),
    ("P05", "Trekking poles pair", 5499),
    ("P06", "Head torch 400lm", 2799),
    ("P07", "Daypack 25L", 6899),
    ("P08", "Wool walking socks 3pk", 1499),
    ("P09", "Down jacket", 15999),
    ("P10", "Trail gaiters", 2299),
    ("P11", "Camping stove", 4799),
    ("P12", "Dry bag set", 1999),
]
CHANNELS = ["organic_search", "paid_search", "email", "social", "direct"]
DEVICES = ["mobile", "desktop", "tablet"]
START = datetime(2026, 8, 1, 0, 0, 0)
DAYS = 92  # 2026-08-01 .. 2026-10-31

SCHEMA = """
CREATE TABLE events (
    event_id    INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL,
    ts          TEXT NOT NULL,          -- ISO 'YYYY-MM-DD HH:MM:SS'
    event_type  TEXT NOT NULL,          -- view/search/product_view/add_to_cart/checkout/purchase
    product_id  TEXT,
    revenue_pence INTEGER NOT NULL DEFAULT 0,
    channel     TEXT NOT NULL,
    device      TEXT NOT NULL
);
CREATE INDEX idx_events_user_ts ON events(user_id, ts);
CREATE INDEX idx_events_ts ON events(ts);
"""


def _session_events(rng: random.Random, user_id: str, start: datetime,
                    channel: str, device: str) -> list[tuple]:
    """One browsing session's events. Steps drop off probabilistically, and
    roughly 1 session in 7 has a >30 min pause in the middle."""
    rows = []
    t = start
    product = rng.choice(PRODUCTS)

    def emit(event_type, product_id=None, revenue=0, gap_minutes=None):
        nonlocal t
        gap = gap_minutes if gap_minutes is not None else rng.uniform(0.5, 7.0)
        t = t + timedelta(minutes=gap)
        rows.append((user_id, t.strftime("%Y-%m-%d %H:%M:%S"), event_type,
                     product_id, revenue, channel, device))

    emit("view", gap_minutes=0)
    if rng.random() < 0.12:  # long pause: sessionisation must split here
        emit("view", gap_minutes=rng.uniform(35, 90))
    if rng.random() < 0.82:
        emit("search")
    if rng.random() < 0.74:
        emit("product_view", product[0])
    else:
        return rows
    if rng.random() < 0.52:
        emit("add_to_cart", product[0])
    else:
        return rows
    if rng.random() < 0.68:
        emit("checkout", product[0])
    else:
        return rows
    if rng.random() < 0.76:
        qty = rng.choice([1, 1, 1, 2])
        emit("purchase", product[0], product[2] * qty)
    return rows


def generate(db_path: str, seed: int = 42, n_users: int = 800) -> dict:
    """Create the SQLite database at ``db_path`` and return summary counts."""
    rng = random.Random(seed)
    all_rows: list[tuple] = []
    for i in range(1, n_users + 1):
        user_id = f"U{i:04d}"
        channel = rng.choice(CHANNELS)
        device = rng.choices(DEVICES, weights=[0.55, 0.35, 0.10])[0]
        n_sessions = rng.choices([1, 2, 3, 4, 5, 6], weights=[30, 28, 20, 12, 7, 3])[0]
        for _ in range(n_sessions):
            day = rng.randrange(DAYS)
            start = START + timedelta(days=day, hours=rng.randrange(7, 23),
                                      minutes=rng.randrange(60))
            all_rows.extend(_session_events(rng, user_id, start, channel, device))
    all_rows.sort(key=lambda r: (r[1], r[0]))
    con = sqlite3.connect(db_path)
    try:
        con.executescript("DROP TABLE IF EXISTS events;" + SCHEMA)
        con.executemany(
            "INSERT INTO events (user_id, ts, event_type, product_id,"
            " revenue_pence, channel, device) VALUES (?,?,?,?,?,?,?)",
            all_rows,
        )
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        revenue = con.execute(
            "SELECT COALESCE(SUM(revenue_pence),0) FROM events"
            " WHERE event_type='purchase'").fetchone()[0]
        purchasers = con.execute(
            "SELECT COUNT(DISTINCT user_id) FROM events"
            " WHERE event_type='purchase'").fetchone()[0]
    finally:
        con.close()
    return {"events": total, "users": n_users,
            "revenue_pence": revenue, "purchasers": purchasers}
