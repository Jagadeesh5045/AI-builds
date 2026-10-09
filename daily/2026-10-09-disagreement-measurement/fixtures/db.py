"""Tiny SQLite helpers for the demo codebase."""

import sqlite3


def connect(db_path):
    """Open a SQLite connection to the given path."""
    return sqlite3.connect(db_path)


def fetch_all(conn, query, params=()):
    """Run a query and return every row as a list of tuples."""
    cur = conn.cursor()
    cur.execute(query, params)
    return cur.fetchall()
