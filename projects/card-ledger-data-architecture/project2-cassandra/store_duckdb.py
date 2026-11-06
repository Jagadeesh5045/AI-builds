"""DuckDB implementation of the identical per-query tables.

Used ONLY as the verification fallback if a Cassandra server cannot be
run in this environment, and by the tests. The tables, keys and query
shapes mirror model/schema.cql one for one: same denormalisation, same
partition and clustering columns, same access patterns. DuckDB SQL
needs ORDER BY where Cassandra uses clustering order; that difference
is noted in RESULTS.md and does not change what is being proved, which
is that each per-query table answers its pattern on its own.
"""
import duckdb

DDL = [
    """CREATE TABLE transactions_by_member_month (
         member_id VARCHAR, month_bucket VARCHAR, transaction_date TIMESTAMP,
         txn_id VARCHAR, card_id VARCHAR, merchant_id VARCHAR,
         merchant_name VARCHAR, amount DECIMAL(12,2), currency VARCHAR,
         txn_type VARCHAR, status VARCHAR)""",
    """CREATE TABLE transactions_by_card (
         card_id VARCHAR, transaction_date TIMESTAMP, txn_id VARCHAR,
         member_id VARCHAR, merchant_id VARCHAR, merchant_name VARCHAR,
         card_number_token VARCHAR, amount DECIMAL(12,2), currency VARCHAR,
         txn_type VARCHAR, status VARCHAR)""",
    """CREATE TABLE transactions_by_merchant_day (
         merchant_id VARCHAR, day_bucket DATE, transaction_date TIMESTAMP,
         txn_id VARCHAR, card_id VARCHAR, member_id VARCHAR,
         amount DECIMAL(12,2), currency VARCHAR, txn_type VARCHAR,
         status VARCHAR)""",
    """CREATE TABLE statements_by_member (
         member_id VARCHAR, period_start DATE, statement_id VARCHAR,
         period_end DATE, opening_balance DECIMAL(12,2),
         closing_balance DECIMAL(12,2), due_date DATE)""",
    """CREATE TABLE settlements_by_merchant (
         merchant_id VARCHAR, settlement_date DATE, settlement_id VARCHAR,
         gross_amount DECIMAL(12,2), fees DECIMAL(12,2),
         net_amount DECIMAL(12,2), txn_count INTEGER)""",
    """CREATE TABLE auth_events_by_card (
         card_id VARCHAR, event_time TIMESTAMP, event_id VARCHAR,
         member_id VARCHAR, merchant_id VARCHAR, amount DECIMAL(12,2),
         currency VARCHAR, outcome VARCHAR, reason VARCHAR)""",
]


def build(data):
    con = duckdb.connect(":memory:")
    for stmt in DDL:
        con.execute(stmt)
    s = str
    con.executemany(
        "INSERT INTO transactions_by_member_month VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(s(t["member_id"]), t["transaction_date"].strftime("%Y-%m"),
          t["transaction_date"], s(t["txn_id"]), s(t["card_id"]),
          s(t["merchant_id"]), t["merchant_name"], t["amount"],
          t["currency"], t["txn_type"], t["status"])
         for t in data["transactions"]])
    con.executemany(
        "INSERT INTO transactions_by_card VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(s(t["card_id"]), t["transaction_date"], s(t["txn_id"]),
          s(t["member_id"]), s(t["merchant_id"]), t["merchant_name"],
          t["card_number_token"], t["amount"], t["currency"],
          t["txn_type"], t["status"]) for t in data["transactions"]])
    con.executemany(
        "INSERT INTO transactions_by_merchant_day VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(s(t["merchant_id"]), t["transaction_date"].date(),
          t["transaction_date"], s(t["txn_id"]), s(t["card_id"]),
          s(t["member_id"]), t["amount"], t["currency"], t["txn_type"],
          t["status"]) for t in data["transactions"]])
    con.executemany(
        "INSERT INTO statements_by_member VALUES (?,?,?,?,?,?,?)",
        [(s(x["member_id"]), x["period_start"], s(x["statement_id"]),
          x["period_end"], x["opening_balance"], x["closing_balance"],
          x["due_date"]) for x in data["statements"]])
    con.executemany(
        "INSERT INTO settlements_by_merchant VALUES (?,?,?,?,?,?,?)",
        [(s(x["merchant_id"]), x["settlement_date"], s(x["settlement_id"]),
          x["gross_amount"], x["fees"], x["net_amount"], x["txn_count"])
         for x in data["settlements"]])
    con.executemany(
        "INSERT INTO auth_events_by_card VALUES (?,?,?,?,?,?,?,?,?)",
        [(s(e["card_id"]), e["event_time"], s(e["event_id"]),
          s(e["member_id"]), s(e["merchant_id"]), e["amount"],
          e["currency"], e["outcome"], e["reason"])
         for e in data["auth_events"]])
    return con
