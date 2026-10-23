"""DuckDB warehouse: loads the synthetic products/orders CSVs into an
in-memory DuckDB database and exposes clean query views."""
import duckdb


def load_warehouse(products_csv, orders_csv):
    con = duckdb.connect()
    con.execute(
        "CREATE VIEW products AS "
        f"SELECT * FROM read_csv('{products_csv}', header=true)"
    )
    con.execute(
        f"""CREATE VIEW orders AS
           SELECT *,
                  quantity * unit_price AS order_total,
                  CAST(strftime(order_date, '%Y-%m') AS VARCHAR) AS order_month,
                  CAST(month(order_date) AS INTEGER) AS order_month_num
           FROM read_csv('{orders_csv}', header=true)"""
    )
    return con
