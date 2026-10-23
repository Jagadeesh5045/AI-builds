"""Rule-based NL -> SQL for the aggregate query families rag-router handles.

Covers: total revenue per month, order counts per month, units sold per
product, average order value per region, top product by revenue per month,
revenue per region per month, order counts per product per month.

Raises UnparseableQuery when the question is outside these families; the
caller should fall back to the vector path.
"""
import re


class UnparseableQuery(Exception):
    pass


def _find_month(query, months):
    ql = query.lower()
    for m, num in months.items():
        if re.search(rf"\b{m}\b", ql):
            return m, num
    return None, None


def _find_region(query, regions):
    words = set(re.findall(r"[a-z]+", query.lower()))
    for r in regions:
        if r in words:
            return r
    return None


def _find_product(query, product_names):
    ql = query.lower()
    for name in product_names:
        if name.lower() in ql:
            return name
    return None


def _intent(query, product_names):
    ql = query.lower()
    words = set(re.findall(r"[a-z]+", ql))
    has = lambda *ws: any(w in ql for w in ws)
    has_word = lambda *ws: bool(words & set(ws))
    if has("average", "avg", "mean", "typical") and has("order"):
        return "avg_order_region"
    if has("highest", "top product", "best-selling", "best selling",
           "most revenue", "earned the most", "highest-grossing"):
        return "top_product_month"
    if has("revenue"):
        return ("sum_revenue_region_month"
                if _find_region(query, ("north", "south", "east", "west"))
                else "sum_revenue_month")
    if has_word("units", "unit", "quantity"):
        return "sum_units_product"
    if has("order") and has_word("many", "count", "number"):
        return ("count_orders_product_month"
                if _find_product(query, product_names) is not None
                else "count_orders_month")
    raise UnparseableQuery(f"no aggregate intent detected: {query!r}")


def to_sql(query, product_names, months, regions):
    """Translate an aggregate question into a single safe SQL statement."""
    month_name, month_num = _find_month(query, months)
    region = _find_region(query, regions)
    product = _find_product(query, product_names)
    intent = _intent(query, product_names)

    month_filter = f"order_month_num = {month_num}" if month_num else "true"
    region_filter = f"lower(region) = '{region}'" if region else "true"

    if intent == "sum_revenue_month":
        if not month_num:
            raise UnparseableQuery("month required")
        return (f"SELECT SUM(order_total) AS value FROM orders "
                f"WHERE {month_filter}")
    if intent == "sum_revenue_region_month":
        if not (month_num and region):
            raise UnparseableQuery("month and region required")
        return (f"SELECT SUM(order_total) AS value FROM orders "
                f"WHERE {month_filter} AND {region_filter}")
    if intent == "count_orders_month":
        if not month_num:
            raise UnparseableQuery("month required")
        return f"SELECT COUNT(*) AS value FROM orders WHERE {month_filter}"
    if intent == "sum_units_product":
        if not product:
            raise UnparseableQuery("product required")
        pname = product.replace("'", "''")
        return (f"SELECT SUM(o.quantity) AS value FROM orders o "
                f"JOIN products p ON o.product_id = p.product_id "
                f"WHERE p.name = '{pname}'")
    if intent == "count_orders_product_month":
        if not (product and month_num):
            raise UnparseableQuery("product and month required")
        pname = product.replace("'", "''")
        return (f"SELECT COUNT(*) AS value FROM orders o "
                f"JOIN products p ON o.product_id = p.product_id "
                f"WHERE p.name = '{pname}' AND {month_filter}")
    if intent == "avg_order_region":
        if not region:
            raise UnparseableQuery("region required")
        return f"SELECT AVG(order_total) AS value FROM orders WHERE {region_filter}"
    if intent == "top_product_month":
        if not month_num:
            raise UnparseableQuery("month required")
        return (f"SELECT p.name AS product, SUM(o.order_total) AS revenue "
                f"FROM orders o JOIN products p ON o.product_id = p.product_id "
                f"WHERE {month_filter} GROUP BY p.name "
                f"ORDER BY revenue DESC LIMIT 1")
    raise UnparseableQuery(f"unhandled intent: {intent}")
