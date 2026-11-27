"""AML Alert Analytics.

A production-style finance analytics pipeline: synthetic transaction
monitoring alerts, ETL with data-quality quarantine into SQLite, SQL
analytics on confirmed-suspicious rates, a gradient boosting risk model
with plain-English reasons, and a grounded briefing whose numbers are
all re-checked against the warehouse before it is trusted.
"""

__all__ = ["generate", "etl", "analytics", "model", "briefing"]
