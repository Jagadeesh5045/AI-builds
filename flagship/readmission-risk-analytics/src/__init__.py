"""Readmission Risk Analytics.

A production-style health analytics pipeline: synthetic hospital
discharge data, ETL with data-quality quarantine into SQLite, SQL
analytics on 30-day readmissions, a gradient boosting risk model with
plain-English reasons, and a grounded briefing whose numbers are all
re-checked against the warehouse before it is trusted.
"""

__all__ = ["generate", "etl", "analytics", "model", "briefing"]
