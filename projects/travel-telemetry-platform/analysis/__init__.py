"""Analysis layer: pure, filter-agnostic functions. No Streamlit imports here.

Layer rule: analysis functions take a DataFrame and parameters and return new
DataFrames or dicts. They never read filter or session state; callers pass the
already-filtered frame in. That keeps every function unit-testable.
"""
