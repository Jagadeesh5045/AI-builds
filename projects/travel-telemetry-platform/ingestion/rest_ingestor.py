"""REST-source ingestor.

In production this source is an HTTP REST API (App Store Connect style store
metrics). In this environment there is no external API to call, so the ingestor
consumes a local JSON file that has the exact shape of the REST response. The
parsing and normalisation code is the same code the HTTP path would use; only
the transport differs, and that is stated honestly in the README.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def parse_store_response(payload: dict) -> pd.DataFrame:
    """Normalise a store-metrics REST payload into a daily DataFrame."""
    df = pd.DataFrame(payload["records"])
    df["date"] = pd.to_datetime(df["date"])
    df["negative_review_share"] = (df["rating_1_star"] / df["reviews_count"]).round(4)
    return df.sort_values("date").reset_index(drop=True)


def fetch_store_metrics(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    return parse_store_response(payload)


if __name__ == "__main__":
    here = Path(__file__).parent
    print(fetch_store_metrics(here / "api_stub.json").to_string(index=False))
