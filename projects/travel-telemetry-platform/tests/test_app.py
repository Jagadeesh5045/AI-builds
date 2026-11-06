"""Streamlit AppTest: the entry point and every page must run without exceptions."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"


def _run(script: Path) -> AppTest:
    at = AppTest.from_file(str(script), default_timeout=120)
    at.run()
    assert not at.exception, f"{script.name} raised: {at.exception}"
    return at


def test_entry_point_runs():
    at = _run(APP_DIR / "app.py")
    assert len(at.metric) >= 4


def test_domain_pages_run():
    for page in ["Bookings.py", "Adoption.py", "Engagement.py", "App_Health.py", "Anomalies.py"]:
        _run(APP_DIR / "pages" / page)
