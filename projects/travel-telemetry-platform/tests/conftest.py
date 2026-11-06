import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.generate_telemetry import generate_telemetry  # noqa: E402


@pytest.fixture(scope="session")
def telemetry():
    return generate_telemetry()
