"""Central configuration for drift-guard.

All randomness is seeded so the full pipeline is reproducible:
`python run_all.py` twice gives the same metrics.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Reproducibility
RANDOM_STATE = 42

# Data
TEST_SIZE = 0.2          # stratified holdout fraction
VAL_SIZE = 0.2           # fraction of the *training* part used for model selection is done via CV instead
N_CV_FOLDS = 5

# Model registry layout (all under ./artifacts, gitignored)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REGISTRY_FILE = ARTIFACTS_DIR / "experiments.jsonl"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Champion selection metric on cross-validation
CHAMPION_METRIC = "roc_auc"

# Serving
API_HOST = "127.0.0.1"
API_PORT = 8000

# Drift: PSI above this on any feature flags the batch
PSI_WARN_THRESHOLD = 0.25
