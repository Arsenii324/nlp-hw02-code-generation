"""Override these paths in cloud runtimes; no data is copied on import."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get(
    "NLP_HW02_DATA_ROOT", PROJECT_ROOT.parent / "data" / "tinysms_qa_12m"
)).expanduser()
SECRETS_ROOT = Path(os.environ.get(
    "NLP_SECRETS_DIR", PROJECT_ROOT.parent / "secrets"
)).expanduser()
