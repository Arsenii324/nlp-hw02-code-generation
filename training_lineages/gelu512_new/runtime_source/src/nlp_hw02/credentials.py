"""Read a credential without printing it or initiating network access."""
import os
from .paths import SECRETS_ROOT


def configure_wandb_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    path = SECRETS_ROOT / "wandb_key.txt"
    key = path.read_text().strip()
    if not key:
        raise ValueError("W&B key file is empty")
    os.environ["WANDB_API_KEY"] = key
