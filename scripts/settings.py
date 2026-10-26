import sys
from pathlib import Path

from dotenv import load_dotenv

root = Path(__file__).resolve().parents[1]

# load_dotenv does not override already set variables, so the priority is:
# shell environment > scripts/.env > root .env
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(root / ".env")

sys.path.append(str(root / "src"))

from dependency.settings import settings  # noqa: E402

__all__ = ["settings"]
