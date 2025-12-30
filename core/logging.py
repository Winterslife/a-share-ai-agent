from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path

def repo_root() -> Path:
    """Returns the repository root directory."""
    return Path(__file__).parent.parent

def runs_dir() -> Path:
    """Returns the data/runs directory, ensuring it exists."""
    path = repo_root() / "data" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def cache_dir() -> Path:
    """Returns the data/cache directory, ensuring it exists."""
    path = repo_root() / "data" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path

def new_run_id(prefix: str = "run") -> str:
    """Generates a new run ID: {prefix}_{YYYYMMDD_HHMMSS}_{random6}."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = secrets.token_hex(3)  # 3 bytes = 6 hex chars
    return f"{prefix}_{now}_{random_suffix}"

def save_run_json(run_id: str, payload: dict) -> Path:
    """Saves a JSON artifact under data/runs/<run_id>.json."""
    file_path = runs_dir() / f"{run_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return file_path
