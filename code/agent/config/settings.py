"""
Configuration loading for the agent pipeline.
"""

import os
from pathlib import Path
from typing import Optional


def load_api_keys(path: Optional[Path] = None) -> dict:
    """Load configuration from JSON file (fallback to environment variables)."""
    import json

    cfg_path = path or (Path(__file__).parent / "api_keys.json")
    keys: dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                keys.update(json.load(f))
        except Exception:
            pass

    # Ollama configuration
    keys.setdefault("OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:72b"))
    keys.setdefault("OLLAMA_BASE_URL", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))

    return keys
