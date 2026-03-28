from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "demo",
    "simulation": {
        "seed": 42,
        "max_steps": 10,
        "threshold": 0.6,
        "mode": "fair_replay",
    },
    "defaults": {
        "goal": "Summarize a research paper concisely",
        "task": "Summarize a research paper concisely",
    },
    "embedding": {
        "provider": "demo",
        "gemini_model": "models/text-embedding-004",
    },
    "agent": {
        "provider": "demo",
        "gemini_model": "gemini-2.0-flash",
        "use_structured_output": True,
    },
    "intervention": {
        "provider": "demo",
        "gemini_model": "gemini-2.0-flash-lite",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = "config.json") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return dict(DEFAULT_CONFIG)

    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    return _deep_merge(DEFAULT_CONFIG, user_config)
