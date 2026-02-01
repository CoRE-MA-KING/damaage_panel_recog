from __future__ import annotations

from typing import Any, Dict
import json
from pathlib import Path


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (in place) and return dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | Path | None) -> Dict[str, Any]:
    """Load config from YAML or JSON.

    - YAML requires PyYAML (`pip install pyyaml`)
    - JSON works with standard library.

    If path is None, returns an empty dict (caller merges defaults).
    """
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with `pip install pyyaml` "
            ) from e
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping/object.")
        return data

    raise ValueError(f"Unsupported config extension: {p.suffix}")


def build_effective_config(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return merged config = defaults <- override."""
    merged = json.loads(json.dumps(defaults))  # deep copy via json
    _deep_update(merged, override)
    return merged
