"""
utils/config_loader.py

Loads configuration from:
  - config.yaml (project-level settings)
  - .env (environment variables; optional)
Merges them and exposes a single Config object.

Usage:
    from utils.config_loader import get_config
    cfg = get_config()
    print(cfg.MODELS['diffusion_model_name'])
"""

from __future__ import annotations
import os
import yaml
from dotenv import load_dotenv
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


# Load .env automatically (safe if .env absent)
load_dotenv()


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dict `base` with `override`.
    Returns a new dict (does not mutate input).
    """
    result = dict(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


@dataclass
class Config:
    # Raw dict store for flexibility
    raw: Dict[str, Any] = field(default_factory=dict)

    # Convenience properties / shortcuts (commonly used fields)
    @property
    def DATA_DIR(self) -> str:
        return self.raw.get("paths", {}).get("data_dir", "data/")

    @property
    def MODELS(self) -> Dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def RETRIEVAL(self) -> Dict[str, Any]:
        return self.raw.get("retrieval", {})

    @property
    def DIFFUSION(self) -> Dict[str, Any]:
        return self.raw.get("diffusion", {})

    @property
    def CONSTRAINTS(self) -> Dict[str, Any]:
        return self.raw.get("constraints", {})

    @property
    def EXPERIMENTS(self) -> Dict[str, Any]:
        return self.raw.get("experiments", {})

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.raw.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.raw)


# Module-level config cache
_config_cache: Optional[Config] = None


def load_yaml_file(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return dict (empty dict if file missing).
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def env_overrides_from_prefix(prefix: str = "IDR_") -> Dict[str, Any]:
    """
    Read environment variables with a given prefix and convert them to a nested dict.
    Example: IDR_paths__data_dir=/tmp -> {'paths': {'data_dir': '/tmp'}}
    Uses double-underscore '__' to indicate nesting.
    """
    out: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        # Remove prefix
        keypath = k[len(prefix) :]
        # Support nested keys via '__' separator
        parts = keypath.split("__")
        target = out
        for i, p in enumerate(parts):
            # normalize to lower-case keys (yaml keys likely lower)
            p_norm = p.lower()
            if i == len(parts) - 1:
                # last part -> set value (try numeric/bool conversion)
                converted = _try_convert_env_value(v)
                target[p_norm] = converted
            else:
                if p_norm not in target or not isinstance(target[p_norm], dict):
                    target[p_norm] = {}
                target = target[p_norm]
    return out


def _try_convert_env_value(v: str) -> Any:
    """
    Try to convert environment string values to pythonic types.
    """
    vl = v.strip()
    if vl.lower() in ("true", "false"):
        return vl.lower() == "true"
    # Int?
    try:
        return int(vl)
    except ValueError:
        pass
    # Float?
    try:
        return float(vl)
    except ValueError:
        pass
    # Comma-separated -> list
    if "," in vl:
        return [item.strip() for item in vl.split(",")]
    return vl


def load_config(
    yaml_path: str = "config.yaml", env_prefix: str = "IDR_"
) -> Config:
    """
    Main loader.
    Order of precedence (highest to lowest):
      1. Environment variables with prefix (IDR_)
      2. config.yaml
      3. defaults (none)

    Example environment override:
      IDR_paths__data_dir=/mnt/data
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    yaml_cfg = load_yaml_file(yaml_path)
    env_cfg = env_overrides_from_prefix(env_prefix)

    merged = _deep_update(yaml_cfg, env_cfg)

    # Insert some derived defaults if absent
    paths = merged.get("paths", {})
    paths.setdefault("data_dir", "data/")
    paths.setdefault("embeddings_dir", os.path.join(paths["data_dir"], "embeddings"))
    paths.setdefault("outputs_dir", "outputs/")
    merged["paths"] = paths

    _config_cache = Config(raw=merged)
    return _config_cache


def get_config() -> Config:
    """
    Return cached config (loading from config.yaml if necessary).
    """
    return load_config()


# Small CLI for quick debug
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load and show merged config")
    parser.add_argument("--config", "-c", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--env-prefix",
        "-e",
        default="IDR_",
        help="prefix for environment overrides (double-underscore for nested keys)",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = parser.parse_args()

    cfg = load_config(args.config, args.env_prefix)
    if args.pretty:
        print(json.dumps(cfg.as_dict(), indent=2))
    else:
        print(cfg.as_dict())