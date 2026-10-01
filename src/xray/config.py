"""Config + secrets loading. The no-synthetic-data hard gates are enforced here."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(os.environ.get("XRAY_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = ROOT / "logs"


class ConfigError(RuntimeError):
    pass


class SourcesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["adzuna"]
    countries: list[str] = Field(min_length=1)
    roles_to_track: list[str] = Field(min_length=1)
    results_per_query: int = Field(gt=0)
    rate_limit_sleep_sec: float = Field(ge=0)
    max_pages: int = Field(gt=0)


def load_yaml(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    path = config_dir / name
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a YAML mapping, got {type(data).__name__}")
    return data


def load_sources(config_dir: Path = CONFIG_DIR) -> SourcesConfig:
    return SourcesConfig.model_validate(load_yaml("sources.yaml", config_dir))


# (file, key) pairs that must be present and literally `false`. Absent counts as a violation.
SYNTHETIC_GATES: tuple[tuple[str, str], ...] = (
    ("project.yaml", "allow_synthetic"),
    ("labeling.yaml", "allow_synthetic_labels"),
)


def enforce_no_synthetic_gates(config_dir: Path = CONFIG_DIR) -> None:
    """Refuse to run unless every synthetic-data gate is explicitly false."""
    for fname, key in SYNTHETIC_GATES:
        cfg = load_yaml(fname, config_dir)
        if key not in cfg:
            raise ConfigError(f"hard gate {fname}:{key} is missing; it must be present and false")
        if cfg[key] is not False:
            raise ConfigError(f"hard gate {fname}:{key} = {cfg[key]!r}; it must be false")


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal KEY=VALUE loader. Real environment variables always win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@lru_cache(maxsize=1)
def adzuna_credentials() -> tuple[str, str]:
    load_dotenv()
    app_id = os.environ.get("ADZUNA_APP_ID", "").strip()
    app_key = os.environ.get("ADZUNA_APP_KEY", "").strip()
    missing = [n for n, v in (("ADZUNA_APP_ID", app_id), ("ADZUNA_APP_KEY", app_key)) if not v]
    if missing:
        raise ConfigError(f"missing credentials: {', '.join(missing)} (set them in .env)")
    from xray.log import register_secret

    register_secret(app_id)
    register_secret(app_key)
    return app_id, app_key
