"""
Configuration loader.

What is it?
    A single place that reads config/settings.yaml, config/risk.yaml
    and the .env file, merges them, and hands back typed Python
    objects. Every other module imports its configuration from here
    instead of reading YAML/env directly.

Why is it needed?
    So there is exactly one source of truth for "what is the current
    mode", "what are the risk limits", etc. — and so secrets (.env)
    stay separate from versioned config (settings.yaml).

How do I use it?
    from crypto_ai.config.loader import get_settings, get_risk_config
    settings = get_settings()
    risk = get_risk_config()

Both are cached (loaded once per process). Call `get_settings.cache_clear()`
in tests if you need to reload after changing environment variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent
DATA_STORE_DIR = PROJECT_ROOT / "data_store"

VALID_MODES = ("RESEARCH", "PAPER", "TESTNET", "LIVE")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required config file: {path}. "
            f"If you deleted it, restore it from the repository."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_dotenv_once() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


class DatabaseConfig(BaseModel):
    url: str = "postgresql+psycopg2://crypto_ai:crypto_ai@localhost:5432/crypto_ai"


class LLMConfig(BaseModel):
    enabled: bool = True
    provider: str = "ollama"
    model: str = "llama3.1:8b-instruct-q4_K_M"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 30


class SafetyConfig(BaseModel):
    mode: str = "RESEARCH"
    trading_enabled: bool = False
    real_trading_enabled: bool = False
    emergency_stop: bool = False


class ResourceLimits(BaseModel):
    max_workers: int = 2
    batch_size: int = 256


class Settings(BaseModel):
    """Top-level, merged application settings."""

    raw: dict[str, Any] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    project_root: Path = PROJECT_ROOT
    data_store_dir: Path = DATA_STORE_DIR

    model_config = {"arbitrary_types_allowed": True}

    def get(self, dotted_path: str, default: Any = None) -> Any:
        """Access a raw settings.yaml value by dotted path, e.g. 'data.primary_timeframe'."""
        node: Any = self.raw
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def mode(self) -> str:
        return self.safety.mode

    @property
    def is_real_trading_allowed_by_mode(self) -> bool:
        return self.mode in ("TESTNET", "LIVE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv_once()
    raw = _load_yaml(CONFIG_DIR / "settings.yaml")

    mode = os.getenv("MODE", raw.get("system", {}).get("mode", "RESEARCH")).upper()
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid MODE '{mode}'. Must be one of {VALID_MODES}. "
            f"Check your .env file."
        )

    database = DatabaseConfig(
        url=os.getenv("DATABASE_URL", "postgresql+psycopg2://crypto_ai:crypto_ai@localhost:5432/crypto_ai")
    )

    llm_raw = raw.get("llm", {})
    llm = LLMConfig(
        enabled=_env_bool("LLM_ENABLED", llm_raw.get("enabled", True)),
        provider=os.getenv("LLM_PROVIDER", llm_raw.get("provider", "ollama")),
        model=os.getenv("LLM_MODEL", llm_raw.get("model", "llama3.1:8b-instruct-q4_K_M")),
        base_url=os.getenv("LLM_BASE_URL", llm_raw.get("base_url", "http://localhost:11434")),
        timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", llm_raw.get("timeout_seconds", 30)),
    )

    safety = SafetyConfig(
        mode=mode,
        trading_enabled=_env_bool("TRADING_ENABLED", False),
        real_trading_enabled=_env_bool("REAL_TRADING_ENABLED", False),
        emergency_stop=_env_bool("EMERGENCY_STOP", False),
    )

    resource_raw = raw.get("resource_limits", {})
    resource_limits = ResourceLimits(
        max_workers=_env_int("MAX_WORKERS", resource_raw.get("max_workers", 2)),
        batch_size=_env_int("BATCH_SIZE", resource_raw.get("batch_size", 256)),
    )

    return Settings(
        raw=raw,
        database=database,
        llm=llm,
        safety=safety,
        resource_limits=resource_limits,
    )


@lru_cache(maxsize=1)
def get_risk_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "risk.yaml")


def reload_config() -> None:
    """Clear cached config — mainly for tests that mutate env vars."""
    get_settings.cache_clear()
    get_risk_config.cache_clear()
