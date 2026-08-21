"""
Local LLM client (Phase 13 / Section 18/40).

What is it?
    A minimal HTTP client for a locally-running LLM server — either
    Ollama or an LM Studio (OpenAI-compatible) server. Both run
    entirely on the user's own machine; nothing here ever calls an
    external/cloud AI API (Section 55: offline/local-first).

Why is it needed?
    The rest of the app must never crash or block just because the
    LLM is off, unreachable, or slow. This client is the one place
    that talks HTTP to it, catches every failure, and returns None
    instead of raising — callers (llm/analyst.py) always have a
    non-LLM fallback path.

How do I start it?
    Install Ollama (recommended for beginners) or LM Studio on your
    own machine, pull/load a model, and make sure LLM_ENABLED=true and
    LLM_BASE_URL point at it. See docs/INSTALLATION.md. Setting
    LLM_ENABLED=false disables this entirely — data collection, ML,
    backtesting and paper trading all keep working.
"""

from __future__ import annotations

import logging

import httpx

from crypto_ai.config.loader import get_settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised internally; callers should prefer the *_safe wrappers that never raise."""


class BaseLLMClient:
    def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(str(exc)) from exc


class LMStudioClient(BaseLLMClient):
    """LM Studio exposes an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            resp = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LLMUnavailableError(str(exc)) from exc


def get_llm_client() -> BaseLLMClient | None:
    """Returns None if the LLM is disabled — callers must handle that."""
    settings = get_settings()
    if not settings.llm.enabled:
        return None
    if settings.llm.provider == "ollama":
        return OllamaClient(settings.llm.base_url, settings.llm.model, settings.llm.timeout_seconds)
    if settings.llm.provider == "lmstudio":
        return LMStudioClient(settings.llm.base_url, settings.llm.model, settings.llm.timeout_seconds)
    logger.warning("Unknown LLM provider '%s'; LLM features disabled.", settings.llm.provider)
    return None


def generate_safe(prompt: str, system: str | None = None) -> str | None:
    """
    Never raises. Returns None if the LLM is disabled, unreachable, or
    errors — the caller (llm/analyst.py) always has a template-based
    fallback for this case.
    """
    client = get_llm_client()
    if client is None:
        return None
    try:
        return client.generate(prompt, system)
    except LLMUnavailableError as exc:
        logger.warning("LLM call failed, continuing without an explanation: %s", exc)
        return None


def check_llm_health() -> dict:
    """Used by monitoring/health.py and the dashboard's System page."""
    settings = get_settings()
    if not settings.llm.enabled:
        return {"status": "DISABLED", "provider": None, "model": None}

    result = generate_safe("Reply with the single word: OK")
    if result is None:
        return {"status": "DOWN", "provider": settings.llm.provider, "model": settings.llm.model}
    return {"status": "OK", "provider": settings.llm.provider, "model": settings.llm.model}
