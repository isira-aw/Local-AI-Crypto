"""
Emergency stop (Section 32).

A file-based flag (data_store/EMERGENCY_STOP) rather than only an env
var, because:
  - It must be settable at RUNTIME from the dashboard button, and env
    vars set in one process aren't visible to others.
  - It must survive a restart until a human explicitly resets it
    (Section 32: "Require manual reset") — a file on the host-mounted
    data_store volume does exactly that, with zero extra infrastructure.

RiskManager.evaluate() checks this on every single call, so activating
it takes effect for the very next signal evaluated in ANY process
(dashboard, scheduler, CLI) — no restart required.
"""

from __future__ import annotations

import datetime as dt

from crypto_ai.config.loader import get_settings


def _stop_file():
    return get_settings().data_store_dir / "EMERGENCY_STOP"


def is_emergency_stop_active() -> bool:
    if get_settings().safety.emergency_stop:
        return True
    return _stop_file().exists()


def activate_emergency_stop(reason: str = "") -> None:
    path = _stop_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Activated at {dt.datetime.now(dt.timezone.utc).isoformat()}\nReason: {reason}\n")


def reset_emergency_stop() -> None:
    path = _stop_file()
    if path.exists():
        path.unlink()


def emergency_stop_details() -> dict | None:
    path = _stop_file()
    if not path.exists():
        return None
    return {"active": True, "details": path.read_text()}
