"""Helper for writing structured system_events rows (Section 34)."""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from crypto_ai.database.models.monitoring import SystemEvent

_SECRET_MARKERS = ("api_secret", "api_key", "password", "secret")


def _scrub(context: dict) -> dict:
    """Defence in depth: never let a secret-looking key end up in the DB."""
    return {
        k: ("***REDACTED***" if any(m in k.lower() for m in _SECRET_MARKERS) else v)
        for k, v in context.items()
    }


def log_event(
    session: Session,
    component: str,
    event: str,
    severity: str = "INFO",
    message: str = "",
    context: dict | None = None,
) -> SystemEvent:
    row = SystemEvent(
        timestamp=dt.datetime.now(dt.timezone.utc),
        component=component,
        event=event,
        severity=severity,
        message=message,
        context=_scrub(context or {}),
    )
    session.add(row)
    session.flush()
    return row
