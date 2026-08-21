"""
Notifications (Section 29).

Default: dashboard/log notifications only — every alert is written as
a system_events row, which the dashboard's System page already reads.
Telegram is OPTIONAL and off unless the user explicitly configures it,
so a beginner never has to set up a bot just to get the system running.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.repositories.events_repo import log_event

logger = logging.getLogger(__name__)

# Section 29's suggested notification types, as event names.
EVENT_SYSTEM_STARTED = "system_started"
EVENT_SYSTEM_STOPPED = "system_stopped"
EVENT_INTERNET_DISCONNECTED = "internet_disconnected"
EVENT_INTERNET_RESTORED = "internet_restored"
EVENT_TRAINING_COMPLETED = "training_completed"
EVENT_NEW_CANDIDATE_MODEL = "new_candidate_model"
EVENT_DRIFT_DETECTED = "drift_detected"
EVENT_PAPER_TRADE_OPENED = "paper_trade_opened"
EVENT_PAPER_TRADE_CLOSED = "paper_trade_closed"
EVENT_DAILY_REPORT = "daily_report"
EVENT_CRITICAL_RISK = "critical_risk_event"


def notify(session: Session, event: str, message: str, severity: str = "INFO", context: dict | None = None) -> None:
    """Always logs to system_events (and the process logger). Also sends
    a Telegram message if NOTIFY_TELEGRAM_ENABLED=true and credentials
    are configured — silently skipped otherwise, and any Telegram
    failure is logged but never raised (a notification must never crash
    the caller)."""
    log_event(session, component="notifications", event=event, severity=severity, message=message, context=context or {})
    session.commit()

    log_fn = {"CRITICAL": logger.critical, "ERROR": logger.error, "WARNING": logger.warning}.get(severity, logger.info)
    log_fn("[%s] %s", event, message)

    _maybe_send_telegram(f"[{severity}] {event}: {message}")


def _maybe_send_telegram(text: str) -> None:
    import os

    if os.getenv("NOTIFY_TELEGRAM_ENABLED", "false").strip().lower() not in ("1", "true", "yes", "on"):
        return
    token = os.getenv("NOTIFY_TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("NOTIFY_TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram notifications enabled but bot token/chat id missing; skipping.")
        return
    try:
        import httpx

        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram notification failed: %s", exc)
