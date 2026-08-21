"""
Startup recovery sequence (Phase 16 / Section 25-27).

    POWER FAILURE -> COMPUTER OFF -> POWER RETURNS -> COMPUTER AUTO
    BOOTS -> APPLICATION AUTO STARTS -> CONNECT TO DATABASE ->
    CONNECT TO EXCHANGE -> CHECK EXCHANGE STATE -> RECOVER -> CONTINUE

What is it?
    The sequence run once at process startup (`run.py start`) before
    the scheduler or API begin normal operation.

Why is it needed?
    Section 26: "Never assume the local database is correct after a
    crash." On every restart, this reconnects to the database with
    retry/backoff, runs a health check sweep, and — critically for a
    future live-trading phase — insists on reconciling against the
    EXCHANGE's own record of open orders/positions rather than trusting
    whatever the local database last recorded, for TESTNET/LIVE modes.

    In RESEARCH/PAPER mode there is no real exchange account to
    reconcile against, so paper-trading state (which lives entirely in
    the database and is read fresh on every engine call — see
    paper_trading/portfolio.py) is already safe to resume from as-is.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.base import wait_for_database
from crypto_ai.monitoring import health
from crypto_ai.monitoring.alerts import EVENT_SYSTEM_STARTED, notify

logger = logging.getLogger(__name__)


class StartupRecoveryError(Exception):
    """Raised when the app cannot safely start (e.g. DB unreachable after retries)."""


def run_startup_sequence(session_factory) -> dict:
    """
    session_factory: a zero-arg callable returning a new Session (e.g.
    crypto_ai.database.base.get_session_factory()) — used because the
    database connection itself isn't guaranteed to exist yet when this
    is called.
    """
    settings = get_settings()

    if not wait_for_database():
        raise StartupRecoveryError(
            "Could not connect to the database after repeated retries. "
            "Check that Postgres is running (`docker compose up -d postgres`) "
            "and DATABASE_URL in .env is correct. See docs/TROUBLESHOOTING.md."
        )

    session: Session = session_factory()
    try:
        summary = health.run_all_health_checks(session)
        notify(
            session, EVENT_SYSTEM_STARTED,
            f"System started in {settings.mode} mode.",
            context={"mode": settings.mode, "health": summary},
        )

        result = {"mode": settings.mode, "health": summary, "exchange_reconciled": True}

        if settings.mode in ("TESTNET", "LIVE"):
            result["exchange_reconciled"] = reconcile_exchange_state(session)

        return result
    finally:
        session.close()


def reconcile_exchange_state(session: Session) -> bool:
    """
    Section 26: for real (testnet/live) trading, the app MUST query the
    exchange for actual open orders/positions after restarting rather
    than trusting the local database.

    This is deliberately a stub that fails safe: Phase 17/18 (testnet
    and live trading) are gated behind the full checklist in
    risk/safety_rules.py and a regulatory access check (Section 4) that
    this repository does not perform on your behalf. Until real
    order-placement code is implemented in exchange/live.py, this
    always returns False and logs why, which keeps
    risk_state.require_state_sync_before_trading (risk.yaml) satisfied
    in its safest position: "not reconciled, don't trade."
    """
    from crypto_ai.database.repositories.events_repo import log_event

    log_event(
        session, component="recovery", event="exchange_reconciliation_not_implemented", severity="WARNING",
        message=(
            "TESTNET/LIVE mode selected, but exchange order/position reconciliation "
            "is not yet implemented (see exchange/live.py). Real trading stays "
            "blocked by the risk engine until this is implemented and the full "
            "Section 59 checklist passes."
        ),
    )
    session.commit()
    return False
