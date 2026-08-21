"""
Health checks (Section 25/28).

What is it?
    Point-in-time checks of the database, exchange connectivity, and
    local LLM — written to the `health_checks` table so the dashboard's
    System page and the daily report can show "is everything actually
    working" at a glance.

Why is it needed?
    Section 25: the app must survive brief outages of any of these
    without crashing. Before it can *recover* from an outage, it has
    to *detect* one — that's what this module is for.
"""

from __future__ import annotations

import datetime as dt
import time

from sqlalchemy import text
from sqlalchemy.orm import Session

from crypto_ai.database.models.monitoring import HealthCheck
from crypto_ai.llm.client import check_llm_health

STATUS_OK = "OK"
STATUS_DEGRADED = "DEGRADED"
STATUS_DOWN = "DOWN"


def _record(session: Session, component: str, status: str, latency_ms: float | None, details: dict) -> HealthCheck:
    row = HealthCheck(
        timestamp=dt.datetime.now(dt.timezone.utc),
        component=component,
        status=status,
        latency_ms=latency_ms,
        details=details,
    )
    session.add(row)
    session.flush()
    return row


def check_database(session: Session) -> HealthCheck:
    start = time.monotonic()
    try:
        session.execute(text("SELECT 1"))
        latency_ms = (time.monotonic() - start) * 1000
        return _record(session, "database", STATUS_OK, latency_ms, {})
    except Exception as exc:  # noqa: BLE001
        return _record(session, "database", STATUS_DOWN, None, {"error": str(exc)})


def check_exchange(session: Session, client=None) -> HealthCheck:
    """client: exchange.market_data.MarketDataClient — injected so tests
    don't need real network access."""
    from crypto_ai.exchange.market_data import MarketDataClient, RetryPolicy

    client = client or MarketDataClient()
    # A health check should fail fast — the multi-attempt exponential
    # backoff in market_data.py is meant for a long-running download,
    # not for "is the exchange up right now".
    client.retry_policy = RetryPolicy(max_attempts=1)
    start = time.monotonic()
    try:
        client.fetch_ohlcv_page(
            "BTC/USDT", "5m", dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30), limit=1
        )
        latency_ms = (time.monotonic() - start) * 1000
        return _record(session, "exchange", STATUS_OK, latency_ms, {})
    except Exception as exc:  # noqa: BLE001
        return _record(session, "exchange", STATUS_DOWN, None, {"error": str(exc)})


def check_llm(session: Session) -> HealthCheck:
    result = check_llm_health()
    status_map = {"OK": STATUS_OK, "DOWN": STATUS_DOWN, "DISABLED": STATUS_DEGRADED}
    return _record(session, "llm", status_map.get(result["status"], STATUS_DOWN), None, result)


def run_all_health_checks(session: Session, exchange_client=None) -> dict:
    db_check = check_database(session)
    llm_check = check_llm(session)
    checks = {"database": db_check, "llm": llm_check}
    try:
        checks["exchange"] = check_exchange(session, exchange_client)
    except Exception:  # noqa: BLE001 - health checks themselves must never crash the caller
        pass
    session.commit()
    return {name: {"status": c.status, "latency_ms": c.latency_ms} for name, c in checks.items()}
