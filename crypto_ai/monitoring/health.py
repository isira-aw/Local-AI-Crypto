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

# Provenance of an exchange health probe. Only SOURCE_REAL counts towards
# the live-trading gate — see has_verified_real_exchange_connectivity().
SOURCE_REAL = "binance_live_api"
SOURCE_SIMULATED = "injected_or_simulated"


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

    # A health check should fail fast — the multi-attempt exponential
    # backoff in market_data.py is meant for a long-running download, not
    # for "is the exchange up right now". Build a separate client rather
    # than mutating the caller's object, which would silently disable
    # retries for whatever else is using it.
    probe = MarketDataClient(
        source=client.source if client is not None else None,
        retry_policy=RetryPolicy(max_attempts=1),
    ) if client is not None else MarketDataClient(retry_policy=RetryPolicy(max_attempts=1))

    # Record whether this probe actually reached Binance or went through an
    # injected/simulated source. The live-trading gate's
    # "exchange connection stable" item may ONLY be satisfied by real
    # probes — otherwise a test harness or simulator could tick off a
    # safety check that exists to prove real connectivity.
    from crypto_ai.exchange.market_data import CcxtBinanceSource

    is_real = isinstance(probe.source, CcxtBinanceSource)
    source_kind = SOURCE_REAL if is_real else SOURCE_SIMULATED

    start = time.monotonic()
    try:
        probe.fetch_ohlcv_page(
            "BTC/USDT", "5m", dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30), limit=1
        )
        latency_ms = (time.monotonic() - start) * 1000
        return _record(session, "exchange", STATUS_OK, latency_ms, {"source": source_kind})
    except Exception as exc:  # noqa: BLE001
        return _record(session, "exchange", STATUS_DOWN, None, {"error": str(exc), "source": source_kind})


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


def has_verified_real_exchange_connectivity(
    session: Session, min_consecutive_ok: int = 20, within_days: int = 7,
) -> tuple[bool, str]:
    """
    Section 59's "exchange connection stable" item.

    Returns (satisfied, explanation). This deliberately CANNOT be satisfied
    by a simulated or injected client: it requires health checks whose
    recorded source is the real Binance API. Running the end-to-end
    validation harness — which injects a fake exchange — will never tick
    this box, which is the point.
    """
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=within_days)
    rows = (
        session.query(HealthCheck)
        .filter(HealthCheck.component == "exchange", HealthCheck.timestamp >= cutoff.replace(tzinfo=None))
        .order_by(HealthCheck.timestamp.desc())
        .limit(min_consecutive_ok * 3)
        .all()
    )

    real_rows = [r for r in rows if (r.details or {}).get("source") == SOURCE_REAL]
    if not real_rows:
        return False, (
            f"no health checks against the real Binance API in the last {within_days} days "
            f"({len(rows)} exchange check(s) found, all simulated/injected). "
            f"This item cannot be satisfied by the validation harness."
        )

    recent_ok = 0
    for r in real_rows:
        if r.status == STATUS_OK:
            recent_ok += 1
        else:
            break

    if recent_ok < min_consecutive_ok:
        return False, (
            f"only {recent_ok} consecutive successful real-API checks "
            f"(need {min_consecutive_ok} within {within_days} days)"
        )
    return True, f"{recent_ok} consecutive successful checks against the real Binance API"
