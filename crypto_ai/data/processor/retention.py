"""
Database retention policy enforcement (Phase 16 / Section 54).

What is it?
    Deletes rows older than the configured retention window for
    tables that would otherwise grow without bound — mainly
    high-frequency `market_data` (especially 1-minute candles) and
    `system_events`/`health_checks` logs.

Why is it needed?
    Section 54: "Do not allow unlimited growth." A year of 1-minute
    BTC/USDT candles is a lot of rows; most of that detail is not
    needed once it's outside the model's training window.

What is NEVER deleted here:
    predictions and paper_trades/backtest_runs — Section 54 says trade
    history is "permanent unless manually deleted," and predictions
    are kept long-term so accuracy can be tracked over the system's
    whole lifetime. A timeframe with `null` in settings.yaml's
    `data.retention` is also kept forever (e.g. 1h/4h/1d candles,
    which are cheap to store and useful for long-run charts).
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import delete
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.market import MarketData
from crypto_ai.database.models.monitoring import HealthCheck, SystemEvent
from crypto_ai.database.models.trading import PortfolioSnapshot
from crypto_ai.database.repositories.events_repo import log_event

logger = logging.getLogger(__name__)


def apply_retention_policy(session: Session, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    settings = get_settings()
    retention = settings.get("data.retention", {})
    symbol = settings.get("exchange.symbol", "BTC/USDT")

    deleted = {}

    for timeframe, days in retention.items():
        if timeframe in ("logs", "predictions", "trades"):
            continue  # handled separately / never auto-deleted
        if days is None:
            continue  # "keep forever"
        cutoff = now - dt.timedelta(days=days)
        result = session.execute(
            delete(MarketData).where(
                MarketData.symbol == symbol, MarketData.timeframe == timeframe, MarketData.timestamp < cutoff,
            )
        )
        deleted[f"market_data:{timeframe}"] = result.rowcount or 0

    log_days = retention.get("logs")
    if log_days is not None:
        cutoff = now - dt.timedelta(days=log_days)
        result = session.execute(delete(SystemEvent).where(SystemEvent.timestamp < cutoff))
        deleted["system_events"] = result.rowcount or 0
        result = session.execute(delete(HealthCheck).where(HealthCheck.timestamp < cutoff))
        deleted["health_checks"] = result.rowcount or 0

    # Portfolio snapshots are written on EVERY bar (~105k rows/year on a
    # 5m timeframe), so they need a bound too. Closed trades and
    # predictions are the permanent record and are never touched here;
    # a snapshot is a redundant mark-to-market that can be re-derived.
    snapshot_days = retention.get("portfolio_snapshots", 90)
    if snapshot_days is not None:
        cutoff = now - dt.timedelta(days=snapshot_days)
        result = session.execute(delete(PortfolioSnapshot).where(PortfolioSnapshot.timestamp < cutoff))
        deleted["portfolio_snapshots"] = result.rowcount or 0

    total = sum(deleted.values())
    if total:
        log_event(
            session, component="retention", event="retention_applied", severity="INFO",
            message=f"Deleted {total} rows past their retention window.", context=deleted,
        )
    session.commit()
    return deleted
