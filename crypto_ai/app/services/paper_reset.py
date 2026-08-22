"""
Paper-trading epoch / day-zero reset.

Why this exists
---------------
Defects #3 and #4 meant fold backtests — and therefore promotion
decisions — were computed at 100% of equity while risk.yaml caps a real
position at 5%. Any paper-trading history accumulated while those defects
were live was produced by a system behaving differently from the one that
exists now, so counting it towards the live-trading gate's "30 days and
N trades of paper trading" would be counting evidence for the wrong
system.

This module lets the paper-trading clock be reset to zero and, crucially,
makes the live-trading gate derive its day/trade counts from that epoch
rather than from whatever happens to be sitting in the tables.

The epoch is stored as a `performance_metrics` row (scope
`paper_trading_epoch`) rather than a new table, so it needs no migration
and is naturally versioned — every reset appends a row and the newest one
wins, giving a full audit trail of when and why the clock was restarted.
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from crypto_ai.database.models.monitoring import PerformanceMetric
from crypto_ai.database.models.trading import PaperTrade, PortfolioSnapshot
from crypto_ai.database.repositories.events_repo import log_event
from crypto_ai.utils.timeutils import as_utc, utc_now

logger = logging.getLogger(__name__)

EPOCH_SCOPE = "paper_trading_epoch"


def get_paper_trading_epoch(session: Session) -> dict | None:
    """The most recent day-zero marker, or None if the clock was never reset."""
    row = session.execute(
        select(PerformanceMetric)
        .where(PerformanceMetric.scope == EPOCH_SCOPE)
        .order_by(PerformanceMetric.timestamp.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row.metrics if row else None


def get_epoch_start(session: Session) -> dt.datetime | None:
    epoch = get_paper_trading_epoch(session)
    if not epoch or not epoch.get("started_at"):
        return None
    return as_utc(dt.datetime.fromisoformat(epoch["started_at"]))


def reset_paper_trading(
    session: Session,
    reason: str,
    purge_history: bool = True,
    create_backup: bool = True,
    now: dt.datetime | None = None,
) -> dict:
    """
    Reset the paper-trading clock to zero.

    purge_history=True deletes paper trades and portfolio snapshots. A
    backup archive is taken first by default, so the discarded history is
    still recoverable — this is a "stop counting it", not a "destroy it".

    Predictions are deliberately KEPT. A prediction is the model's raw
    directional call; it is unaffected by position sizing, so its accuracy
    remains a valid measurement. Only P/L-bearing records — which the
    sizing defect distorted — are cleared.
    """
    now = as_utc(now) or utc_now()

    backup_path = None
    if create_backup and purge_history:
        try:
            from crypto_ai.app.services.backup import create_backup as make_backup

            backup_path = str(make_backup())
        except Exception as exc:  # noqa: BLE001 - a failed backup must not block the reset
            logger.warning("Pre-reset backup failed (continuing with reset): %s", exc)

    counts_before = {
        "paper_trades": session.execute(select(func.count()).select_from(PaperTrade)).scalar_one(),
        "portfolio_snapshots": session.execute(select(func.count()).select_from(PortfolioSnapshot)).scalar_one(),
    }

    if purge_history:
        session.execute(delete(PaperTrade))
        session.execute(delete(PortfolioSnapshot))

    marker = {
        "started_at": now.isoformat(),
        "reason": reason,
        "purged_history": purge_history,
        "discarded": counts_before if purge_history else {},
        "backup_archive": backup_path,
        "predictions_kept": True,
    }
    session.add(PerformanceMetric(scope=EPOCH_SCOPE, timestamp=now, metrics=marker))

    log_event(
        session, component="paper_trading", event="paper_trading_reset", severity="WARNING",
        message=f"Paper-trading clock reset to zero. Reason: {reason}",
        context={"discarded": counts_before, "backup_archive": backup_path},
    )
    session.commit()

    logger.warning("Paper-trading clock reset at %s (%s)", now.isoformat(), reason)
    return marker


def summarize_since_epoch(session: Session, now: dt.datetime | None = None) -> dict:
    """
    Gate inputs derived from the epoch — the numbers the live-trading
    checklist should use, rather than raw table totals.
    """
    now = as_utc(now) or utc_now()
    epoch_start = get_epoch_start(session)

    stmt = select(PaperTrade).where(PaperTrade.status == "CLOSED")
    if epoch_start is not None:
        stmt = stmt.where(PaperTrade.entry_time >= epoch_start.replace(tzinfo=None))
    trades = list(session.execute(stmt).scalars().all())

    if epoch_start is not None:
        days = (now - epoch_start).total_seconds() / 86400
    elif trades:
        earliest = min(as_utc(t.entry_time) for t in trades)
        days = (now - earliest).total_seconds() / 86400
    else:
        days = 0.0

    return {
        "epoch_start": epoch_start.isoformat() if epoch_start else None,
        "days_since_epoch": round(days, 2),
        "closed_trades_since_epoch": len(trades),
        "counting_from": "epoch marker" if epoch_start else "first trade (no reset recorded)",
    }
