"""
Paper-trading portfolio state (Section 17).

State is read from and written to the database on every call — never
kept only in memory — so the paper-trading engine can be restarted
(power failure, crash, redeploy) and pick up exactly where it left off
using the exchange... except there is no real exchange in PAPER mode,
so the database itself is the source of truth (Section 26 requires
this pattern for real trading; doing it here too means the code path
is already exercised well before real money is involved).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.trading import PaperTrade, PortfolioSnapshot
from crypto_ai.utils.timeutils import index_as_utc


def get_open_trade(session: Session, symbol: str) -> PaperTrade | None:
    stmt = (
        select(PaperTrade)
        .where(PaperTrade.symbol == symbol, PaperTrade.status == "OPEN")
        .order_by(PaperTrade.entry_time.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def get_latest_snapshot(session: Session, mode: str = "PAPER") -> PortfolioSnapshot | None:
    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.mode == mode)
        .order_by(PortfolioSnapshot.timestamp.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def get_cash_balance(session: Session, mode: str = "PAPER") -> float:
    snapshot = get_latest_snapshot(session, mode)
    if snapshot is not None:
        return snapshot.cash_balance
    return get_settings().get("paper_trading.starting_balance_usdt", 1000.0)


def record_snapshot(
    session: Session,
    mode: str,
    timestamp: dt.datetime,
    cash_balance: float,
    position_qty: float,
    current_price: float,
) -> PortfolioSnapshot:
    position_value = position_qty * current_price
    snapshot = PortfolioSnapshot(
        mode=mode,
        timestamp=timestamp,
        cash_balance=cash_balance,
        position_qty=position_qty,
        position_value=position_value,
        total_equity=cash_balance + position_value,
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def get_closed_trades(session: Session, symbol: str) -> list[PaperTrade]:
    stmt = (
        select(PaperTrade)
        .where(PaperTrade.symbol == symbol, PaperTrade.status == "CLOSED")
        .order_by(PaperTrade.exit_time.asc())
    )
    return list(session.execute(stmt).scalars().all())


def get_equity_curve(session: Session, mode: str = "PAPER") -> pd.Series:
    stmt = select(PortfolioSnapshot).where(PortfolioSnapshot.mode == mode).order_by(PortfolioSnapshot.timestamp.asc())
    rows = session.execute(stmt).scalars().all()
    if not rows:
        return pd.Series(dtype=float)
    # Normalize the index to tz-aware UTC: SQLite returns naive datetimes
    # while Postgres returns aware ones, and callers compare this index
    # against tz-aware "now" values.
    return pd.Series(
        [r.total_equity for r in rows], index=index_as_utc([r.timestamp for r in rows]), name="equity",
    )
