"""
Repository for reading/writing OHLCV candles.

Handles de-duplication (Section 10: "detect duplicate records") via an
upsert keyed on (symbol, timeframe, timestamp).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from crypto_ai.database.models.market import MarketData


def upsert_candles(session: Session, symbol: str, timeframe: str, rows: list[dict]) -> int:
    """
    Insert candles, skipping/updating any that already exist for the
    same (symbol, timeframe, timestamp). Returns the number of rows
    processed.
    """
    if not rows:
        return 0

    dialect = session.bind.dialect.name
    insert_fn = pg_insert if dialect == "postgresql" else sqlite_insert

    for row in rows:
        row.setdefault("symbol", symbol)
        row.setdefault("timeframe", timeframe)

    stmt = insert_fn(MarketData).values(rows)
    update_cols = {c: getattr(stmt.excluded, c) for c in ("open", "high", "low", "close", "volume")}
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "timeframe", "timestamp"],
        set_=update_cols,
    )
    session.execute(stmt)
    return len(rows)


def get_latest_timestamp(session: Session, symbol: str, timeframe: str) -> dt.datetime | None:
    stmt = (
        select(MarketData.timestamp)
        .where(MarketData.symbol == symbol, MarketData.timeframe == timeframe)
        .order_by(MarketData.timestamp.desc())
        .limit(1)
    )
    result = session.execute(stmt).scalar_one_or_none()
    return result


def load_candles_df(
    session: Session,
    symbol: str,
    timeframe: str,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> pd.DataFrame:
    stmt = select(MarketData).where(MarketData.symbol == symbol, MarketData.timeframe == timeframe)
    if start is not None:
        stmt = stmt.where(MarketData.timestamp >= start)
    if end is not None:
        stmt = stmt.where(MarketData.timestamp <= end)
    stmt = stmt.order_by(MarketData.timestamp.asc())

    rows = session.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    data = [
        {
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ]
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df
