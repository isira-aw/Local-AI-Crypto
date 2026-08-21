"""
Persistence for the structured Signal objects described in Section 19.

Why this exists: the signal is the auditable record of *what the system
decided and why* — the model's raw prediction, the risk engine's verdict,
the reason codes explaining any override, and (optionally) the LLM's
plain-language explanation. Without persisting it, the `signals` table
stays empty and there is no history of decisions to review, only the
trades that happened to result from them.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.database.models.trading import Signal as SignalRow
from crypto_ai.strategies.signal_engine import Signal as SignalObj


def save_signal(
    session: Session,
    signal: SignalObj,
    prediction_id: int | None = None,
) -> SignalRow:
    row = SignalRow(
        prediction_id=prediction_id,
        symbol=signal.symbol,
        timestamp=signal.timestamp,
        signal=signal.signal,
        confidence=signal.confidence,
        risk_score=signal.risk_score,
        model_version=signal.model_version,
        strategy_version=signal.strategy_version,
        reason_codes=signal.reason_codes,
        llm_explanation=signal.llm_explanation,
    )
    session.add(row)
    session.flush()
    return row


def get_recent_signals(session: Session, symbol: str | None = None, limit: int = 50) -> list[SignalRow]:
    stmt = select(SignalRow)
    if symbol:
        stmt = stmt.where(SignalRow.symbol == symbol)
    stmt = stmt.order_by(SignalRow.timestamp.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


def get_latest_signal(session: Session, symbol: str) -> SignalRow | None:
    stmt = (
        select(SignalRow)
        .where(SignalRow.symbol == symbol)
        .order_by(SignalRow.timestamp.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def count_signals_by_type(session: Session, symbol: str, since: dt.datetime) -> dict:
    """Used by the daily report's 'Predictions: BUY/SELL/WAIT' section."""
    rows = session.execute(
        select(SignalRow).where(SignalRow.symbol == symbol, SignalRow.timestamp >= since)
    ).scalars().all()
    counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
    for r in rows:
        if r.signal in counts:
            counts[r.signal] += 1
    return counts
