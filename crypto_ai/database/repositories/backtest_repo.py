"""Persists a BacktestReport (backtesting/engine.py) to the backtest_runs table."""

from __future__ import annotations

from sqlalchemy.orm import Session

from crypto_ai.backtesting.engine import BacktestReport
from crypto_ai.database.models.trading import BacktestRun


def save_backtest_run(
    session: Session,
    report: BacktestReport,
    strategy_version: str,
    start_time,
    end_time,
    model_version: str | None = None,
    fold_index: int | None = None,
) -> BacktestRun:
    row = BacktestRun(
        strategy_version=strategy_version,
        model_version=model_version,
        symbol=report.symbol,
        timeframe=report.timeframe,
        start_time=start_time,
        end_time=end_time,
        initial_balance=report.initial_balance,
        final_balance=report.final_balance,
        metrics=report.metrics,
        equity_curve=[{"timestamp": ts.isoformat(), "equity": eq} for ts, eq in report.equity_curve.items()],
        fold_index=fold_index,
    )
    session.add(row)
    session.flush()
    return row
