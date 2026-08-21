"""
Paper-trading simulator loop and performance summary (Phase 12).

`replay_bars` drives the engine bar-by-bar from any source of OHLCV +
signal data — live data one bar at a time (called from the scheduler,
Section 44) or a whole historical DataFrame at once (useful for
catching up after downtime, or for tests). `signal_fn` is injected
rather than hard-coded so this module doesn't need to know about
model/LLM/risk internals — the caller decides how a signal is produced.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sqlalchemy.orm import Session

from crypto_ai.backtesting import metrics as m
from crypto_ai.paper_trading.engine import PaperTradingEngine
from crypto_ai.paper_trading.portfolio import get_closed_trades, get_equity_curve

SignalFn = Callable[[pd.Series], tuple[str, str, str | None, str]]
"""row -> (signal, strategy_version, model_version, reason)"""


def replay_bars(session: Session, engine: PaperTradingEngine, df: pd.DataFrame, signal_fn: SignalFn) -> list[dict]:
    """
    df: columns [timestamp, close] and optionally [high, low].
    Returns the list of per-bar actions (mostly {"action": "NONE"}).
    """
    has_hl = {"high", "low"}.issubset(df.columns)
    actions = []
    for _, row in df.iterrows():
        signal, strategy_version, model_version, reason = signal_fn(row)
        action = engine.process_bar(
            session,
            timestamp=row["timestamp"],
            close=float(row["close"]),
            signal=signal,
            strategy_version=strategy_version,
            model_version=model_version,
            high=float(row["high"]) if has_hl else None,
            low=float(row["low"]) if has_hl else None,
            reason=reason,
        )
        actions.append(action)
    return actions


def summarize_paper_trading(session: Session, symbol: str, timeframe: str = "5m") -> dict:
    """
    The Section 17/21/45 numbers: virtual balance, P/L, win rate,
    profit factor, drawdown, Sharpe — computed from the actual
    paper_trades + portfolio tables, not re-simulated.
    """
    equity_curve = get_equity_curve(session, mode="PAPER")
    trades = get_closed_trades(session, symbol)
    pnls = [t.pnl for t in trades if t.pnl is not None]

    if equity_curve.empty:
        return {
            "n_trades": 0, "starting_balance": None, "current_equity": None,
            "total_return_pct": None, "max_drawdown_pct": None, "sharpe_ratio": None,
            "win_rate_pct": None, "profit_factor": None,
        }

    avg_win, avg_loss = m.average_win_loss(pnls)
    metrics = {
        "n_trades": len(trades),
        "starting_balance": float(equity_curve.iloc[0]),
        "current_equity": float(equity_curve.iloc[-1]),
        "total_return_pct": m.total_return_pct(equity_curve),
        "max_drawdown_pct": m.max_drawdown_pct(equity_curve),
        "sharpe_ratio": m.sharpe_ratio(equity_curve, timeframe),
        "win_rate_pct": m.win_rate_pct(pnls),
        "profit_factor": m.profit_factor(pnls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }
    metrics["suspicious_flags"] = m.suspicious_result_flags(metrics, len(trades))
    return metrics
