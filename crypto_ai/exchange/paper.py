"""
Paper-trading "exchange" adapter (Section 17).

Not a real exchange connection — it just gives the scheduler/dashboard
a consistent interface (`get_balance`, `get_position`) whether the
system is in PAPER mode (this module) or, eventually, TESTNET/LIVE
mode (exchange/live.py). No API key is needed or used here.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from crypto_ai.paper_trading.portfolio import get_cash_balance, get_open_trade


def get_balance(session: Session, mode: str = "PAPER") -> float:
    return get_cash_balance(session, mode=mode)


def get_position(session: Session, symbol: str) -> dict | None:
    trade = get_open_trade(session, symbol)
    if trade is None:
        return None
    return {
        "symbol": trade.symbol,
        "quantity": trade.quantity,
        "entry_price": trade.entry_price,
        "entry_time": trade.entry_time.isoformat(),
    }
