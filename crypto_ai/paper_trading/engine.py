"""
Paper trading engine (Phase 12 / Section 17).

What is it?
    Simulates trading BTC/USDT with FAKE money using REAL (or replayed
    historical) prices — the same cost model as the backtester
    (fees + slippage), but driven bar-by-bar as live data arrives
    instead of over a whole historical window at once.

Why is it needed?
    This is "the most important first live stage" (Section 17):
    proof that the strategy behaves sensibly against live market
    conditions before any real money — or even a testnet account — is
    ever involved.

How does it work?
    On every new bar:
      1. Read current state (open position? cash balance?) from the
         database — never from in-memory state alone, so a restart
         mid-trade resumes correctly (Section 25/26).
      2. If flat and the signal is BUY, open a position.
      3. If in a position, check stop-loss/take-profit against the
         bar's high/low, then the signal for a SELL.
      4. Record a portfolio snapshot every bar (mark-to-market), and a
         full paper_trades row on every entry/exit.

Spot-only, single-position-at-a-time, no leverage — matching Section 5.
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.trading import PaperTrade
from crypto_ai.database.repositories.events_repo import log_event
from crypto_ai.paper_trading.portfolio import get_cash_balance, get_open_trade, record_snapshot

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        fee_pct: float | None = None,
        slippage_pct: float | None = None,
        position_size_pct: float | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ):
        settings = get_settings()
        label_cfg = settings.get("labeling", {})
        risk_cfg = settings.get("risk", {}) or {}
        self.symbol = symbol
        self.fee_pct = fee_pct if fee_pct is not None else label_cfg.get("assumed_fee_pct", 0.001)
        self.slippage_pct = slippage_pct if slippage_pct is not None else label_cfg.get("assumed_slippage_pct", 0.0005)
        self.position_size_pct = position_size_pct if position_size_pct is not None else 1.0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def process_bar(
        self,
        session: Session,
        timestamp: dt.datetime,
        close: float,
        signal: str,
        strategy_version: str,
        model_version: str | None = None,
        high: float | None = None,
        low: float | None = None,
        reason: str = "",
    ) -> dict:
        """
        Advances the paper portfolio by one bar. Returns a small dict
        describing what happened: {"action": "OPENED"|"CLOSED"|"NONE", ...}.
        """
        high = high if high is not None else close
        low = low if low is not None else close

        cash = get_cash_balance(session, mode="PAPER")
        open_trade = get_open_trade(session, self.symbol)
        action = {"action": "NONE"}

        if open_trade is not None:
            exit_price = None
            exit_reason = ""
            if self.stop_loss_pct is not None and low <= open_trade.entry_price * (1 - self.stop_loss_pct):
                exit_price = open_trade.entry_price * (1 - self.stop_loss_pct)
                exit_reason = "stop_loss"
            elif self.take_profit_pct is not None and high >= open_trade.entry_price * (1 + self.take_profit_pct):
                exit_price = open_trade.entry_price * (1 + self.take_profit_pct)
                exit_reason = "take_profit"
            elif signal == "SELL":
                exit_price = close
                exit_reason = "signal"

            if exit_price is not None:
                sell_price = exit_price * (1 - self.slippage_pct)
                proceeds = open_trade.quantity * sell_price
                fee = proceeds * self.fee_pct
                cash += proceeds - fee

                entry_cost = open_trade.entry_price * open_trade.quantity + open_trade.fee
                open_trade.exit_time = timestamp
                open_trade.exit_price = sell_price
                open_trade.fee = open_trade.fee + fee
                open_trade.pnl = (proceeds - fee) - entry_cost
                open_trade.status = "CLOSED"
                open_trade.reason = exit_reason
                session.flush()

                log_event(
                    session, component="paper_trading", event="trade_closed", severity="INFO",
                    message=f"Closed {self.symbol} @ {sell_price:.2f} ({exit_reason}), pnl={open_trade.pnl:.2f}",
                    context={"symbol": self.symbol, "pnl": open_trade.pnl, "exit_reason": exit_reason},
                )
                action = {"action": "CLOSED", "trade_id": open_trade.id, "pnl": open_trade.pnl, "exit_reason": exit_reason}
                position_qty = 0.0

            else:
                position_qty = open_trade.quantity

        elif signal == "BUY" and cash > 0:
            buy_price = close * (1 + self.slippage_pct)
            allocation = cash * self.position_size_pct
            fee = allocation * self.fee_pct
            spendable = allocation - fee
            qty = spendable / buy_price
            if qty > 0:
                cash -= allocation
                trade = PaperTrade(
                    symbol=self.symbol,
                    side="BUY",
                    entry_time=timestamp,
                    entry_price=buy_price,
                    quantity=qty,
                    fee=fee,
                    slippage=self.slippage_pct,
                    status="OPEN",
                    strategy_version=strategy_version,
                    model_version=model_version,
                    reason=reason,
                )
                session.add(trade)
                session.flush()
                log_event(
                    session, component="paper_trading", event="trade_opened", severity="INFO",
                    message=f"Opened {self.symbol} @ {buy_price:.2f} qty={qty:.6f}",
                    context={"symbol": self.symbol, "entry_price": buy_price, "quantity": qty},
                )
                action = {"action": "OPENED", "trade_id": trade.id, "entry_price": buy_price, "quantity": qty}
                position_qty = qty
            else:
                position_qty = 0.0
        else:
            position_qty = 0.0

        record_snapshot(session, mode="PAPER", timestamp=timestamp, cash_balance=cash, position_qty=position_qty, current_price=close)
        session.flush()
        return action
