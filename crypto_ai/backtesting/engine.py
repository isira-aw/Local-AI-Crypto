"""
Backtesting engine (Phase 10 / Section 16).

What is it?
    A realistic, bar-by-bar spot-only long/flat simulator. It never
    looks at future bars: at bar i it only ever acts on the signal and
    price available at bar i.

Why is it needed?
    A model with 55% classification accuracy might still lose money
    after fees and slippage. This is what actually answers "would this
    have made money" — see Section 6's success criteria.

What goes in?
    A DataFrame with [timestamp, close, signal] (signal in
    {BUY, SELL, HOLD} — BUY/SELL are interpreted as "enter/exit a long
    position", HOLD as "no action"), plus cost assumptions.

What comes out?
    A BacktestReport: initial/final balance, trade list, equity curve,
    and the full metrics dict from metrics.py.

How does it work?
    Spot-only, no shorting, no leverage (Section 5). At most one open
    position at a time. Each bar:
      1. If in a position, check stop-loss/take-profit against that
         bar's low/high.
      2. Otherwise, act on the bar's signal (enter on BUY if flat,
         exit on SELL if in position).
      3. Mark the portfolio to market at the bar's close for the
         equity curve, regardless of whether a trade happened.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from crypto_ai.backtesting import metrics as m


@dataclass
class Trade:
    entry_time: dt.datetime
    entry_price: float
    exit_time: dt.datetime | None = None
    exit_price: float | None = None
    quantity: float = 0.0
    fee_paid: float = 0.0
    slippage_cost: float = 0.0
    pnl: float | None = None
    exit_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "fee_paid": self.fee_paid,
            "slippage_cost": self.slippage_cost,
            "pnl": self.pnl,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestReport:
    symbol: str
    timeframe: str
    initial_balance: float
    final_balance: float
    trades: list[Trade]
    equity_curve: pd.Series  # indexed by timestamp
    buy_and_hold_final_balance: float
    fees_paid: float
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "net_profit": self.final_balance - self.initial_balance,
            "return_pct": (self.final_balance / self.initial_balance - 1) * 100 if self.initial_balance else 0.0,
            "n_trades": len(self.trades),
            "fees_paid": self.fees_paid,
            "buy_and_hold_final_balance": self.buy_and_hold_final_balance,
            "buy_and_hold_return_pct": (
                (self.buy_and_hold_final_balance / self.initial_balance - 1) * 100 if self.initial_balance else 0.0
            ),
            "metrics": self.metrics,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq} for ts, eq in self.equity_curve.items()
            ],
        }


class BacktestEngine:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 1.0,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        timeframe: str = "5m",
    ):
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.timeframe = timeframe

    def run(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> BacktestReport:
        """
        df: columns [timestamp, close, signal], optionally [high, low]
            for intrabar stop-loss/take-profit checks (falls back to
            close if not provided).
        """
        required = {"timestamp", "close", "signal"}
        if not required.issubset(df.columns):
            raise ValueError(f"df must contain columns {required}, got {list(df.columns)}")

        cash = self.initial_balance
        position_qty = 0.0
        entry_price = 0.0
        current_trade: Trade | None = None
        trades: list[Trade] = []
        equity_values = []
        fees_paid = 0.0

        has_hl = {"high", "low"}.issubset(df.columns)

        for _, row in df.iterrows():
            ts = row["timestamp"]
            close = float(row["close"])
            high = float(row["high"]) if has_hl else close
            low = float(row["low"]) if has_hl else close
            signal = row["signal"]

            if position_qty > 0:
                exit_price = None
                exit_reason = ""
                if self.stop_loss_pct is not None and low <= entry_price * (1 - self.stop_loss_pct):
                    exit_price = entry_price * (1 - self.stop_loss_pct)
                    exit_reason = "stop_loss"
                elif self.take_profit_pct is not None and high >= entry_price * (1 + self.take_profit_pct):
                    exit_price = entry_price * (1 + self.take_profit_pct)
                    exit_reason = "take_profit"
                elif signal == "SELL":
                    exit_price = close
                    exit_reason = "signal"

                if exit_price is not None:
                    sell_price = exit_price * (1 - self.slippage_pct)
                    proceeds = position_qty * sell_price
                    fee = proceeds * self.fee_pct
                    cash += proceeds - fee
                    fees_paid += fee

                    # PnL = cash received on exit minus cash spent to enter
                    # (entry notional + entry fee), both already recorded
                    # on the trade before this exit fee was added.
                    entry_cost = current_trade.entry_price * current_trade.quantity + current_trade.fee_paid
                    current_trade.exit_time = ts
                    current_trade.exit_price = sell_price
                    current_trade.fee_paid += fee
                    current_trade.pnl = (proceeds - fee) - entry_cost
                    current_trade.exit_reason = exit_reason
                    trades.append(current_trade)
                    current_trade = None
                    position_qty = 0.0
                    entry_price = 0.0

            elif signal == "BUY" and cash > 0:
                buy_price = close * (1 + self.slippage_pct)
                allocation = cash * self.position_size_pct
                fee = allocation * self.fee_pct
                spendable = allocation - fee
                qty = spendable / buy_price
                if qty > 0:
                    cash -= allocation
                    position_qty = qty
                    entry_price = buy_price
                    fees_paid += fee
                    current_trade = Trade(
                        entry_time=ts, entry_price=buy_price, quantity=qty, fee_paid=fee,
                    )

            equity = cash + position_qty * close
            equity_values.append((ts, equity))

        # Force-close any position still open at the end of the data
        # window so metrics reflect a complete, comparable period.
        if position_qty > 0 and current_trade is not None:
            last_close = float(df["close"].iloc[-1])
            sell_price = last_close * (1 - self.slippage_pct)
            proceeds = position_qty * sell_price
            fee = proceeds * self.fee_pct
            cash += proceeds - fee
            fees_paid += fee
            entry_cost = current_trade.entry_price * current_trade.quantity + current_trade.fee_paid
            current_trade.exit_time = df["timestamp"].iloc[-1]
            current_trade.exit_price = sell_price
            current_trade.fee_paid += fee
            current_trade.pnl = (proceeds - fee) - entry_cost
            current_trade.exit_reason = "end_of_data"
            trades.append(current_trade)
            equity_values[-1] = (equity_values[-1][0], cash)

        equity_curve = pd.Series(
            [v for _, v in equity_values], index=[t for t, _ in equity_values], name="equity"
        )

        buy_and_hold_qty = self.initial_balance * (1 - self.fee_pct) / (df["close"].iloc[0] * (1 + self.slippage_pct))
        buy_and_hold_final = buy_and_hold_qty * df["close"].iloc[-1] * (1 - self.fee_pct) * (1 - self.slippage_pct)

        pnls = [t.pnl for t in trades if t.pnl is not None]
        avg_win, avg_loss = m.average_win_loss(pnls)
        metrics = {
            "total_return_pct": m.total_return_pct(equity_curve),
            "max_drawdown_pct": m.max_drawdown_pct(equity_curve),
            "sharpe_ratio": m.sharpe_ratio(equity_curve, self.timeframe),
            "sortino_ratio": m.sortino_ratio(equity_curve, self.timeframe),
            "win_rate_pct": m.win_rate_pct(pnls),
            "profit_factor": m.profit_factor(pnls),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "n_trades": len(trades),
        }
        metrics["suspicious_flags"] = m.suspicious_result_flags(metrics, len(trades))

        return BacktestReport(
            symbol=symbol,
            timeframe=self.timeframe,
            initial_balance=self.initial_balance,
            final_balance=cash,
            trades=trades,
            equity_curve=equity_curve,
            buy_and_hold_final_balance=buy_and_hold_final,
            fees_paid=fees_paid,
            metrics=metrics,
        )
