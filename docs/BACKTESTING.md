# Backtesting

## What is it?

Simulating a strategy's signals against **historical** price data,
bar-by-bar, with realistic trading costs — `crypto_ai/backtesting/engine.py`.

## Why is it needed?

A model can have great classification accuracy and still lose money once
fees and slippage are accounted for. Backtesting is the first, fastest check
of whether a strategy is even worth paper trading.

## What goes in / comes out?

**In:** a DataFrame of `[timestamp, close, signal]` (and optionally
`[high, low]` for intrabar stop-loss/take-profit), plus cost assumptions
(fee %, slippage %, position size %, optional stop-loss/take-profit %).
**Out:** a `BacktestReport` — final balance, every trade, an equity curve,
and a metrics dict.

**Start it:** `python run.py backtest` (runs the current champion model
against recent history).

## How does it work?

Spot-only, single position at a time, no leverage or shorting (Section 5).
On each bar:

1. If holding a position, check stop-loss/take-profit against that bar's
   high/low first.
2. Otherwise, act on the bar's signal: enter on `BUY` if flat, exit on
   `SELL` if in a position.
3. Mark the portfolio to market at the bar's close, every bar, whether or
   not a trade happened — this is what produces the equity curve.
4. Any position still open at the end of the data window is force-closed,
   so results are comparable across different time windows.

**No future data is ever used.** At bar *i*, the engine only ever sees the
signal and price for bar *i* — this is tested explicitly
(`tests/test_backtesting.py::test_no_lookahead_backtest_uses_only_current_and_past_signal`).

## Reading the report

```
Initial balance / Final balance / Net profit / Return %
Buy-and-hold return %          <- the benchmark, always shown
Number of trades / Win rate / Average win / Average loss
Profit factor / Maximum drawdown / Sharpe ratio / Sortino ratio
Fees paid
```

**Always compare Return % to the buy-and-hold return.** A strategy that
returns 8% while BTC itself went up 15% did **worse** than doing nothing,
even though 8% sounds fine on its own.

A PNG chart of the equity curve (strategy vs. an approximate buy-and-hold
line) is saved automatically to `data_store/reports/`.

## Warnings you should never ignore

If the report includes a `WARNINGS` section (unrealistically high Sharpe,
very high win rate, extreme return, big train/test gap, too few trades),
that is the too-good-to-be-true detector (Section 47) telling you the
result likely isn't real — usually a sign of overfitting or a leftover data
leak, not a genuinely great strategy. See `docs/MODEL_TRAINING.md`.

## Relationship to paper trading

Backtesting is fast but optimistic — it can't fully capture real order-book
depth, partial fills, or execution delay. Passing backtesting is necessary
but not sufficient; see `docs/PAPER_TRADING.md` for the next, slower, more
realistic stage.
