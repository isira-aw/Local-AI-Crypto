# Paper Trading

## What is it?

"Trading" with fake money against **live** market prices, using the exact
same fee/slippage cost model as the backtester —
`crypto_ai/paper_trading/engine.py`.

## Why is it needed?

Section 17 calls this "the most important first live stage." It's the first
point where the strategy faces conditions a backtest can't fully capture:
real-time data delivery, whatever the market is actually doing *right now*,
and everything the risk engine would do in real trading — without risking
a cent.

## What goes in / comes out?

**In:** the current champion model's prediction for the latest bar, passed
through the risk engine. **Out:** rows in `paper_trades` (every simulated
entry/exit, with fees, slippage, and P/L) and `portfolio` (an equity
snapshot every bar).

**Start it:** paper trading runs automatically once you `python run.py
start` (the scheduler ticks it on `scheduler.prediction_interval_minutes`).
For a single manual tick: `python run.py paper`.
**Stop it:** stop the scheduler/app (`python run.py stop`, or `docker
compose down` if using Docker). Paper trading state lives entirely in the
database, so stopping and restarting the app does not lose your position or
history.

## How does it work?

1. Load enough recent candles to compute features.
2. Ask the current **champion** model for a prediction (refuses to trade if
   no champion exists yet, or if the champion's feature version doesn't
   match the current feature pipeline — see `docs/MODEL_TRAINING.md`).
3. Compute the account's current risk state (today's P/L, drawdown,
   consecutive losses) from the paper portfolio's own history.
4. Pass the model's proposed signal through `RiskManager.evaluate()` — this
   can override the model to `WAIT` (max position count, daily loss limit,
   drawdown limit, cooldown after consecutive losses, low confidence,
   emergency stop). See `docs/REAL_TRADING.md` for the full risk engine.
5. Act on whatever the risk engine allowed, via the paper-trading engine —
   same fee/slippage assumptions as the backtester.
6. Record a `Prediction` row so accuracy can be measured later (see below).

**State recovery:** every step reads current position/balance fresh from
the database, never from in-memory state. A crash or restart mid-trade
resumes correctly — this is tested explicitly
(`tests/test_paper_trading.py::test_engine_state_recovers_after_simulated_restart`).

## Prediction evaluation (Section 21)

Separately from whether a trade was opened/closed, every prediction is
checked against what *actually* happened once its time horizon elapses:

```
Prediction at 10:00 -> wait 1 hour -> look up actual price -> was it right?
```

Run manually with `python run.py evaluate`, or let the scheduler do it
automatically. This produces prediction accuracy / directional accuracy
numbers shown on the dashboard's AI page — kept separate from the
paper-trading P/L numbers, because a model can have OK P/L while its raw
directional accuracy is mediocre, or vice versa (Section 46: don't conflate
these).

## What "success" looks like here

**Not** "the balance went up once." Look for, over weeks:

- Enough trades to trust the numbers (dozens, not a handful)
- Positive risk-adjusted return (Sharpe/Sortino), not just positive return
- Reasonable, not extreme, drawdown
- No suspicious-flags warnings on the summary
- Performance that's roughly consistent over time, not concentrated in one
  lucky week

The dashboard and daily report (`python run.py report`) surface all of this.
The report's recommendation field is deliberately conservative — see
`app/services/reporting.py` — and will say `CONTINUE RESEARCH` or
`NOT READY FOR REAL TRADING` far more often than `CANDIDATE FOR TESTNET`.

## Next step

Only after real, sustained evidence: `docs/REAL_TRADING.md`.
