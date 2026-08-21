# Real Trading (Testnet and Live)

**Read `docs/REGULATORY_NOTES.md` before anything in this document.**
Testnet and live trading both require an exchange account and API keys,
which puts you in scope of your local regulations in a way research/
backtesting/paper trading (no account needed) never does.

## The current state of this repository

`crypto_ai/exchange/live.py` is a **deliberately gated stub**, not a working
order-execution client. It refuses every call unless a full safety gate
passes, and raises `NotImplementedError` even then. Implementing real order
placement (creating orders, canceling them, fetching open orders/positions
via Binance's private API) is intentionally left as the next step for
*your* specific situation, after everything below is actually true for you
— not something this codebase should decide on your behalf.

## Why real trading is disabled by default

```
REAL_TRADING_ENABLED=false     (.env)
TRADING_ENABLED=false          (.env)
MODE=RESEARCH                  (.env)
```

All three start at their safest value. Nothing in this system will change
them for you. Even if every number looks great, the system will never
"decide" to start trading real money — see Section 31 of the design
document: real trading requires **explicit, repeated** user action, not
good performance.

## The live-trading gate (Section 59)

`crypto_ai/risk/safety_rules.py:evaluate_live_trading_gate()` checks a full
checklist and blocks unless **every single item** passes:

```
[ ] Historical data validated
[ ] Backtest completed
[ ] Walk-forward validation passed (not just the final test set)
[ ] Enough paper trading time (default: 30+ days, risk.yaml)
[ ] Enough paper trades (default: 100+, risk.yaml)
[ ] Positive net paper-trading return
[ ] Paper-trading drawdown within limits
[ ] No critical system errors recently (default: 14+ days clean)
[ ] Regulatory/access check acknowledged (see docs/REGULATORY_NOTES.md)
[ ] Exchange connection stable
[ ] Emergency stop tested
[ ] Recovery (restart) tested
[ ] Backup/restore tested at least once
[ ] API key has withdrawals disabled
[ ] MODE is TESTNET or LIVE (not RESEARCH/PAPER)
[ ] You explicitly enabled live trading
[ ] REAL_TRADING_ENABLED=true
```

All thresholds are configurable in `risk.yaml` under `live_trading_gate:` —
but loosening them doesn't make a strategy more proven, it just lowers the
bar you're checking it against. Change them thoughtfully, not to make a
red checklist turn green faster.

## API key safety (Section 60)

When you eventually create exchange API keys for testnet or live use:

- **Never enable withdrawal permission.** The bot should only ever be able
  to trade, never move funds out of the account.
- Grant only the permissions actually needed (spot trading).
- Use IP restrictions if your exchange supports them.
- Use **separate** key pairs for testnet and live — never reuse one.
- Put keys only in `.env` (already `.gitignore`d) — never in code, never in
  a commit, never sent to the LLM, never logged (the event logger in
  `database/repositories/events_repo.py` redacts anything that looks like a
  secret, but don't rely on that as your only safeguard).
- Never expose keys to the frontend/dashboard — the API layer
  (`app/api/routes.py`) never returns them, by design.

## Position sizing, once live

Even after the gate passes, start capped:

```
MAX_LIVE_CAPITAL = $50     (risk.yaml: live_trading_gate.max_live_capital_usdt)
```

This is configurable but intentionally small. Increase it only after real,
sustained live-trading evidence — the same "prove it, don't assume it"
principle that governs every earlier stage.

## Emergency stop (Section 32)

A hard stop is available from the dashboard's Emergency Stop button (writes
a flag file at `data_store/EMERGENCY_STOP`, checked by the risk engine on
every single decision — see `risk/emergency.py`). It:

- Blocks all new `BUY` signals in every process (dashboard, scheduler, CLI)
  immediately, no restart needed.
- Requires a **manual** reset (dashboard button, or delete the flag file) —
  it will not silently clear itself.

There is also a **hardware-level fallback** for the live-trading phase,
separate from this software switch: if you're ever unsure whether the
software stop actually worked, physically stop the machine running the bot
and check your open positions directly from the exchange's own app/website
on your phone. That is always available to you and doesn't depend on this
codebase working correctly.

## Testnet vs. Live

**Testnet**: Binance's practice-money sandbox — a real API, fake funds. Good
for testing that order-placement code actually works before touching real
money. Still requires an account and its own API key, and still requires
you to have completed `docs/REGULATORY_NOTES.md` (the exchange account
itself is real, even if the funds aren't).

**Live**: real money, real orders, capped at `MAX_LIVE_CAPITAL`. Everything
above applies, plus real financial risk. Nothing here promises this will be
profitable — see `docs/CRYPTO_BASICS.md`.
