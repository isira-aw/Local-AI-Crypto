# Data Pipeline

## What is it?

The chain of code that turns raw exchange prices into something a model can
learn from: **downloader → validator → feature pipeline → labels**.

## Why is it needed?

Section 10 of the design document is explicit: the user should never have to
manually find, clean, or label a dataset. This pipeline automates all of it.

## 1. Historical downloader (`crypto_ai/data/downloader/historical.py`)

**In:** nothing but your configured symbol/timeframes (`settings.yaml`).
**Out:** rows in the `market_data` table.

Uses Binance's **public** OHLCV endpoint (Open/High/Low/Close/Volume candles)
— no API key needed. On each run it:

- Checks the database for the most recent stored candle per timeframe and
  resumes from there (so re-running is fast and never re-downloads
  everything).
- Pages through the exchange's API and writes each page to the database
  immediately, so an interrupted download loses at most one page of
  progress, not the whole run.
- Retries transient network errors with exponential backoff.

**Start it:** `python run.py download-data`
**Troubleshoot:** if it can't reach Binance, check your internet connection
and whether your country/network can reach `api.binance.com` (see
`docs/REGULATORY_NOTES.md` — this only affects live/testnet trading
legality, not whether the public data endpoint is reachable, which is a
purely technical/network question).

## 2. Data validation (`crypto_ai/data/validators/ohlcv_validators.py`)

**In:** a DataFrame of candles. **Out:** a `ValidationReport`.

Checks for:
- **Duplicate timestamps** (same candle stored twice)
- **Gaps** (missing candles — the exchange had an outage, or a request hit a
  rate limit)
- **Bad OHLC values** (e.g. `high` lower than `open`/`close`, negative
  prices/volume)
- **Non-monotonic timestamps** (data out of order)

The downloader runs this automatically after every backfill and logs any
issues found as a `WARNING` system event (visible on the dashboard's System
page) — it does not silently trust whatever the exchange returned.

## 3. Feature pipeline (`crypto_ai/features/feature_pipeline.py`)

**In:** clean OHLCV candles. **Out:** a feature matrix — one row per candle,
one column per indicator.

Computes standard technical indicators (`crypto_ai/features/technical.py` and
`volatility.py`): SMA, EMA, RSI, MACD, ATR, Bollinger Bands, rolling
volatility, momentum, and volume-relative-to-average. Every indicator is
**strictly causal** — it only ever looks at the current bar and earlier ones,
never the future (this is tested explicitly; see `tests/test_features.py`'s
`test_no_lookahead_bias_in_indicators`).

Every feature row is tagged with a `feature_version` string. If you change
how features are computed, bump `FEATURE_VERSION` in
`feature_pipeline.py` — this keeps old and new feature sets from silently
mixing together in the database, and the live-prediction pipeline refuses to
use a model trained on a feature version that no longer matches (see
`app/services/live_pipeline.py`).

## 4. Label generation (`crypto_ai/features/labels.py`)

**In:** OHLCV candles + config (horizon, thresholds, assumed fees/slippage).
**Out:** a `BUY` / `HOLD` / `SELL` label per candle.

```
future_return = close[t + horizon] / close[t] - 1
net_return    = future_return - (2 * (fee_pct + slippage_pct))   # round-trip cost
label         = BUY  if net_return > positive_threshold
                SELL if net_return < negative_threshold
                HOLD otherwise
```

This is fully automatic — you never label a single row by hand. Costs are
baked into the label itself, so a `BUY` label means "a move large enough to
plausibly be worth trading after costs," not just statistical noise.

**Configurable in `settings.yaml` under `labeling:`.** Nothing here is
hard-coded — see the comments in that file for the defaults and why they
were chosen.

## 5. Live collector (`crypto_ai/data/collector/live_collector.py`)

**In:** a symbol/timeframe. **Out:** rows appended to `market_data` the
moment each candle closes.

The REST downloader above polls on a schedule and is the source of truth
for history and gap-filling. The live collector is a WebSocket supplement
that delivers a candle as soon as it closes, rather than up to one poll
interval later.

Two deliberate behaviours:

- **Only closed candles are stored.** An in-progress candle's
  high/low/close keep changing; storing one would poison every feature
  computed from it.
- **On reconnect it triggers a REST backfill.** A dropped WebSocket is
  normal on a 24/7 system, and assuming the stream resumed cleanly would
  leave a silent hole in the data.

**Start it:** `python run.py collect-live`

## Retention

Raw 1-minute data can get large fast. `settings.yaml`'s `data.retention`
section defines how long each timeframe/table is kept (see Section 54 of the
design doc) — this is enforced by a scheduled cleanup job, not automatic
database growth limits, so check that job's logs if disk usage looks off.

## Where this data goes next

`docs/MODEL_TRAINING.md` — features + labels feed directly into the
walk-forward training pipeline.
