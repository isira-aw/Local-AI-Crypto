"""
Automatic historical data downloader (Phase 4 / Section 10).

What is it?
    Backfills OHLCV candles for one symbol across all configured
    timeframes, storing them in the `market_data` table.

Why is it needed?
    The user should never have to manually find or download a crypto
    dataset. This runs once (or incrementally) and does it for them.

How does it work?
    For each timeframe:
      1. Check the database for the latest stored candle (resume point).
      2. If none exists, start from `initial_history_days` ago.
      3. Page through the exchange's public OHLCV endpoint up to "now",
         writing each page to the database as it arrives (so a crash
         or Ctrl-C partway through loses at most one page of progress,
         not the whole run).
      4. Run the Phase 5 validators on the result and log any gaps —
         gaps usually mean the exchange had a data issue or the
         request hit a rate limit; they are logged, not silently
         ignored.

How do I start it?
    python run.py download-data
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.data.validators.ohlcv_validators import validate_ohlcv
from crypto_ai.database.repositories.events_repo import log_event
from crypto_ai.database.repositories.market_data_repo import (
    get_latest_timestamp,
    load_candles_df,
    upsert_candles,
)
from crypto_ai.exchange.market_data import MarketDataClient

logger = logging.getLogger(__name__)


def backfill_symbol_timeframe(
    session: Session,
    client: MarketDataClient,
    symbol: str,
    timeframe: str,
    initial_history_days: int,
    now: dt.datetime | None = None,
) -> dict:
    """Backfill a single (symbol, timeframe) pair. Returns a small summary dict."""
    now = now or dt.datetime.now(dt.timezone.utc)
    latest = get_latest_timestamp(session, symbol, timeframe)

    if latest is not None:
        # Resume from just after the last stored bar (Section 10:
        # "resume interrupted downloads").
        start = latest.replace(tzinfo=latest.tzinfo or dt.timezone.utc) + _one_bar(timeframe)
    else:
        start = now - dt.timedelta(days=initial_history_days)

    if start >= now:
        logger.info("%s %s already up to date (latest=%s)", symbol, timeframe, latest)
        return {"symbol": symbol, "timeframe": timeframe, "new_rows": 0, "resumed_from": latest}

    written = 0

    def _on_page(page: list[dict]) -> None:
        nonlocal written
        if not page:
            return
        upsert_candles(session, symbol, timeframe, page)
        session.commit()
        written += len(page)

    client.fetch_ohlcv_range(symbol, timeframe, start, now, on_page=_on_page)

    log_event(
        session,
        component="downloader",
        event="backfill_complete",
        severity="INFO",
        message=f"Backfilled {written} bars for {symbol} {timeframe}",
        context={"symbol": symbol, "timeframe": timeframe, "new_rows": written},
    )
    session.commit()

    # Validate what we now have and log any gaps rather than silently
    # trusting the exchange response.
    df = load_candles_df(session, symbol, timeframe)
    report = validate_ohlcv(df, timeframe)
    if not report.is_clean:
        log_event(
            session,
            component="downloader",
            event="validation_issue",
            severity="WARNING",
            message=report.summary(),
            context={"symbol": symbol, "timeframe": timeframe},
        )
        session.commit()

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "new_rows": written,
        "resumed_from": latest,
        "validation": report.summary(),
    }


def backfill_all(session: Session, client: MarketDataClient | None = None) -> list[dict]:
    settings = get_settings()
    client = client or MarketDataClient()
    symbol = settings.get("exchange.symbol", "BTC/USDT")
    timeframes = settings.get("data.timeframes", ["5m"])
    initial_history_days = settings.get("data.initial_history_days", 730)

    results = []
    for timeframe in timeframes:
        try:
            result = backfill_symbol_timeframe(session, client, symbol, timeframe, initial_history_days)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Backfill failed for %s %s", symbol, timeframe)
            log_event(
                session,
                component="downloader",
                event="backfill_failed",
                severity="ERROR",
                message=str(exc),
                context={"symbol": symbol, "timeframe": timeframe},
            )
            session.commit()
            results.append({"symbol": symbol, "timeframe": timeframe, "error": str(exc)})
    return results


_TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _one_bar(timeframe: str) -> dt.timedelta:
    return dt.timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])
