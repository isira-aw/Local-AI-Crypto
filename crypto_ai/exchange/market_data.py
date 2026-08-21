"""
Public market-data client (no API key required).

What is it?
    A thin, retrying wrapper around ccxt's Binance client for reading
    PUBLIC market data only: OHLCV candles and the current ticker.

Why is it needed?
    - Keeps all "talk to Binance" code in one place.
    - Adds retry-with-backoff around transient network errors, since
      the whole system must survive brief internet interruptions
      (Section 25/27) instead of crashing.
    - Never touches private endpoints, so it never needs an API key —
      historical download, backtesting and paper trading all work with
      zero account setup (Section 4).

What goes in / comes out?
    In: symbol ("BTC/USDT"), timeframe ("5m"), time range.
    Out: pandas-friendly list of OHLCV rows:
        [{"timestamp": datetime, "open": .., "high": .., "low": ..,
          "close": .., "volume": ..}, ...]

How do I troubleshoot it?
    See docs/TROUBLESHOOTING.md — "cannot reach Binance". Most often
    this is a firewall/VPN/geo-restriction issue, not a bug.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

MAX_CANDLES_PER_REQUEST = 1000  # Binance's per-request cap


class OHLCVSource(Protocol):
    """Anything that can fetch raw OHLCV rows — implemented by ccxt, or a
    fake in tests."""

    def fetch_ohlcv(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> list[list]:
        ...


class CcxtBinanceSource:
    """Real implementation backed by ccxt. Public endpoints only."""

    def __init__(self) -> None:
        import ccxt  # imported lazily so tests don't need network libs configured

        self._exchange = ccxt.binance({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> list[list]:
        return self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        return self._exchange.fetch_ticker(symbol)


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0


def with_retry(fn, retry_policy: RetryPolicy, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying transient errors with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - network/exchange errors are all retryable here
            last_exc = exc
            delay = min(retry_policy.base_delay_seconds * (2 ** (attempt - 1)), retry_policy.max_delay_seconds)
            logger.warning(
                "Market data call failed (attempt %s/%s): %s. Retrying in %.1fs",
                attempt, retry_policy.max_attempts, exc, delay,
            )
            if attempt < retry_policy.max_attempts:
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _to_rows(raw: list[list]) -> list[dict]:
    return [
        {
            "timestamp": dt.datetime.fromtimestamp(r[0] / 1000, tz=dt.timezone.utc),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        }
        for r in raw
    ]


class MarketDataClient:
    """Public-data client used by the downloader, feature pipeline and dashboard."""

    def __init__(self, source: OHLCVSource | None = None, retry_policy: RetryPolicy | None = None):
        self.source = source or CcxtBinanceSource()
        self.retry_policy = retry_policy or RetryPolicy()

    def fetch_ohlcv_page(self, symbol: str, timeframe: str, since: dt.datetime, limit: int = MAX_CANDLES_PER_REQUEST) -> list[dict]:
        since_ms = int(since.timestamp() * 1000)
        raw = with_retry(self.source.fetch_ohlcv, self.retry_policy, symbol, timeframe, since_ms, limit)
        return _to_rows(raw)

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start: dt.datetime,
        end: dt.datetime,
        on_page=None,
    ) -> list[dict]:
        """
        Fetch all candles in [start, end), paginating in chunks of
        MAX_CANDLES_PER_REQUEST. `on_page`, if given, is called with
        each page of rows as it arrives — used by the downloader to
        write to the database incrementally so an interrupted download
        can resume instead of losing all progress.
        """
        all_rows: list[dict] = []
        cursor = start
        timeframe_delta = _timeframe_to_timedelta(timeframe)

        while cursor < end:
            page = self.fetch_ohlcv_page(symbol, timeframe, cursor, MAX_CANDLES_PER_REQUEST)
            if not page:
                break
            page = [r for r in page if r["timestamp"] < end]
            if on_page:
                on_page(page)
            all_rows.extend(page)

            last_ts = page[-1]["timestamp"] if page else None
            if last_ts is None:
                break
            next_cursor = last_ts + timeframe_delta
            if next_cursor <= cursor:
                # Exchange returned no forward progress — stop to avoid an infinite loop.
                break
            cursor = next_cursor

        return all_rows


_TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
}


def _timeframe_to_timedelta(timeframe: str) -> dt.timedelta:
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return dt.timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])
