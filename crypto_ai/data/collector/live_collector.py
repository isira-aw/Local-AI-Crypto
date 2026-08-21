"""
Live market-data collector (Section 10: "Also implement live WebSocket
market-data collection where appropriate").

What is it?
    A resilient WebSocket client for Binance's public kline (candle)
    stream, which pushes each candle as it updates and marks it closed.
    Only CLOSED candles are written to `market_data` — an in-progress
    candle's high/low/close keep changing, and storing one would poison
    every feature computed from it.

Why is it needed?
    Polling the REST endpoint every N minutes (what the scheduler's
    `collect_data` job does) is simple and sufficient for a 5m strategy,
    but it always lags the market by up to one poll interval. The
    WebSocket stream delivers a candle the moment it closes, which
    matters if you ever move to a shorter timeframe.

    The REST poller remains the source of truth for backfilling history
    and for filling gaps after a disconnection — this collector is a
    live supplement, never a replacement. That is deliberate: a
    WebSocket that silently dies would otherwise create a silent hole in
    the data.

What goes in / comes out?
    In: a symbol and timeframe. Out: rows appended to `market_data`
    (same table and dedup rules as the REST downloader).

How do I start it?
    python run.py collect-live
    (or call `run_collector()` from your own code)

How do I stop it?
    Ctrl+C, or set the returned collector's `.stop()`.

How do I troubleshoot it?
    Connection failures are logged and retried with exponential backoff
    forever — a dropped connection is expected behaviour on a 24/7
    system, not an error condition (Section 25/27). On reconnect the
    collector triggers a REST backfill so any candles missed while
    disconnected are recovered rather than silently lost.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging

from crypto_ai.database.base import session_scope
from crypto_ai.database.repositories.events_repo import log_event
from crypto_ai.database.repositories.market_data_repo import upsert_candles

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"

# Binance stream names use the same timeframe tokens this project does.
SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")


def stream_name(symbol: str, timeframe: str) -> str:
    """'BTC/USDT' + '5m' -> 'btcusdt@kline_5m'"""
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {SUPPORTED_TIMEFRAMES}")
    return f"{symbol.replace('/', '').lower()}@kline_{timeframe}"


def parse_kline_message(raw: str | dict) -> dict | None:
    """
    Parse one Binance kline payload.

    Returns a candle dict ready for `upsert_candles`, or None if the
    message is not a closed candle (in-progress updates are ignored by
    design — see module docstring).
    """
    msg = json.loads(raw) if isinstance(raw, str) else raw
    kline = msg.get("k")
    if not kline:
        return None
    if not kline.get("x"):  # "x" == is this candle closed?
        return None
    return {
        "timestamp": dt.datetime.fromtimestamp(kline["t"] / 1000, tz=dt.timezone.utc),
        "open": float(kline["o"]),
        "high": float(kline["h"]),
        "low": float(kline["l"]),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
    }


class LiveCollector:
    """
    Reconnecting WebSocket collector.

    `connect_factory` is injected so tests can drive the collector with a
    fake stream instead of opening a real socket.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        connect_factory=None,
        backfill_on_reconnect: bool = True,
        max_backoff_seconds: float = 60.0,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.connect_factory = connect_factory
        self.backfill_on_reconnect = backfill_on_reconnect
        self.max_backoff_seconds = max_backoff_seconds
        self._stop = asyncio.Event()
        self.candles_written = 0
        self.reconnections = 0

    def stop(self) -> None:
        self._stop.set()

    def _default_connect_factory(self):
        import websockets

        return websockets.connect(f"{BINANCE_WS_BASE}/{stream_name(self.symbol, self.timeframe)}")

    def handle_message(self, raw: str | dict) -> bool:
        """Parse and persist one message. Returns True if a candle was written."""
        candle = parse_kline_message(raw)
        if candle is None:
            return False
        with session_scope() as session:
            upsert_candles(session, self.symbol, self.timeframe, [candle])
        self.candles_written += 1
        return True

    def _backfill_gap(self) -> None:
        """
        After a reconnect, pull anything missed over REST rather than
        assuming the stream resumed cleanly.
        """
        try:
            from crypto_ai.data.downloader.historical import backfill_symbol_timeframe
            from crypto_ai.exchange.market_data import MarketDataClient

            with session_scope() as session:
                backfill_symbol_timeframe(
                    session, MarketDataClient(), self.symbol, self.timeframe, initial_history_days=1,
                )
        except Exception as exc:  # noqa: BLE001 - backfill failure must not kill the collector
            logger.warning("Post-reconnect backfill failed (will retry on next reconnect): %s", exc)

    async def run(self) -> None:
        backoff = 1.0
        first_connection = True

        while not self._stop.is_set():
            try:
                factory = self.connect_factory or self._default_connect_factory
                async with factory() as ws:
                    logger.info("Live collector connected: %s %s", self.symbol, self.timeframe)
                    if not first_connection:
                        self.reconnections += 1
                        if self.backfill_on_reconnect:
                            self._backfill_gap()
                    first_connection = False
                    backoff = 1.0  # reset only after a successful connection

                    with session_scope() as session:
                        log_event(
                            session, component="live_collector", event="connected", severity="INFO",
                            message=f"Streaming {self.symbol} {self.timeframe}",
                        )

                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            self.handle_message(raw)
                        except Exception as exc:  # noqa: BLE001 - one bad message must not drop the stream
                            logger.warning("Failed to handle stream message: %s", exc)

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on ANY transport error
                if self._stop.is_set():
                    break
                logger.warning(
                    "Live collector disconnected (%s). Reconnecting in %.1fs", exc, backoff,
                )
                with contextlib.suppress(Exception):
                    with session_scope() as session:
                        log_event(
                            session, component="live_collector", event="disconnected", severity="WARNING",
                            message=str(exc), context={"retry_in_seconds": backoff},
                        )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, self.max_backoff_seconds)

        logger.info("Live collector stopped (%s candles written, %s reconnections).",
                    self.candles_written, self.reconnections)


def run_collector(symbol: str | None = None, timeframe: str | None = None) -> None:
    """Blocking entry point used by `python run.py collect-live`."""
    from crypto_ai.config.loader import get_settings

    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    collector = LiveCollector(symbol, timeframe)
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        collector.stop()
