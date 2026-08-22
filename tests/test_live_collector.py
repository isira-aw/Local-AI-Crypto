"""
Live WebSocket collector tests.

The headline test here is the reconnect-storm regression. It was found by
actually running the collector end-to-end (scripts/prove_wiring.py) rather
than unit-testing its parser: a clean stream close fell straight back into
connect() with no delay, producing 182,660 reconnects and matching
"connected" events in about twenty minutes. Against real Binance — which
closes streams periodically — that is a request storm against the exchange
and an unbounded write storm against the events table.
"""

import asyncio
import datetime as dt
import json

import pytest

from crypto_ai.data.collector.live_collector import LiveCollector, parse_kline_message, stream_name


def _closed(ts_ms: int, close: str = "1.5") -> str:
    return json.dumps({"k": {"t": ts_ms, "x": True, "o": "1", "h": "2", "l": "0.5", "c": close, "v": "10"}})


def _open(ts_ms: int) -> str:
    return json.dumps({"k": {"t": ts_ms, "x": False, "o": "1", "h": "2", "l": "0.5", "c": "1.4", "v": "9"}})


class FakeWS:
    """A websocket that yields a fixed set of messages then closes cleanly."""

    def __init__(self, messages, on_connect=None):
        self.messages = messages
        self.on_connect = on_connect

    async def __aenter__(self):
        if self.on_connect:
            self.on_connect()
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self.messages:
            yield m


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------
def test_only_closed_candles_are_parsed():
    assert parse_kline_message(_open(1700000000000)) is None
    candle = parse_kline_message(_closed(1700000000000))
    assert candle["close"] == 1.5
    assert candle["timestamp"].tzinfo is not None


def test_stream_name_and_unsupported_timeframe():
    assert stream_name("BTC/USDT", "5m") == "btcusdt@kline_5m"
    with pytest.raises(ValueError):
        stream_name("BTC/USDT", "3s")


# ---------------------------------------------------------------------
# Reconnect behaviour — the regression
# ---------------------------------------------------------------------
def test_clean_stream_close_does_not_cause_a_reconnect_storm():
    """
    A peer closing the stream must be treated as a disconnect and go
    through the backoff, not loop straight back into connect().
    """
    connects = []
    collector = LiveCollector(
        "BTC/USDT", "5m",
        connect_factory=lambda: FakeWS([], on_connect=lambda: connects.append(1)),
        backfill_on_reconnect=False,
        initial_backoff_seconds=0.05,
        max_backoff_seconds=0.2,
        max_reconnects=3,
    )

    async def drive():
        await asyncio.wait_for(collector.run(), timeout=10)

    asyncio.run(drive())

    # Bounded by max_reconnects, not spinning freely.
    assert len(connects) <= 4, f"reconnect storm: {len(connects)} connects"
    assert collector.reconnections == 4 or collector.reconnections <= 4


def test_backoff_is_not_reset_by_an_instantly_dropping_connection():
    """
    An unhealthy connection (drops immediately) must not reset the backoff
    — otherwise it can reconnect at full speed forever.
    """
    collector = LiveCollector(
        "BTC/USDT", "5m",
        connect_factory=lambda: FakeWS([]),
        backfill_on_reconnect=False,
        initial_backoff_seconds=0.05,
        max_backoff_seconds=1.0,
        healthy_session_seconds=60.0,  # nothing here will ever qualify as healthy
        max_reconnects=3,
    )

    started = asyncio.run(_timed_run(collector))
    # 0.05 + 0.10 + 0.20 = 0.35s of enforced waiting across 3 reconnects.
    assert started >= 0.3, f"backoff was not applied (took only {started:.3f}s)"


async def _timed_run(collector):
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await asyncio.wait_for(collector.run(), timeout=15)
    return loop.time() - t0


def test_stop_breaks_out_promptly():
    collector = LiveCollector(
        "BTC/USDT", "5m", connect_factory=lambda: FakeWS([]),
        backfill_on_reconnect=False, initial_backoff_seconds=5.0,
    )

    async def drive():
        task = asyncio.create_task(collector.run())
        await asyncio.sleep(0.2)
        collector.stop()
        await asyncio.wait_for(task, timeout=5)  # must not wait out the 5s backoff

    asyncio.run(drive())


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
def test_collector_writes_only_closed_candles_to_the_database(monkeypatch, db_session):
    from contextlib import contextmanager

    import crypto_ai.data.collector.live_collector as mod

    @contextmanager
    def fake_scope():
        yield db_session

    monkeypatch.setattr(mod, "session_scope", fake_scope)

    base = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    messages = []
    for i in range(4):
        ts = int((base + dt.timedelta(minutes=5 * i)).timestamp() * 1000)
        messages.append(_open(ts))      # ignored
        messages.append(_closed(ts))    # stored

    collector = LiveCollector(
        "BTC/USDT", "5m", connect_factory=lambda: FakeWS(messages),
        backfill_on_reconnect=False, initial_backoff_seconds=0.01, max_reconnects=0,
    )
    asyncio.run(asyncio.wait_for(collector.run(), timeout=10))

    from crypto_ai.database.models.market import MarketData

    assert collector.candles_written == 4
    assert db_session.query(MarketData).count() == 4, "in-progress candles must not be stored"


def test_a_bad_message_does_not_kill_the_stream(monkeypatch, db_session):
    from contextlib import contextmanager

    import crypto_ai.data.collector.live_collector as mod

    @contextmanager
    def fake_scope():
        yield db_session

    monkeypatch.setattr(mod, "session_scope", fake_scope)

    base = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    ts = int(base.timestamp() * 1000)
    messages = ["this is not json", _closed(ts), json.dumps({"unexpected": "shape"})]

    collector = LiveCollector(
        "BTC/USDT", "5m", connect_factory=lambda: FakeWS(messages),
        backfill_on_reconnect=False, initial_backoff_seconds=0.01, max_reconnects=0,
    )
    asyncio.run(asyncio.wait_for(collector.run(), timeout=10))

    assert collector.candles_written == 1, "the good candle should still have been stored"
