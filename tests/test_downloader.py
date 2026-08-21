import datetime as dt

import pandas as pd
import pytest

from crypto_ai.data.downloader.historical import backfill_symbol_timeframe
from crypto_ai.data.validators.ohlcv_validators import (
    detect_bad_ohlc,
    detect_duplicates,
    detect_gaps,
    validate_ohlcv,
)
from crypto_ai.database.repositories.market_data_repo import load_candles_df
from crypto_ai.exchange.market_data import MarketDataClient, MAX_CANDLES_PER_REQUEST


class FakeSource:
    """Simulates a Binance-like OHLCV endpoint over an in-memory series."""

    def __init__(self, all_rows: list[dict], page_size: int = 500, flaky_once: bool = False):
        self.all_rows = sorted(all_rows, key=lambda r: r["timestamp"])
        self.page_size = page_size
        self._flaky_once = flaky_once
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since_ms, limit):
        self.calls += 1
        if self._flaky_once and self.calls == 1:
            self._flaky_once = False
            raise ConnectionError("simulated network blip")

        since = dt.datetime.fromtimestamp(since_ms / 1000, tz=dt.timezone.utc)
        page = [r for r in self.all_rows if r["timestamp"] >= since][: min(limit, self.page_size)]
        return [
            [int(r["timestamp"].timestamp() * 1000), r["open"], r["high"], r["low"], r["close"], r["volume"]]
            for r in page
        ]


def _make_series(start: dt.datetime, n: int, timeframe_minutes: int = 5) -> list[dict]:
    rows = []
    price = 100.0
    for i in range(n):
        ts = start + dt.timedelta(minutes=timeframe_minutes * i)
        rows.append({"timestamp": ts, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 5.0})
        price += 0.1
    return rows


def test_backfill_writes_all_bars_and_paginates(db_session):
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = _make_series(start, n=1500)  # forces pagination beyond one page of 1000
    now = start + dt.timedelta(minutes=5 * 1500)

    client = MarketDataClient(source=FakeSource(rows))
    result = backfill_symbol_timeframe(db_session, client, "BTC/USDT", "5m", initial_history_days=9999, now=now)

    df = load_candles_df(db_session, "BTC/USDT", "5m")
    assert len(df) == 1500
    assert result["new_rows"] == 1500


def test_backfill_resumes_from_last_stored_bar(db_session):
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = _make_series(start, n=200)
    now = start + dt.timedelta(minutes=5 * 200)

    client = MarketDataClient(source=FakeSource(rows))
    backfill_symbol_timeframe(db_session, client, "BTC/USDT", "5m", initial_history_days=9999, now=now)

    # Simulate new candles arriving later and re-running the backfill —
    # it should only fetch/write the new ones, not redownload everything.
    more_rows = rows + _make_series(start + dt.timedelta(minutes=5 * 200), n=50)
    later_now = start + dt.timedelta(minutes=5 * 250)
    client2 = MarketDataClient(source=FakeSource(more_rows))
    result2 = backfill_symbol_timeframe(db_session, client2, "BTC/USDT", "5m", initial_history_days=9999, now=later_now)

    assert result2["new_rows"] == 50
    df = load_candles_df(db_session, "BTC/USDT", "5m")
    assert len(df) == 250


def test_backfill_survives_transient_network_error(db_session):
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = _make_series(start, n=100)
    now = start + dt.timedelta(minutes=5 * 100)

    client = MarketDataClient(source=FakeSource(rows, flaky_once=True))
    result = backfill_symbol_timeframe(db_session, client, "BTC/USDT", "5m", initial_history_days=9999, now=now)

    assert result["new_rows"] == 100


def test_detect_gaps_finds_missing_bars():
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = _make_series(start, n=10)
    # Remove bars 4 and 5 to create a gap of 2 missing bars.
    del rows[4:6]
    df = pd.DataFrame(rows)

    gaps = detect_gaps(df, "5m")
    assert len(gaps) == 1
    assert gaps[0].missing_bars == 2


def test_detect_duplicates():
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = _make_series(start, n=5)
    rows.append(rows[0])
    df = pd.DataFrame(rows)
    assert detect_duplicates(df) == 1


def test_detect_bad_ohlc_flags_impossible_bar():
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = _make_series(start, n=3)
    rows[1]["high"] = 50.0  # lower than open/close -> invalid
    df = pd.DataFrame(rows)
    bad = detect_bad_ohlc(df)
    assert 1 in bad


def test_validate_ohlcv_clean_series_reports_ok():
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    df = pd.DataFrame(_make_series(start, n=20))
    report = validate_ohlcv(df, "5m")
    assert report.is_clean
    assert "OK" in report.summary()
