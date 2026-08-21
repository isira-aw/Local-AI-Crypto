import datetime as dt

from crypto_ai.database.repositories.market_data_repo import (
    get_latest_timestamp,
    load_candles_df,
    upsert_candles,
)
from crypto_ai.database.repositories.events_repo import log_event


def _bar(ts, price=100.0):
    return {
        "timestamp": ts,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 10.0,
    }


def test_upsert_candles_deduplicates(db_session):
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = [_bar(base + dt.timedelta(minutes=5 * i)) for i in range(5)]

    upsert_candles(db_session, "BTC/USDT", "5m", rows)
    db_session.commit()

    # Re-insert the same rows plus one new one -> should not duplicate.
    rows2 = [_bar(base + dt.timedelta(minutes=5 * i), price=200.0) for i in range(5)]
    rows2.append(_bar(base + dt.timedelta(minutes=25), price=300.0))
    upsert_candles(db_session, "BTC/USDT", "5m", rows2)
    db_session.commit()

    df = load_candles_df(db_session, "BTC/USDT", "5m")
    assert len(df) == 6
    # Updated values should reflect the second upsert (price=200) since
    # this is an update-on-conflict, not insert-or-ignore.
    assert df.iloc[0]["close"] == 200.0


def test_get_latest_timestamp(db_session):
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    rows = [_bar(base + dt.timedelta(minutes=5 * i)) for i in range(3)]
    upsert_candles(db_session, "BTC/USDT", "5m", rows)
    db_session.commit()

    latest = get_latest_timestamp(db_session, "BTC/USDT", "5m")
    # SQLite (used only in tests) does not preserve tzinfo on DateTime
    # columns the way Postgres does, so compare naive values here.
    expected = (base + dt.timedelta(minutes=10)).replace(tzinfo=None)
    assert latest.replace(tzinfo=None) == expected


def test_get_latest_timestamp_empty(db_session):
    assert get_latest_timestamp(db_session, "BTC/USDT", "5m") is None


def test_log_event_redacts_secrets(db_session):
    row = log_event(
        db_session,
        component="exchange",
        event="api_key_loaded",
        severity="INFO",
        message="loaded key",
        context={"api_secret": "supersecret", "symbol": "BTC/USDT"},
    )
    db_session.commit()
    assert row.context["api_secret"] == "***REDACTED***"
    assert row.context["symbol"] == "BTC/USDT"
