import datetime as dt

from crypto_ai.data.processor.retention import apply_retention_policy
from crypto_ai.database.repositories.market_data_repo import upsert_candles

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def test_old_1m_candles_deleted_but_1h_kept_forever(db_session):
    old_1m = [
        {"timestamp": NOW - dt.timedelta(days=200), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ]
    recent_1m = [
        {"timestamp": NOW - dt.timedelta(days=5), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ]
    old_1h = [
        {"timestamp": NOW - dt.timedelta(days=2000), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ]
    upsert_candles(db_session, "BTC/USDT", "1m", old_1m + recent_1m)
    upsert_candles(db_session, "BTC/USDT", "1h", old_1h)
    db_session.commit()

    deleted = apply_retention_policy(db_session, now=NOW)

    assert deleted["market_data:1m"] == 1  # only the 200-day-old row
    assert "market_data:1h" not in deleted  # 1h retention is null -> kept forever, not even checked

    from crypto_ai.database.repositories.market_data_repo import load_candles_df
    remaining_1m = load_candles_df(db_session, "BTC/USDT", "1m")
    assert len(remaining_1m) == 1
    remaining_1h = load_candles_df(db_session, "BTC/USDT", "1h")
    assert len(remaining_1h) == 1


def test_old_system_events_deleted(db_session):
    from crypto_ai.database.repositories.events_repo import log_event
    from crypto_ai.database.models.monitoring import SystemEvent

    old_event = log_event(db_session, "test", "old_event", context={})
    old_event.timestamp = NOW - dt.timedelta(days=60)
    db_session.commit()

    apply_retention_policy(db_session, now=NOW)

    remaining = db_session.query(SystemEvent).filter_by(event="old_event").count()
    assert remaining == 0


def test_retention_never_touches_predictions_or_trades(db_session):
    from crypto_ai.paper_trading.engine import PaperTradingEngine

    engine = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0)
    old_time = NOW - dt.timedelta(days=2000)
    engine.process_bar(db_session, old_time, close=100.0, signal="BUY", strategy_version="s1")
    db_session.commit()

    apply_retention_policy(db_session, now=NOW)

    from crypto_ai.database.models.trading import PaperTrade
    assert db_session.query(PaperTrade).count() == 1
