import datetime as dt

import pandas as pd
import pytest

from crypto_ai.paper_trading.engine import PaperTradingEngine
from crypto_ai.paper_trading.portfolio import get_cash_balance, get_open_trade
from crypto_ai.paper_trading.simulator import replay_bars, summarize_paper_trading

START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


def test_engine_opens_position_on_buy_signal(db_session):
    engine = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0)
    action = engine.process_bar(db_session, START, close=100.0, signal="BUY", strategy_version="s1")
    db_session.commit()

    assert action["action"] == "OPENED"
    open_trade = get_open_trade(db_session, "BTC/USDT")
    assert open_trade is not None
    assert open_trade.entry_price == pytest.approx(100.0)


def test_engine_does_not_open_second_position_while_one_is_open(db_session):
    engine = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0)
    engine.process_bar(db_session, START, close=100.0, signal="BUY", strategy_version="s1")
    db_session.commit()
    action2 = engine.process_bar(db_session, START + dt.timedelta(minutes=5), close=101.0, signal="BUY", strategy_version="s1")
    db_session.commit()

    assert action2["action"] == "NONE"


def test_engine_closes_position_on_sell_signal_and_records_pnl(db_session):
    engine = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0)
    engine.process_bar(db_session, START, close=100.0, signal="BUY", strategy_version="s1")
    db_session.commit()
    action = engine.process_bar(db_session, START + dt.timedelta(minutes=5), close=110.0, signal="SELL", strategy_version="s1")
    db_session.commit()

    assert action["action"] == "CLOSED"
    assert action["pnl"] == pytest.approx(10.0 * (1000.0 / 100.0), rel=1e-6)  # 10% gain on full balance
    assert get_open_trade(db_session, "BTC/USDT") is None


def test_engine_state_recovers_after_simulated_restart(db_session):
    """A fresh engine instance (simulating a process restart) must read
    the open position from the DB, not lose track of it."""
    engine1 = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0)
    engine1.process_bar(db_session, START, close=100.0, signal="BUY", strategy_version="s1")
    db_session.commit()

    engine2 = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0)  # "new process"
    action = engine2.process_bar(db_session, START + dt.timedelta(minutes=5), close=120.0, signal="SELL", strategy_version="s1")
    db_session.commit()

    assert action["action"] == "CLOSED"
    assert action["pnl"] > 0


def test_stop_loss_closes_position(db_session):
    engine = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0, stop_loss_pct=0.05)
    engine.process_bar(db_session, START, close=100.0, signal="BUY", strategy_version="s1")
    db_session.commit()
    action = engine.process_bar(
        db_session, START + dt.timedelta(minutes=5), close=90.0, signal="HOLD", strategy_version="s1",
        high=90.0, low=88.0,
    )
    db_session.commit()
    assert action["action"] == "CLOSED"
    assert action["exit_reason"] == "stop_loss"
    assert action["pnl"] < 0


def test_fees_reduce_cash_balance(db_session):
    engine = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.01, slippage_pct=0.0)
    engine.process_bar(db_session, START, close=100.0, signal="BUY", strategy_version="s1")
    db_session.commit()
    cash = get_cash_balance(db_session, mode="PAPER")
    assert cash < 1.0  # nearly all cash allocated (position_size_pct=1.0 by default), fee eats a bit more


def test_replay_bars_and_summary(db_session):
    df = pd.DataFrame(
        {
            "timestamp": [START + dt.timedelta(minutes=5 * i) for i in range(6)],
            "close": [100, 100, 110, 110, 105, 105],
        }
    )

    def signal_fn(row):
        if row.name == 1:
            return "BUY", "s1", None, "test buy"
        if row.name == 3:
            return "SELL", "s1", None, "test sell"
        return "HOLD", "s1", None, ""

    engine = PaperTradingEngine(symbol="BTC/USDT", fee_pct=0.0, slippage_pct=0.0)
    actions = replay_bars(db_session, engine, df, signal_fn)
    db_session.commit()

    assert actions[1]["action"] == "OPENED"
    assert actions[3]["action"] == "CLOSED"

    summary = summarize_paper_trading(db_session, "BTC/USDT")
    assert summary["n_trades"] == 1
    assert summary["total_return_pct"] == pytest.approx(10.0, rel=1e-6)
    assert summary["win_rate_pct"] == 100.0
