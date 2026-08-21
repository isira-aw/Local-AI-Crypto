import os

import pytest

from crypto_ai.config import loader as config_loader
from crypto_ai.llm import analyst
from crypto_ai.llm.client import check_llm_health, generate_safe, get_llm_client


@pytest.fixture()
def llm_disabled():
    os.environ["LLM_ENABLED"] = "false"
    config_loader.reload_config()
    yield
    os.environ["LLM_ENABLED"] = "false"  # tests/conftest.py default
    config_loader.reload_config()


@pytest.fixture()
def llm_enabled_unreachable():
    os.environ["LLM_ENABLED"] = "true"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1"  # guaranteed nothing is listening
    os.environ["LLM_TIMEOUT_SECONDS"] = "1"
    config_loader.reload_config()
    yield
    os.environ["LLM_ENABLED"] = "false"
    config_loader.reload_config()


def test_get_llm_client_returns_none_when_disabled(llm_disabled):
    assert get_llm_client() is None


def test_generate_safe_returns_none_when_disabled(llm_disabled):
    assert generate_safe("hello") is None


def test_check_llm_health_reports_disabled(llm_disabled):
    health = check_llm_health()
    assert health["status"] == "DISABLED"


def test_generate_safe_never_raises_when_unreachable(llm_enabled_unreachable):
    # Must not raise even though nothing is listening on that port.
    result = generate_safe("hello")
    assert result is None


def test_check_llm_health_reports_down_when_unreachable(llm_enabled_unreachable):
    health = check_llm_health()
    assert health["status"] == "DOWN"


def test_explain_signal_falls_back_to_template_when_llm_disabled(llm_disabled):
    text = analyst.explain_signal({
        "symbol": "BTC/USDT", "price": 65000, "signal": "WAIT", "confidence": 0.6, "risk_score": 0.3,
    })
    assert "WAIT" in text
    assert "unavailable" in text.lower() or "template" in text.lower()


def test_summarize_daily_report_falls_back_when_llm_disabled(llm_disabled):
    text = analyst.summarize_daily_report({"date": "2026-01-01", "mode": "PAPER"})
    assert isinstance(text, str)
    assert len(text) > 0


def test_trading_pipeline_unaffected_by_llm_being_off(llm_disabled):
    """Section 18: the trading engine must still function if the LLM is
    unavailable — a lightweight smoke check that nothing else imports
    or requires the LLM to be reachable."""
    from crypto_ai.backtesting.engine import BacktestEngine
    import pandas as pd
    import datetime as dt

    df = pd.DataFrame({
        "timestamp": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=5 * i) for i in range(4)],
        "close": [100, 100, 110, 110],
        "signal": ["HOLD", "BUY", "HOLD", "SELL"],
    })
    report = BacktestEngine(initial_balance=1000).run(df)
    assert report.metrics["n_trades"] == 1
