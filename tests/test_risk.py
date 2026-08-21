import datetime as dt
import os

import pytest

from crypto_ai.config import loader as config_loader
from crypto_ai.risk.manager import RiskManager, RiskState
from crypto_ai.risk.position_sizing import calculate_position_size_pct
from crypto_ai.risk.safety_rules import evaluate_live_trading_gate

DEFAULT_RISK_CFG = {
    "position_sizing": {"max_position_percent": 5.0, "min_confidence_to_trade": 0.55},
    "loss_limits": {"max_daily_loss_percent": 2.0, "max_total_drawdown_percent": 20.0, "consecutive_loss_limit": 3, "cooldown_minutes": 120},
    "exposure": {"max_open_positions": 1},
    "safety": {"trading_enabled": True, "emergency_stop": False},
}


def _state(**overrides):
    base = dict(current_equity=1000, starting_equity_today=1000, peak_equity=1000, consecutive_losses=0, open_positions=0, now=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc))
    base.update(overrides)
    return RiskState(**base)


def test_position_size_scales_with_confidence():
    low = calculate_position_size_pct(confidence=0.56, volatility_pct=1.0, max_position_percent=5.0, min_confidence_to_trade=0.55)
    high = calculate_position_size_pct(confidence=0.99, volatility_pct=1.0, max_position_percent=5.0, min_confidence_to_trade=0.55)
    assert 0 <= low < high <= 5.0


def test_position_size_zero_below_min_confidence():
    size = calculate_position_size_pct(confidence=0.4, volatility_pct=1.0, max_position_percent=5.0, min_confidence_to_trade=0.55)
    assert size == 0.0


def test_position_size_shrinks_with_higher_volatility():
    calm = calculate_position_size_pct(confidence=0.9, volatility_pct=1.0, max_position_percent=5.0, min_confidence_to_trade=0.55)
    volatile = calculate_position_size_pct(confidence=0.9, volatility_pct=5.0, max_position_percent=5.0, min_confidence_to_trade=0.55)
    assert volatile < calm


def test_risk_manager_allows_buy_when_healthy():
    rm = RiskManager(DEFAULT_RISK_CFG)
    decision = rm.evaluate("BUY", confidence=0.9, state=_state(), volatility_pct=1.0)
    assert decision.allowed_signal == "BUY"
    assert not decision.blocked
    assert decision.position_size_pct > 0


def test_risk_manager_blocks_on_emergency_stop():
    cfg = {**DEFAULT_RISK_CFG, "safety": {"trading_enabled": True, "emergency_stop": True}}
    rm = RiskManager(cfg)
    decision = rm.evaluate("BUY", confidence=0.9, state=_state(), volatility_pct=1.0)
    assert decision.allowed_signal == "WAIT"
    assert "emergency_stop_active" in decision.reason_codes


def test_risk_manager_blocks_on_daily_loss_limit():
    rm = RiskManager(DEFAULT_RISK_CFG)
    state = _state(current_equity=970, starting_equity_today=1000)  # -3% today, limit is 2%
    decision = rm.evaluate("BUY", confidence=0.9, state=state, volatility_pct=1.0)
    assert decision.allowed_signal == "WAIT"
    assert "daily_loss_limit_reached" in decision.reason_codes


def test_risk_manager_blocks_on_max_drawdown():
    rm = RiskManager(DEFAULT_RISK_CFG)
    state = _state(current_equity=750, peak_equity=1000, starting_equity_today=760)  # -25% drawdown, limit 20%
    decision = rm.evaluate("BUY", confidence=0.9, state=state, volatility_pct=1.0)
    assert decision.allowed_signal == "WAIT"
    assert "max_drawdown_reached" in decision.reason_codes


def test_risk_manager_enforces_cooldown_after_consecutive_losses():
    rm = RiskManager(DEFAULT_RISK_CFG)
    now = dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    state = _state(consecutive_losses=3, last_loss_at=now - dt.timedelta(minutes=10), now=now)
    decision = rm.evaluate("BUY", confidence=0.9, state=state, volatility_pct=1.0)
    assert decision.allowed_signal == "WAIT"
    assert "cooldown_after_consecutive_losses" in decision.reason_codes


def test_risk_manager_allows_after_cooldown_expires():
    rm = RiskManager(DEFAULT_RISK_CFG)
    now = dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    state = _state(consecutive_losses=3, last_loss_at=now - dt.timedelta(minutes=200), now=now)
    decision = rm.evaluate("BUY", confidence=0.9, state=state, volatility_pct=1.0)
    assert decision.allowed_signal == "BUY"


def test_risk_manager_blocks_low_confidence():
    rm = RiskManager(DEFAULT_RISK_CFG)
    decision = rm.evaluate("BUY", confidence=0.3, state=_state(), volatility_pct=1.0)
    assert decision.allowed_signal == "WAIT"
    assert "confidence_below_minimum" in decision.reason_codes


def test_risk_manager_blocks_second_position():
    rm = RiskManager(DEFAULT_RISK_CFG)
    decision = rm.evaluate("BUY", confidence=0.9, state=_state(open_positions=1), volatility_pct=1.0)
    assert decision.allowed_signal == "WAIT"
    assert "max_open_positions_reached" in decision.reason_codes


def test_risk_manager_never_blocks_wait_or_sell():
    rm = RiskManager(DEFAULT_RISK_CFG)
    decision = rm.evaluate("SELL", confidence=0.9, state=_state(open_positions=1), volatility_pct=1.0)
    assert decision.allowed_signal == "SELL"


def test_risk_manager_respects_trading_disabled_in_live_mode():
    os.environ["MODE"] = "LIVE"
    os.environ["TRADING_ENABLED"] = "false"
    config_loader.reload_config()
    try:
        rm = RiskManager(DEFAULT_RISK_CFG)
        decision = rm.evaluate("BUY", confidence=0.9, state=_state(), volatility_pct=1.0)
        assert decision.allowed_signal == "WAIT"
        assert "trading_disabled" in decision.reason_codes
    finally:
        os.environ["MODE"] = "RESEARCH"
        os.environ["TRADING_ENABLED"] = "false"
        config_loader.reload_config()


def test_risk_score_increases_with_worse_conditions():
    rm = RiskManager(DEFAULT_RISK_CFG)
    calm = rm.compute_risk_score(_state(), volatility_pct=0.5)
    stressed = rm.compute_risk_score(
        _state(current_equity=800, peak_equity=1000, starting_equity_today=820, consecutive_losses=3), volatility_pct=8.0
    )
    assert stressed > calm


def _full_pass_gate_kwargs(**overrides):
    kwargs = dict(
        paper_trading_days=45,
        paper_trade_count=150,
        paper_net_return_pct=5.0,
        paper_max_drawdown_pct=10.0,
        walk_forward_passed=True,
        days_since_critical_error=30,
        regulatory_check_acknowledged=True,
        exchange_connection_stable=True,
        emergency_stop_tested=True,
        recovery_tested=True,
        backup_restore_tested=True,
        api_withdrawal_disabled=True,
        user_explicitly_enabled_live=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_live_gate_blocked_when_not_enough_paper_trading(db_session=None):
    result = evaluate_live_trading_gate(**_full_pass_gate_kwargs(paper_trading_days=5, paper_trade_count=10))
    assert result.allowed is False
    assert "paper_trading_long_enough" in result.blocking_reasons
    assert "enough_paper_trades" in result.blocking_reasons


def test_live_gate_blocked_when_regulatory_check_missing():
    result = evaluate_live_trading_gate(**_full_pass_gate_kwargs(regulatory_check_acknowledged=False))
    assert result.allowed is False
    assert "regulatory_check_acknowledged" in result.blocking_reasons


def test_live_gate_blocked_by_default_flags_even_if_everything_else_passes():
    # In RESEARCH mode with real_trading_enabled=false (test defaults),
    # the gate must still refuse — this is the "never automatic" guarantee.
    result = evaluate_live_trading_gate(**_full_pass_gate_kwargs())
    assert result.allowed is False
    assert "real_trading_enabled_flag_set" in result.blocking_reasons
    assert "mode_is_testnet_or_live" in result.blocking_reasons


def test_live_gate_allowed_when_everything_passes_and_flags_set():
    os.environ["MODE"] = "LIVE"
    os.environ["REAL_TRADING_ENABLED"] = "true"
    config_loader.reload_config()
    try:
        result = evaluate_live_trading_gate(**_full_pass_gate_kwargs())
        assert result.allowed is True
        assert result.blocking_reasons == []
    finally:
        os.environ["MODE"] = "RESEARCH"
        os.environ["REAL_TRADING_ENABLED"] = "false"
        config_loader.reload_config()
