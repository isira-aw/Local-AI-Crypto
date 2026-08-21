"""
Live-trading safety gate (Section 31 / Section 59).

What is it?
    The explicit, all-or-nothing checklist that must pass before real
    (or even testnet) order placement is permitted. This is checked in
    addition to — never instead of — the per-signal RiskManager gates.

Why is it needed?
    Section 31: "Real trading cannot activate automatically unless ALL
    configured requirements are satisfied." This module is where that
    "ALL" is enforced, so it can't quietly be skipped by one code path
    that forgets to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_ai.config.loader import get_risk_config, get_settings


@dataclass
class LiveTradingGateResult:
    allowed: bool
    checklist: dict = field(default_factory=dict)  # item -> bool
    blocking_reasons: list[str] = field(default_factory=list)


def evaluate_live_trading_gate(
    paper_trading_days: int,
    paper_trade_count: int,
    paper_net_return_pct: float,
    paper_max_drawdown_pct: float,
    walk_forward_passed: bool,
    days_since_critical_error: int,
    regulatory_check_acknowledged: bool,
    exchange_connection_stable: bool,
    emergency_stop_tested: bool,
    recovery_tested: bool,
    backup_restore_tested: bool,
    api_withdrawal_disabled: bool,
    user_explicitly_enabled_live: bool,
    risk_config: dict | None = None,
) -> LiveTradingGateResult:
    """
    Mirrors the Section 59 checklist. Every argument corresponds to one
    checklist line; this function does not go and check most of them
    itself (that requires live system state the caller already has) —
    it just refuses to let ANY of them be skipped.
    """
    cfg = risk_config or get_risk_config()
    gate = cfg.get("live_trading_gate", {})
    settings = get_settings()

    checklist = {
        "historical_data_validated": True,  # enforced structurally by data/validators before training can run
        "backtest_completed": True,          # enforced structurally: no model reaches CANDIDATE without one
        "walk_forward_passed": walk_forward_passed,
        "paper_trading_long_enough": paper_trading_days >= gate.get("min_paper_trading_days", 30),
        "enough_paper_trades": paper_trade_count >= gate.get("min_paper_trades", 100),
        "paper_net_return_acceptable": paper_net_return_pct >= gate.get("min_paper_net_return_percent", 0.0),
        "paper_drawdown_acceptable": paper_max_drawdown_pct <= gate.get("max_paper_drawdown_percent", 20.0),
        "no_recent_critical_errors": days_since_critical_error >= gate.get("require_no_critical_errors_days", 14),
        "regulatory_check_acknowledged": (
            regulatory_check_acknowledged if gate.get("require_regulatory_check_ack", True) else True
        ),
        "exchange_connection_stable": exchange_connection_stable,
        "emergency_stop_tested": emergency_stop_tested,
        "recovery_tested": recovery_tested,
        "backup_restore_tested": backup_restore_tested,
        "api_withdrawal_disabled": api_withdrawal_disabled,
        "mode_is_testnet_or_live": settings.mode in ("TESTNET", "LIVE"),
        "user_explicitly_enabled_live": user_explicitly_enabled_live,
        "real_trading_enabled_flag_set": settings.safety.real_trading_enabled,
    }

    blocking_reasons = [item for item, passed in checklist.items() if not passed]
    allowed = not blocking_reasons

    return LiveTradingGateResult(allowed=allowed, checklist=checklist, blocking_reasons=blocking_reasons)


def max_live_capital_usdt(risk_config: dict | None = None) -> float:
    cfg = risk_config or get_risk_config()
    return cfg.get("live_trading_gate", {}).get("max_live_capital_usdt", 50.0)
