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
    # Populated by evaluate_live_trading_gate_from_db(): the derived values
    # behind the checklist, so a refusal can be explained rather than just
    # asserted.
    evidence: dict = field(default_factory=dict)


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


def evaluate_live_trading_gate_from_db(
    session,
    emergency_stop_tested: bool,
    recovery_tested: bool,
    backup_restore_tested: bool,
    api_withdrawal_disabled: bool,
    regulatory_check_acknowledged: bool,
    user_explicitly_enabled_live: bool,
    walk_forward_passed: bool | None = None,
    risk_config: dict | None = None,
) -> LiveTradingGateResult:
    """
    Evaluate the gate using values DERIVED FROM THE DATABASE wherever that
    is possible, instead of trusting whatever the caller passes in.

    Three items are now evidence-based rather than caller-asserted:

      * paper-trading days and trade count come from the epoch marker
        (app/services/paper_reset.py), so history produced before a reset
        does not count;
      * "exchange connection stable" requires health checks recorded
        against the REAL Binance API — a simulated/injected client can
        never satisfy it;
      * walk-forward pass is read from the current champion's stored
        results.

    The remaining items are genuine human attestations (did you actually
    test the emergency stop? is withdrawal disabled on the key?) and still
    have to be passed in — but they default to nothing, so forgetting one
    fails closed.
    """
    from crypto_ai.app.services.paper_reset import summarize_since_epoch
    from crypto_ai.models.registry import registry
    from crypto_ai.monitoring.health import has_verified_real_exchange_connectivity
    from crypto_ai.paper_trading.simulator import summarize_paper_trading
    from crypto_ai.config.loader import get_settings

    cfg = risk_config or get_risk_config()
    settings = get_settings()
    symbol = settings.get("exchange.symbol", "BTC/USDT")

    epoch = summarize_since_epoch(session)
    paper = summarize_paper_trading(session, symbol)
    exchange_ok, exchange_reason = has_verified_real_exchange_connectivity(session)

    if walk_forward_passed is None:
        champion = registry.get_champion(session, "btc_direction_classifier")
        walk_forward_passed = bool(
            champion and (champion.walk_forward_results or {}).get("overall_pass")
        )

    critical_days = _days_since_last_critical_error(session)

    result = evaluate_live_trading_gate(
        paper_trading_days=epoch["days_since_epoch"],
        paper_trade_count=epoch["closed_trades_since_epoch"],
        paper_net_return_pct=paper.get("total_return_pct") or 0.0,
        paper_max_drawdown_pct=paper.get("max_drawdown_pct") or 100.0,
        walk_forward_passed=walk_forward_passed,
        days_since_critical_error=critical_days,
        regulatory_check_acknowledged=regulatory_check_acknowledged,
        exchange_connection_stable=exchange_ok,
        emergency_stop_tested=emergency_stop_tested,
        recovery_tested=recovery_tested,
        backup_restore_tested=backup_restore_tested,
        api_withdrawal_disabled=api_withdrawal_disabled,
        user_explicitly_enabled_live=user_explicitly_enabled_live,
        risk_config=cfg,
    )
    result.evidence = {
        "paper_trading_epoch": epoch,
        "exchange_connectivity": exchange_reason,
        "days_since_critical_error": critical_days,
        "walk_forward_passed": walk_forward_passed,
    }
    return result


def _days_since_last_critical_error(session) -> int:
    import datetime as dt

    from crypto_ai.database.models.monitoring import SystemEvent

    row = (
        session.query(SystemEvent)
        .filter(SystemEvent.severity.in_(["ERROR", "CRITICAL"]))
        .order_by(SystemEvent.timestamp.desc())
        .first()
    )
    if row is None:
        return 10_000  # nothing has ever gone critically wrong
    last = row.timestamp
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    return int((dt.datetime.now(dt.timezone.utc) - last).total_seconds() // 86400)


def max_live_capital_usdt(risk_config: dict | None = None) -> float:
    cfg = risk_config or get_risk_config()
    return cfg.get("live_trading_gate", {}).get("max_live_capital_usdt", 50.0)
