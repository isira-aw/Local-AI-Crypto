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
    paper_net_return_pct: float | None,
    paper_max_drawdown_pct: float | None,
    walk_forward_passed: bool,
    days_since_critical_error: int,
    regulatory_check_acknowledged: bool,
    exchange_connection_stable: bool,
    emergency_stop_tested: bool,
    recovery_tested: bool,
    backup_restore_tested: bool,
    api_withdrawal_disabled: bool,
    user_explicitly_enabled_live: bool,
    historical_data_validated: bool = False,
    backtest_completed: bool = False,
    days_of_operating_history: float = 0.0,
    risk_config: dict | None = None,
) -> LiveTradingGateResult:
    """
    Mirrors the Section 59 checklist.

    Every item FAILS CLOSED. In particular the three arguments added after
    the first version — historical_data_validated, backtest_completed and
    days_of_operating_history — default to the failing value, because
    their predecessors were hardcoded True / silently satisfied by an
    empty database. An absent measurement means "unknown", and unknown
    must never read as PASS on a checklist whose whole purpose is to
    gate real money.
    """
    cfg = risk_config or get_risk_config()
    gate = cfg.get("live_trading_gate", {})
    settings = get_settings()

    min_clean_days = gate.get("require_no_critical_errors_days", 14)

    # "No critical errors for 14 days" is only meaningful if the system has
    # actually been observed for 14 days. A fresh database has no errors
    # because it has no history — that is not a clean track record.
    enough_history_to_judge = days_of_operating_history >= min_clean_days

    # A return of 0.0 from an account that never traded is not a passing
    # return, it is an absent one.
    has_paper_results = paper_trade_count > 0 and paper_net_return_pct is not None

    checklist = {
        "historical_data_validated": historical_data_validated,
        "backtest_completed": backtest_completed,
        "walk_forward_passed": walk_forward_passed,
        "paper_trading_long_enough": paper_trading_days >= gate.get("min_paper_trading_days", 30),
        "enough_paper_trades": paper_trade_count >= gate.get("min_paper_trades", 100),
        "paper_net_return_acceptable": (
            has_paper_results and paper_net_return_pct >= gate.get("min_paper_net_return_percent", 0.0)
        ),
        "paper_drawdown_acceptable": (
            paper_max_drawdown_pct is not None
            and paper_max_drawdown_pct <= gate.get("max_paper_drawdown_percent", 20.0)
        ),
        "no_recent_critical_errors": (
            enough_history_to_judge and days_since_critical_error >= min_clean_days
        ),
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
    data_ok, data_reason = check_historical_data_validated(session, symbol)
    backtest_ok, backtest_reason = check_backtest_completed(session, symbol)
    history_days = days_of_operating_history(session)

    result = evaluate_live_trading_gate(
        paper_trading_days=epoch["days_since_epoch"],
        paper_trade_count=epoch["closed_trades_since_epoch"],
        # Pass these through as-is. Coercing None to 0.0/100.0 here is what
        # made item 6 pass vacuously on an empty account; the gate now
        # distinguishes "no result" from "a result of zero".
        paper_net_return_pct=paper.get("total_return_pct"),
        paper_max_drawdown_pct=paper.get("max_drawdown_pct"),
        walk_forward_passed=walk_forward_passed,
        days_since_critical_error=critical_days,
        regulatory_check_acknowledged=regulatory_check_acknowledged,
        exchange_connection_stable=exchange_ok,
        emergency_stop_tested=emergency_stop_tested,
        recovery_tested=recovery_tested,
        backup_restore_tested=backup_restore_tested,
        api_withdrawal_disabled=api_withdrawal_disabled,
        user_explicitly_enabled_live=user_explicitly_enabled_live,
        historical_data_validated=data_ok,
        backtest_completed=backtest_ok,
        days_of_operating_history=history_days,
        risk_config=cfg,
    )
    result.evidence = {
        "historical_data": data_reason,
        "backtest": backtest_reason,
        "days_of_operating_history": round(history_days, 2),
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


def check_historical_data_validated(
    session, symbol: str | None = None, timeframe: str | None = None,
) -> tuple[bool, str]:
    """
    Section 59 item 1, as a REAL check.

    Was hardcoded True on the reasoning that "validators run before
    training anyway". That is an argument about the code, not evidence
    about the data actually sitting in this database — so it would have
    reported PASS on an empty database, or on data full of gaps.

    This loads the configured symbol's primary-timeframe candles and runs
    the same validator the downloader runs (data/validators), requiring:
    enough bars to train on, and a clean report.
    """
    from crypto_ai.data.validators.ohlcv_validators import validate_ohlcv
    from crypto_ai.database.repositories.market_data_repo import load_candles_df

    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    wf = settings.get("walk_forward", {}) or {}
    min_bars = wf.get("min_train_bars", 20000) + wf.get("final_test_bars", 4000)

    df = load_candles_df(session, symbol, timeframe)
    if df.empty:
        return False, f"no {symbol} {timeframe} market data in the database"
    if len(df) < min_bars:
        return False, (
            f"only {len(df)} {timeframe} bars stored; need >= {min_bars} "
            f"to train and hold out a test set"
        )

    report = validate_ohlcv(df, timeframe)
    if not report.is_clean:
        return False, f"validator reports issues: {report.summary()}"
    return True, f"{len(df)} {timeframe} bars validated clean"


def check_backtest_completed(session, symbol: str | None = None) -> tuple[bool, str]:
    """
    Section 59 item 2, as a REAL check.

    Was hardcoded True. Now inspects the backtest_runs table the same way
    item 3 inspects model_versions: a backtest must actually have been
    recorded for this symbol.
    """
    from crypto_ai.database.models.trading import BacktestRun

    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")

    row = (
        session.query(BacktestRun)
        .filter(BacktestRun.symbol == symbol)
        .order_by(BacktestRun.created_at.desc())
        .first()
    )
    if row is None:
        return False, f"no backtest run recorded for {symbol} (run: python run.py backtest)"

    metrics = row.metrics or {}
    return True, (
        f"most recent backtest {row.created_at:%Y-%m-%d} "
        f"({metrics.get('n_trades', 0)} trades, "
        f"return {metrics.get('total_return_pct', 0):.2f}%)"
    )


def days_of_operating_history(session) -> float:
    """
    How long this installation has actually been observed, measured from
    the oldest system event. Feeds the "no critical errors for N days"
    item, which is meaningless without N days of observation.
    """
    import datetime as dt

    from crypto_ai.database.models.monitoring import SystemEvent

    row = session.query(SystemEvent).order_by(SystemEvent.timestamp.asc()).first()
    if row is None:
        return 0.0
    oldest = row.timestamp
    if oldest.tzinfo is None:
        oldest = oldest.replace(tzinfo=dt.timezone.utc)
    return max(0.0, (dt.datetime.now(dt.timezone.utc) - oldest).total_seconds() / 86400)


def max_live_capital_usdt(risk_config: dict | None = None) -> float:
    cfg = risk_config or get_risk_config()
    return cfg.get("live_trading_gate", {}).get("max_live_capital_usdt", 50.0)
