#!/usr/bin/env python3
"""
End-to-end validation harness.

Exercises the REAL code paths (downloader, validators, feature pipeline,
labeling, walk-forward training, registry, backtester, risk engine, paper
trading, prediction evaluation, drift detection, reporting, scheduler jobs)
against a SIMULATED Binance API, because a sandbox/CI environment normally
has no route to api.binance.com.

This is not a unit test — it's a full-workflow smoke/integration run that
reports a pass/fail table. Run with:

    python scripts/e2e_validation.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/data_store/_e2e.db")
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("MODE", "RESEARCH")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def check(name: str):
    """Decorator-ish context: records PASS/FAIL for a validation step."""

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc is None:
                RESULTS.append((name, "PASS", ""))
            else:
                RESULTS.append((name, "FAIL", f"{exc_type.__name__}: {exc}"))
                traceback.print_exception(exc_type, exc, tb)
            return True  # swallow, keep going

    return _Ctx()


# ---------------------------------------------------------------------
# Simulated exchange
# ---------------------------------------------------------------------
class SimulatedBinance:
    """
    Generates a realistic BTC-like price series (geometric brownian motion
    with volatility clustering) and serves it through the same interface
    ccxt's fetch_ohlcv exposes, including a 1000-candle page cap.
    """

    def __init__(self, n_bars: int = 60000, timeframe_minutes: int = 5, seed: int = 7,
                 fail_every: int | None = None):
        rng = np.random.default_rng(seed)
        # Volatility clustering: regime-switching sigma
        regimes = rng.choice([0.0008, 0.002, 0.005], size=n_bars, p=[0.6, 0.3, 0.1])
        returns = rng.normal(0, 1, n_bars) * regimes
        close = 40000 * np.exp(np.cumsum(returns))
        open_ = np.concatenate([[close[0]], close[:-1]])
        # A real candle always satisfies high >= max(open, close) and
        # low <= min(open, close). Deriving high/low from close alone
        # produces impossible bars (which the validator correctly rejects).
        spread = np.abs(rng.normal(0, 0.0007, n_bars))
        body_high = np.maximum(open_, close)
        body_low = np.minimum(open_, close)
        high = body_high * (1 + spread)
        low = body_low * (1 - spread)
        volume = np.abs(rng.normal(50, 20, n_bars)) + 1

        start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        self.rows = [
            {
                "timestamp": start + dt.timedelta(minutes=timeframe_minutes * i),
                "open": float(open_[i]), "high": float(high[i]), "low": float(low[i]),
                "close": float(close[i]), "volume": float(volume[i]),
            }
            for i in range(n_bars)
        ]
        self.calls = 0
        self.fail_every = fail_every

    def fetch_ohlcv(self, symbol, timeframe, since_ms, limit):
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise ConnectionError("simulated transient network failure")
        since = dt.datetime.fromtimestamp(since_ms / 1000, tz=dt.timezone.utc)
        page = [r for r in self.rows if r["timestamp"] >= since][:min(limit, 1000)]
        return [
            [int(r["timestamp"].timestamp() * 1000), r["open"], r["high"], r["low"], r["close"], r["volume"]]
            for r in page
        ]

    @property
    def last_timestamp(self):
        return self.rows[-1]["timestamp"]


def fresh_database():
    db_path = PROJECT_ROOT / "data_store" / "_e2e.db"
    if db_path.exists():
        db_path.unlink()
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")


def main() -> int:
    print("=" * 78)
    print("END-TO-END VALIDATION — Local AI Crypto")
    print("=" * 78)

    from crypto_ai.database.base import session_scope

    # ---------- 0. Schema ----------
    with check("00 database schema creates cleanly (alembic upgrade head)"):
        fresh_database()

    SYMBOL, TIMEFRAME = "BTC/USDT", "5m"
    sim = SimulatedBinance(n_bars=60000, fail_every=17)  # inject periodic failures

    # ---------- 1. Downloader ----------
    with check("01 downloader backfills through simulated exchange (with transient failures)"):
        from crypto_ai.data.downloader.historical import backfill_symbol_timeframe
        from crypto_ai.exchange.market_data import MarketDataClient

        client = MarketDataClient(source=sim)
        with session_scope() as s:
            res = backfill_symbol_timeframe(
                s, client, SYMBOL, TIMEFRAME, initial_history_days=99999,
                now=sim.last_timestamp + dt.timedelta(minutes=5),
            )
        assert res["new_rows"] == 60000, f"expected 60000 rows, got {res['new_rows']}"

    with check("02 downloader is idempotent / resumes without duplicating"):
        from crypto_ai.data.downloader.historical import backfill_symbol_timeframe
        from crypto_ai.database.repositories.market_data_repo import load_candles_df
        from crypto_ai.exchange.market_data import MarketDataClient

        with session_scope() as s:
            res2 = backfill_symbol_timeframe(
                s, MarketDataClient(source=sim), SYMBOL, TIMEFRAME, initial_history_days=99999,
                now=sim.last_timestamp + dt.timedelta(minutes=5),
            )
            df = load_candles_df(s, SYMBOL, TIMEFRAME)
        assert res2["new_rows"] == 0, f"re-run should add 0 rows, added {res2['new_rows']}"
        assert len(df) == 60000, f"table should still hold 60000 rows, has {len(df)}"

    # ---------- 2. Validation ----------
    with check("03 validators report clean data as clean"):
        from crypto_ai.data.validators.ohlcv_validators import validate_ohlcv
        from crypto_ai.database.repositories.market_data_repo import load_candles_df

        with session_scope() as s:
            df = load_candles_df(s, SYMBOL, TIMEFRAME)
        report = validate_ohlcv(df, TIMEFRAME)
        assert report.is_clean, f"expected clean, got: {report.summary()}"

    # ---------- 3. Features + labels ----------
    with check("04 feature pipeline produces finite, NaN-free features"):
        from crypto_ai.features.feature_pipeline import compute_features
        from crypto_ai.database.repositories.market_data_repo import load_candles_df

        with session_scope() as s:
            df = load_candles_df(s, SYMBOL, TIMEFRAME)
        global FEATURES, RAW_DF
        RAW_DF = df
        FEATURES = compute_features(df)
        assert not FEATURES.isna().any().any(), "features contain NaN"
        num = FEATURES.drop(columns=["timestamp"])
        assert np.isfinite(num.to_numpy()).all(), "features contain inf"
        assert len(FEATURES) > 59000, f"unexpectedly few feature rows: {len(FEATURES)}"

    with check("05 labels generated automatically with all three classes present"):
        from crypto_ai.features.labels import compute_labels, label_distribution

        global LABELS
        LABELS = compute_labels(RAW_DF)
        dist = label_distribution(LABELS["label"])
        assert dist["BUY"]["count"] > 0 and dist["SELL"]["count"] > 0 and dist["HOLD"]["count"] > 0, dist
        print(f"    label distribution: {dist}")

    # ---------- 4. Leakage guard ----------
    with check("06 walk-forward plan has no leakage and never touches final test set"):
        from crypto_ai.config.loader import get_settings
        from crypto_ai.models.training.walk_forward import assert_no_leakage, generate_walk_forward_folds

        st = get_settings()
        wf = st.get("walk_forward", {})
        horizon = st.get("labeling.horizon_bars", 12)
        merged_len = min(len(FEATURES), len(LABELS))
        plan = generate_walk_forward_folds(
            n_rows=merged_len, n_folds=wf["n_folds"], min_train_bars=wf["min_train_bars"],
            validation_bars=wf["validation_bars"], embargo_bars=wf["embargo_bars"],
            final_test_bars=wf["final_test_bars"],
        )
        assert_no_leakage(plan, wf["embargo_bars"], horizon)
        assert len(plan.folds) >= 3, f"expected multiple folds, got {len(plan.folds)}"
        print(f"    folds={len(plan.folds)} final_test_start={plan.final_test_start} of {merged_len}")

    # ---------- 5. Training ----------
    with check("07 full walk-forward training pipeline runs on 60k bars"):
        from crypto_ai.models.training.pipeline import run_training_pipeline

        global TRAIN_SUMMARY
        with session_scope() as s:
            TRAIN_SUMMARY = run_training_pipeline(
                s, SYMBOL, TIMEFRAME, FEATURES, LABELS, model_name="btc_direction_classifier",
            )
        for r in TRAIN_SUMMARY["results"]:
            assert "error" not in r, f"{r['algorithm']} errored: {r.get('error')}"
            assert "version_label" in r, f"{r['algorithm']} produced no registered version: {r}"
            print(f"    {r['algorithm']}: pass={r['overall_pass']} version={r.get('version_label')}")
        print(f"    promotion: {TRAIN_SUMMARY['promotion']}")

    with check("08 every training attempt is registered (candidate OR rejected, none lost)"):
        from crypto_ai.database.models.ml import ModelVersion

        with session_scope() as s:
            versions = s.query(ModelVersion).all()
            statuses = [v.status for v in versions]
        assert len(versions) == len(TRAIN_SUMMARY["results"]), f"{len(versions)} versions vs {len(TRAIN_SUMMARY['results'])} attempts"
        assert all(st in ("CANDIDATE", "CHAMPION", "REJECTED", "RETIRED") for st in statuses), statuses
        print(f"    statuses: {statuses}")

    with check("09 model artifacts written to disk and reloadable"):
        import joblib
        from crypto_ai.database.models.ml import ModelVersion

        with session_scope() as s:
            for v in s.query(ModelVersion).all():
                p = Path(v.artifact_path)
                assert p.exists(), f"missing artifact {p}"
                est = joblib.load(p)
                assert hasattr(est, "predict_proba"), f"{v.version_label} has no predict_proba"

    # ---------- 6. Backtest ----------
    with check("10 backtester runs on champion signals and reports full metrics"):
        from crypto_ai.backtesting.engine import BacktestEngine
        from crypto_ai.models.registry import registry
        from crypto_ai.strategies.ml_strategy import predictions_to_signals

        with session_scope() as s:
            champ = registry.get_champion(s, "btc_direction_classifier")
            if champ is None:
                print("    (no champion promoted — backtesting the best rejected model instead)")
                from crypto_ai.database.models.ml import ModelVersion
                champ = s.query(ModelVersion).first()
            est = registry.load_estimator(champ)
        fcols = [c for c in FEATURES.columns if c != "timestamp"]
        proba = est.predict_proba(FEATURES[fcols])
        cls = est.classes_
        pred = pd.Series([cls[i] for i in proba.argmax(axis=1)])
        conf = pd.Series(proba[np.arange(len(proba)), proba.argmax(axis=1)])
        sig = predictions_to_signals(pred, conf)
        bt_df = pd.DataFrame({
            "timestamp": FEATURES["timestamp"].values,
            "close": RAW_DF.set_index("timestamp").loc[FEATURES["timestamp"], "close"].values,
            "signal": sig.values,
        })
        rep = BacktestEngine(initial_balance=1000, timeframe=TIMEFRAME).run(bt_df, symbol=SYMBOL)
        m = rep.metrics
        for k in ("total_return_pct", "max_drawdown_pct", "sharpe_ratio", "win_rate_pct", "profit_factor", "n_trades"):
            assert k in m, f"metric {k} missing"
        print(f"    trades={m['n_trades']} return={m['total_return_pct']:.2f}% "
              f"buyhold={(rep.buy_and_hold_final_balance/1000-1)*100:.2f}% "
              f"dd={m['max_drawdown_pct']:.2f}% sharpe={m['sharpe_ratio']:.2f}")
        if m["suspicious_flags"]:
            print(f"    suspicious flags (expected on synthetic data): {m['suspicious_flags']}")

    with check("11 backtest persists to backtest_runs table"):
        from crypto_ai.database.repositories.backtest_repo import save_backtest_run
        from crypto_ai.database.models.trading import BacktestRun

        with session_scope() as s:
            save_backtest_run(s, rep, "ml_strategy_v1", bt_df["timestamp"].iloc[0], bt_df["timestamp"].iloc[-1])
        with session_scope() as s:
            assert s.query(BacktestRun).count() == 1

    # ---------- 7. Risk engine ----------
    with check("12 risk engine blocks every unsafe condition"):
        from crypto_ai.risk.manager import RiskManager, RiskState

        rm = RiskManager()
        now = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
        base = dict(current_equity=1000, starting_equity_today=1000, peak_equity=1000, now=now)
        cases = {
            "daily_loss": (RiskState(**{**base, "current_equity": 900}), "daily_loss_limit_reached"),
            "drawdown": (RiskState(**{**base, "current_equity": 700, "peak_equity": 1000, "starting_equity_today": 705}), "max_drawdown_reached"),
            "open_position": (RiskState(**base, open_positions=1), "max_open_positions_reached"),
            "cooldown": (RiskState(**base, consecutive_losses=5, last_loss_at=now - dt.timedelta(minutes=1)), "cooldown_after_consecutive_losses"),
        }
        for label, (state, expected_code) in cases.items():
            d = rm.evaluate("BUY", confidence=0.99, state=state, volatility_pct=1.0)
            assert d.allowed_signal == "WAIT", f"{label} did not block"
            assert expected_code in d.reason_codes, f"{label}: expected {expected_code}, got {d.reason_codes}"
        d_ok = rm.evaluate("BUY", confidence=0.99, state=RiskState(**base), volatility_pct=1.0)
        assert d_ok.allowed_signal == "BUY", f"healthy state should allow BUY: {d_ok.reason_codes}"
        assert 0 < d_ok.position_size_pct <= 5.0, f"position size out of bounds: {d_ok.position_size_pct}"

    with check("13 live-trading gate refuses by default even when all metrics pass"):
        from crypto_ai.risk.safety_rules import evaluate_live_trading_gate

        g = evaluate_live_trading_gate(
            paper_trading_days=999, paper_trade_count=9999, paper_net_return_pct=50.0,
            paper_max_drawdown_pct=1.0, walk_forward_passed=True, days_since_critical_error=999,
            regulatory_check_acknowledged=True, exchange_connection_stable=True,
            emergency_stop_tested=True, recovery_tested=True, backup_restore_tested=True,
            api_withdrawal_disabled=True, user_explicitly_enabled_live=True,
        )
        assert g.allowed is False, "gate must refuse in RESEARCH mode with real trading disabled"
        assert "real_trading_enabled_flag_set" in g.blocking_reasons

    # ---------- 8. Paper trading over many bars ----------
    with check("14 paper trading runs 500 live ticks end-to-end via live_pipeline"):
        from crypto_ai.app.services.live_pipeline import run_prediction_and_paper_trade_tick
        from crypto_ai.database.repositories.market_data_repo import upsert_candles
        from crypto_ai.models.registry import registry

        # Reset market_data to a prefix, then feed bars one at a time so the
        # tick sees a growing "live" series exactly as it would in production.
        with session_scope() as s:
            if registry.get_champion(s, "btc_direction_classifier") is None:
                # force-promote the first candidate/rejected so the tick path is exercised
                from crypto_ai.database.models.ml import ModelVersion
                v = s.query(ModelVersion).first()
                registry.promote_to_champion(s, v)

        statuses = {}
        actions = {}
        with session_scope() as s:
            # Trim back to the first 55k bars so the loop below can feed the
            # rest in one at a time. Compare on a naive value: SQLite stores
            # these columns without tzinfo, and binding a tz-aware datetime
            # makes the comparison ambiguous.
            from crypto_ai.database.models.market import MarketData
            cutoff = sim.rows[55000]["timestamp"].replace(tzinfo=None)
            s.query(MarketData).filter(MarketData.timestamp > cutoff).delete()
        for i in range(55000, 55500):
            with session_scope() as s:
                upsert_candles(s, SYMBOL, TIMEFRAME, [dict(sim.rows[i])])
            with session_scope() as s:
                r = run_prediction_and_paper_trade_tick(s, SYMBOL, TIMEFRAME)
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            if r["status"] == "ok":
                actions[r["action"]["action"]] = actions.get(r["action"]["action"], 0) + 1
        print(f"    tick statuses: {statuses}")
        print(f"    paper actions: {actions}")
        assert statuses.get("ok", 0) > 400, f"most ticks should succeed: {statuses}"

    with check("15 paper trades + portfolio snapshots recorded with sane P/L"):
        from crypto_ai.database.models.trading import PaperTrade, PortfolioSnapshot

        with session_scope() as s:
            trades = s.query(PaperTrade).all()
            snaps = s.query(PortfolioSnapshot).count()
        closed = [t for t in trades if t.status == "CLOSED"]
        print(f"    trades={len(trades)} closed={len(closed)} snapshots={snaps}")
        assert snaps > 0, "no portfolio snapshots written"
        for t in closed:
            assert t.pnl is not None and np.isfinite(t.pnl), f"bad pnl {t.pnl}"
            assert t.exit_price is not None and t.exit_price > 0
            assert t.fee >= 0

    with check("16 paper-trading summary metrics computed"):
        from crypto_ai.paper_trading.simulator import summarize_paper_trading

        with session_scope() as s:
            summ = summarize_paper_trading(s, SYMBOL, TIMEFRAME)
        print(f"    equity={summ['current_equity']} return={summ['total_return_pct']} trades={summ['n_trades']}")
        assert summ["current_equity"] is not None

    # ---------- 9. Prediction evaluation ----------
    with check("17 predictions recorded and evaluated once horizon elapses"):
        from crypto_ai.models.evaluation.prediction_evaluator import (
            evaluate_due_predictions, summarize_prediction_accuracy,
        )
        from crypto_ai.database.models.trading import Prediction
        from crypto_ai.database.repositories.market_data_repo import upsert_candles

        # feed forward enough bars so early predictions' horizons elapse
        with session_scope() as s:
            upsert_candles(s, SYMBOL, TIMEFRAME, [dict(r) for r in sim.rows[55500:55600]])
        with session_scope() as s:
            n_eval = evaluate_due_predictions(s, now=sim.rows[55599]["timestamp"])
            total = s.query(Prediction).count()
            acc = summarize_prediction_accuracy(s, symbol=SYMBOL)
        print(f"    predictions={total} evaluated_now={n_eval} accuracy={acc['accuracy_pct']}% "
              f"directional={acc['directional_accuracy_pct']}%")
        assert total >= 450, f"expected ~500 predictions from 500 ticks, got {total}"
        assert n_eval > 0, "no predictions became due — evaluator may be broken"
        assert acc["n_predictions"] > 0

    # ---------- 10. Drift ----------
    with check("18 drift detection distinguishes stable vs shifted regimes"):
        from crypto_ai.monitoring.drift import run_drift_check

        fcols = [c for c in FEATURES.columns if c != "timestamp"]
        train_part = FEATURES.iloc[:30000]
        same_part = FEATURES.iloc[30000:40000]
        stable = run_drift_check(train_part, same_part, fcols)
        shifted_df = same_part.copy()
        for c in fcols:
            shifted_df[c] = shifted_df[c] * 50 + 1000
        shifted = run_drift_check(train_part, shifted_df, fcols, recent_accuracy_pct=20.0, expected_accuracy_pct=60.0)
        print(f"    stable max_psi={stable.max_psi:.4f} flagged={stable.flagged_for_review}")
        print(f"    shifted max_psi={shifted.max_psi:.4f} flagged={shifted.flagged_for_review}")
        assert shifted.flagged_for_review, "drift check failed to flag an obvious distribution shift"
        assert shifted.accuracy_drift_flagged, "accuracy drift not flagged"

    # ---------- 11. Reporting ----------
    with check("19 daily report generates with all sections"):
        from crypto_ai.app.services.reporting import format_text_report, generate_daily_report

        with session_scope() as s:
            rep_d = generate_daily_report(s, symbol=SYMBOL)
        txt = format_text_report(rep_d)
        for section in ("CRYPTO AI DAILY REPORT", "Paper trading:", "Model:", "Recommendation:"):
            assert section in txt, f"report missing section: {section}"
        assert rep_d["recommendation"] in ("CONTINUE RESEARCH", "NOT READY FOR REAL TRADING", "CANDIDATE FOR TESTNET")
        print(f"    recommendation={rep_d['recommendation']}")

    # ---------- 12. Scheduler jobs ----------
    with check("20 every scheduler job is callable and guarded (no crash propagation)"):
        from crypto_ai.scheduler import jobs

        job_fns = [
            jobs.health_check_job, jobs.evaluate_predictions_job,
            jobs.apply_retention_policy_job, jobs.daily_report_job,
            jobs.predict_and_paper_trade_job, jobs.weekly_retrain_job,
        ]
        for fn in job_fns:
            fn()  # guarded: must never raise
        sched = jobs.build_scheduler()
        ids = sorted(j.id for j in sched.get_jobs())
        print(f"    scheduled jobs: {ids}")
        assert len(ids) >= 6, f"expected >=6 scheduled jobs, got {ids}"

    # ---------- 14. API ----------
    with check("22 all dashboard API endpoints return 200 with populated data"):
        from fastapi.testclient import TestClient
        from crypto_ai.app.main import create_app

        app = create_app()
        with TestClient(app) as c:
            for path in ("/api/overview", "/api/market", "/api/ai", "/api/backtests",
                         "/api/paper-trading", "/api/system", "/api/report/daily"):
                r = c.get(path)
                assert r.status_code == 200, f"{path} -> {r.status_code}"
                assert r.json() is not None
            ov = c.get("/api/overview").json()
            assert ov["current_price"] is not None, "overview shows no price despite data present"
            assert ov["current_signal"] is not None, "overview shows no signal despite ticks having run"
            bt = c.get("/api/backtests").json()
            assert len(bt) == 1, f"backtests endpoint returned {len(bt)}"
        print("    all 7 endpoints OK with populated data")

    # ---------- 15. Emergency stop ----------
    with check("23 emergency stop blocks trading across the whole system"):
        from crypto_ai.risk.emergency import activate_emergency_stop, is_emergency_stop_active, reset_emergency_stop
        from crypto_ai.app.services.live_pipeline import run_prediction_and_paper_trade_tick

        try:
            activate_emergency_stop("e2e validation")
            assert is_emergency_stop_active()
            with session_scope() as s:
                r = run_prediction_and_paper_trade_tick(s, SYMBOL, TIMEFRAME)
            assert r["final_signal"] == "WAIT", f"emergency stop did not force WAIT: {r['final_signal']}"
            assert "emergency_stop_active" in r["reason_codes"]
            assert r["action"]["action"] == "NONE"
        finally:
            reset_emergency_stop()
        assert not is_emergency_stop_active(), "emergency stop did not reset"

    # ---------- 13. Retention ----------
    with check("21 retention policy deletes only what it should"):
        from crypto_ai.data.processor.retention import apply_retention_policy
        from crypto_ai.database.models.trading import PaperTrade, Prediction

        with session_scope() as s:
            before_trades = s.query(PaperTrade).count()
            before_preds = s.query(Prediction).count()
            deleted = apply_retention_policy(s, now=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc))
            after_trades = s.query(PaperTrade).count()
            after_preds = s.query(Prediction).count()
        print(f"    deleted={deleted}")
        assert after_trades == before_trades, "retention deleted trade history (must never happen)"
        assert after_preds == before_preds, "retention deleted predictions (must never happen)"

    # ---------- 16. Edge cases ----------
    with check("24 edge cases handled without crashing"):
        from crypto_ai.data.validators.ohlcv_validators import validate_ohlcv
        from crypto_ai.features.feature_pipeline import compute_features
        from crypto_ai.features.labels import compute_labels
        from crypto_ai.backtesting.engine import BacktestEngine

        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        assert validate_ohlcv(empty, "5m").n_rows == 0

        tiny = RAW_DF.head(5)
        assert len(compute_features(tiny)) == 0, "features on too-short series should be empty, not crash"

        one_bar_df = pd.DataFrame({
            "timestamp": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)],
            "close": [100.0], "signal": ["BUY"],
        })
        r_one = BacktestEngine(initial_balance=1000).run(one_bar_df)
        assert np.isfinite(r_one.final_balance)

        flat = pd.DataFrame({
            "timestamp": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=5 * i) for i in range(50)],
            "close": [100.0] * 50,
        })
        lab = compute_labels(flat, horizon_bars=5)
        assert set(lab["label"].unique()) <= {"BUY", "HOLD", "SELL"}
        assert (lab["label"] == "HOLD").all(), "flat prices must all be HOLD"

        all_hold = pd.DataFrame({
            "timestamp": flat["timestamp"], "close": flat["close"], "signal": ["HOLD"] * 50,
        })
        r_hold = BacktestEngine(initial_balance=1000).run(all_hold)
        assert r_hold.metrics["n_trades"] == 0
        assert r_hold.final_balance == 1000

    # ---------- Summary ----------
    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    width = max(len(n) for n, _, _ in RESULTS)
    n_fail = 0
    for name, status, err in RESULTS:
        mark = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{mark}] {name.ljust(width)}  {err}")
        if status == "FAIL":
            n_fail += 1
    print("-" * 78)
    print(f"  {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
