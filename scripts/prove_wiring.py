#!/usr/bin/env python3
"""
Proof that the previously-dead code paths actually FIRE.

"The function exists" is not evidence. For each of the four paths this
script invokes the real production entry point and then counts the rows it
should have written / the events it should have logged, printing before and
after. If a path is silently not running, the delta is zero and the check
fails.

    python scripts/prove_wiring.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DATA_STORE_DIR"] = tempfile.mkdtemp(prefix="crypto_ai_wiring_")
os.environ["DATABASE_URL"] = f"sqlite:///{PROJECT_ROOT}/data_store/_wiring.db"
os.environ["LLM_ENABLED"] = "false"
os.environ["MODE"] = "RESEARCH"

import numpy as np  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str):
    RESULTS.append((name, ok, detail))
    print(f"  [{'FIRES' if ok else 'DEAD '}] {name}: {detail}")


def fresh_db():
    db = PROJECT_ROOT / "data_store" / "_wiring.db"
    if db.exists():
        db.unlink()
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")


def seed(session, n=1500):
    from crypto_ai.database.repositories.market_data_repo import upsert_candles

    rng = np.random.default_rng(11)
    close = 40000 * np.cumprod(1 + rng.normal(0, 0.002, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0007, n))
    end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    start = end - dt.timedelta(minutes=5 * n)
    rows = [{
        "timestamp": start + dt.timedelta(minutes=5 * i),
        "open": float(open_[i]),
        "high": float(max(open_[i], close[i]) * (1 + spread[i])),
        "low": float(min(open_[i], close[i]) * (1 - spread[i])),
        "close": float(close[i]), "volume": float(abs(rng.normal(50, 10))),
    } for i in range(n)]
    upsert_candles(session, "BTC/USDT", "5m", rows)
    session.commit()


def main() -> int:
    print("=" * 78)
    print("WIRING PROOF — do the previously-dead paths actually run?")
    print("=" * 78)
    fresh_db()

    from crypto_ai.database.base import session_scope
    from crypto_ai.database.models.monitoring import PerformanceMetric, SystemEvent
    from crypto_ai.database.models.market import MarketData
    from crypto_ai.database.models.trading import Signal

    with session_scope() as s:
        seed(s)

    # ---------------------------------------------------------------
    # 1. signals table — written by the live tick
    # ---------------------------------------------------------------
    print("\n1. signals table (was: build_signal() had zero callers)")
    from crypto_ai.app.services.live_pipeline import run_prediction_and_paper_trade_tick
    from crypto_ai.models.registry import registry
    from crypto_ai.features.feature_pipeline import compute_features, FEATURE_VERSION
    from crypto_ai.features.labels import compute_labels
    from crypto_ai.models.training.pipeline import run_training_pipeline
    from crypto_ai.database.repositories.market_data_repo import load_candles_df

    with session_scope() as s:
        df = load_candles_df(s, "BTC/USDT", "5m")
        run_training_pipeline(
            s, "BTC/USDT", "5m", compute_features(df), compute_labels(df),
            model_name="btc_direction_classifier", algorithms=["logistic_regression"],
            walk_forward_cfg={"n_folds": 3, "min_train_bars": 400, "validation_bars": 200,
                              "embargo_bars": 12, "final_test_bars": 200},
            promotion_criteria={"min_folds_passed_pct": 0.0, "min_folds_required": 1,
                                "min_precision": 0.0, "max_drawdown_pct": 9.99,
                                "min_sharpe": -99, "min_trades_per_fold": 0},
        )
    with session_scope() as s:
        if registry.get_champion(s, "btc_direction_classifier") is None:
            from crypto_ai.database.models.ml import ModelVersion
            registry.promote_to_champion(s, s.query(ModelVersion).first())
            s.commit()

    with session_scope() as s:
        before = s.query(Signal).count()
    for _ in range(3):
        with session_scope() as s:
            run_prediction_and_paper_trade_tick(s, "BTC/USDT", "5m")
    with session_scope() as s:
        after = s.query(Signal).count()
        sample = s.query(Signal).order_by(Signal.id.desc()).first()
    record("signals written per tick", after - before == 3,
           f"{before} -> {after} rows (+{after-before}); latest="
           f"{sample.signal if sample else None} reasons={sample.reason_codes if sample else None}")

    # ---------------------------------------------------------------
    # 2. drift detection — invoked by the scheduler job
    # ---------------------------------------------------------------
    print("\n2. drift detection (was: run_drift_check() had zero callers)")
    from crypto_ai.scheduler.jobs import drift_check_job

    with session_scope() as s:
        before = s.query(PerformanceMetric).filter_by(scope="drift").count()
    drift_check_job()  # the exact callable the scheduler registers
    with session_scope() as s:
        after = s.query(PerformanceMetric).filter_by(scope="drift").count()
        latest = (s.query(PerformanceMetric).filter_by(scope="drift")
                  .order_by(PerformanceMetric.timestamp.desc()).first())
    record("drift check writes a result row", after > before,
           f"{before} -> {after} rows; max_psi="
           f"{(latest.metrics or {}).get('max_psi') if latest else None} "
           f"flagged={(latest.metrics or {}).get('flagged_for_review') if latest else None}")

    # ---------------------------------------------------------------
    # 3. weekly retrain — invoked by the scheduler job
    # ---------------------------------------------------------------
    print("\n3. weekly retrain (was: a no-op that logged a TODO)")
    from crypto_ai.scheduler.jobs import weekly_retrain_job
    from crypto_ai.database.models.ml import ModelVersion

    # The retrain job reads walk-forward settings from settings.yaml, whose
    # defaults need ~44k bars. With the 1500 bars seeded here it would
    # correctly decline to train (0 folds) — which would make this proof
    # vacuous. Shrink the window so the job does REAL work we can observe.
    from crypto_ai.config.loader import get_settings

    get_settings().raw["walk_forward"] = {
        "n_folds": 3, "min_train_bars": 400, "validation_bars": 200,
        "embargo_bars": 12, "final_test_bars": 200,
    }
    get_settings().raw["models"]["promotion_criteria"] = {
        "min_folds_required": 3, "min_folds_passed_pct": 0.8, "min_precision": 0.0,
        "max_drawdown_pct": 9.99, "min_sharpe": -99, "min_trades_per_fold": 0,
    }

    with session_scope() as s:
        before_versions = s.query(ModelVersion).count()
        before_events = s.query(SystemEvent).filter_by(event="training_completed").count()
    weekly_retrain_job()
    with session_scope() as s:
        after_versions = s.query(ModelVersion).count()
        after_events = s.query(SystemEvent).filter_by(event="training_completed").count()
    record("weekly retrain trains and logs", after_versions > before_versions and after_events > before_events,
           f"model_versions {before_versions} -> {after_versions}, "
           f"training_completed events {before_events} -> {after_events}")

    # ---------------------------------------------------------------
    # 4. live WebSocket collector — end to end against a fake server
    # ---------------------------------------------------------------
    print("\n4. live WebSocket collector (was: module did not exist)")
    from crypto_ai.data.collector.live_collector import LiveCollector

    closed_bars = []
    base = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    for i in range(5):
        ts = int((base + dt.timedelta(minutes=5 * i)).timestamp() * 1000)
        # one in-progress update then the closed candle, as Binance streams it
        closed_bars.append(json.dumps({"k": {"t": ts, "x": False, "o": "1", "h": "2", "l": "0.5", "c": "1.4", "v": "9"}}))
        closed_bars.append(json.dumps({"k": {"t": ts, "x": True, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"}}))

    class FakeWS:
        """Stands in for a websockets connection."""
        def __init__(self, messages): self.messages = messages
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def __aiter__(self): return self._gen()
        async def _gen(self):
            for m in self.messages:
                yield m

    # max_reconnects=0 -> consume the stream once and stop. Without a bound
    # the collector reconnects forever by design (it is a 24/7 service).
    collector = LiveCollector("BTC/USDT", "5m", connect_factory=lambda: FakeWS(closed_bars),
                              backfill_on_reconnect=False, initial_backoff_seconds=0.01,
                              max_reconnects=0)

    async def drive():
        await asyncio.wait_for(collector.run(), timeout=30)

    with session_scope() as s:
        before_bars = s.query(MarketData).count()
        before_conn = s.query(SystemEvent).filter_by(event="connected").count()
    asyncio.run(drive())
    with session_scope() as s:
        after_bars = s.query(MarketData).count()
        after_conn = s.query(SystemEvent).filter_by(event="connected").count()
    record("WS collector writes CLOSED candles only",
           collector.candles_written == 5 and after_bars - before_bars == 5,
           f"handled 10 messages (5 in-progress + 5 closed) -> wrote "
           f"{collector.candles_written} candles; market_data {before_bars} -> {after_bars}; "
           f"connect events {before_conn} -> {after_conn}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    n_dead = sum(1 for _, ok, _ in RESULTS if not ok)
    for name, ok, _ in RESULTS:
        print(f"  {'FIRES' if ok else 'DEAD '}  {name}")
    print("-" * 78)
    print(f"  {len(RESULTS) - n_dead}/{len(RESULTS)} paths confirmed firing")
    print("=" * 78)
    return 1 if n_dead else 0


if __name__ == "__main__":
    sys.exit(main())
