#!/usr/bin/env python3
"""
Before/after comparison for defects #3 and #4.

Both defects distorted the SAME thing — the performance numbers used to
decide whether a model gets promoted:

  #4  risk parameters were read via settings.get("risk...."), but
      settings.yaml's `risk:` block only contains `config_file: risk.yaml`.
      The user's actual risk.yaml was silently ignored in favour of
      hard-coded defaults.

  #3  fold backtests therefore ran at the BacktestEngine default of 100%
      of equity per position, while risk.yaml caps a real position at 5%
      and applies a 2% stop / 4% target.

So every Sharpe, drawdown and return produced before the fix described a
strategy the risk engine would never have permitted.

This script runs the identical walk-forward training twice on identical
data — once reproducing the old behaviour, once with the fix — and prints
the numbers side by side, per fold.

    python scripts/before_after_comparison.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DATA_STORE_DIR"] = tempfile.mkdtemp(prefix="crypto_ai_compare_")
os.environ["DATABASE_URL"] = f"sqlite:///{PROJECT_ROOT}/data_store/_compare.db"
os.environ["LLM_ENABLED"] = "false"
os.environ["MODE"] = "RESEARCH"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_BARS = 60000
SEED = 7
ALGORITHMS = ["logistic_regression", "random_forest", "gradient_boosting"]


def build_market_data(n_bars: int = N_BARS, seed: int = SEED) -> pd.DataFrame:
    """Same generator the e2e harness uses: GBM with volatility clustering."""
    rng = np.random.default_rng(seed)
    regimes = rng.choice([0.0008, 0.002, 0.005], size=n_bars, p=[0.6, 0.3, 0.1])
    returns = rng.normal(0, 1, n_bars) * regimes
    close = 40000 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0007, n_bars))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = np.abs(rng.normal(50, 20, n_bars)) + 1

    end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    start = end - dt.timedelta(minutes=5 * n_bars)
    return pd.DataFrame({
        "timestamp": [start + dt.timedelta(minutes=5 * i) for i in range(n_bars)],
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def run_config(merged, label: str, position_size_pct: float,
               stop_loss_pct: float | None, take_profit_pct: float | None) -> dict:
    """
    Run every algorithm's walk-forward folds under one sizing configuration.
    Calls the pipeline's own fold machinery so this measures the real code
    path, not a reimplementation.
    """
    from crypto_ai.config.loader import get_settings
    from crypto_ai.models.evaluation.criteria import aggregate_walk_forward, evaluate_fold
    from crypto_ai.models.training.models import build_model
    from crypto_ai.models.training.pipeline import _classification_metrics, _predict_with_confidence, _run_backtest_on_slice
    from crypto_ai.models.training.walk_forward import generate_walk_forward_folds
    from sklearn.utils.class_weight import compute_sample_weight

    settings = get_settings()
    wf = settings.get("walk_forward", {})
    label_cfg = settings.get("labeling", {})
    criteria = settings.get("models.promotion_criteria", {})
    balance = settings.get("paper_trading.starting_balance_usdt", 1000.0)
    fee = label_cfg.get("assumed_fee_pct", 0.001)
    slip = label_cfg.get("assumed_slippage_pct", 0.0005)
    min_conf = 0.55

    feature_cols = [c for c in merged.columns if c not in ("timestamp", "close", "label")]
    plan = generate_walk_forward_folds(
        n_rows=len(merged), n_folds=wf["n_folds"], min_train_bars=wf["min_train_bars"],
        validation_bars=wf["validation_bars"], embargo_bars=wf["embargo_bars"],
        final_test_bars=wf["final_test_bars"],
    )

    out = {"label": label, "n_folds": len(plan.folds), "algorithms": {}}
    for algo in ALGORITHMS:
        fold_evals = []
        per_fold = []
        for fold in plan.folds:
            X_tr = merged.iloc[fold.train_slice][feature_cols]
            y_tr = merged.iloc[fold.train_slice]["label"]
            X_va = merged.iloc[fold.val_slice][feature_cols]
            y_va = merged.iloc[fold.val_slice]["label"]

            model = build_model(algo, n_jobs=settings.resource_limits.max_workers)
            model.fit(X_tr, y_tr, clf__sample_weight=compute_sample_weight("balanced", y_tr))

            pred, conf = _predict_with_confidence(model, X_va)
            proba = model.predict_proba(X_va)
            cls_m = _classification_metrics(y_va, pred, proba, model.classes_)
            bt_m = _run_backtest_on_slice(
                merged, fold.val_slice, pred, conf, min_conf, balance, fee, slip, "5m",
                position_size_pct, stop_loss_pct, take_profit_pct,
            )
            fold_evals.append(evaluate_fold(fold.fold_index, cls_m, bt_m, criteria))
            per_fold.append({
                "fold": fold.fold_index,
                "train_bars": fold.train_end - fold.train_start,
                "val_bars": fold.val_end - fold.val_start,
                "precision": cls_m["precision_macro"],
                "return_pct": bt_m["total_return_pct"],
                "drawdown_pct": bt_m["max_drawdown_pct"],
                "sharpe": bt_m["sharpe_ratio"],
                "n_trades": bt_m["n_trades"],
                "win_rate": bt_m["win_rate_pct"],
                "passed": fold_evals[-1].passed,
                "reasons": fold_evals[-1].reasons,
            })

        summary = aggregate_walk_forward(fold_evals, criteria)
        out["algorithms"][algo] = {"folds": per_fold, "summary": summary}
        print(f"    [{label}] {algo}: {summary['n_passed']}/{summary['n_folds']} folds passed, "
              f"overall_pass={summary['overall_pass']}")
    return out


def main() -> int:
    from crypto_ai.config.loader import get_risk_config
    from crypto_ai.features.feature_pipeline import compute_features
    from crypto_ai.features.labels import compute_labels
    from crypto_ai.models.training.pipeline import merge_features_and_labels

    print("=" * 100)
    print("BEFORE / AFTER — defects #3 (100% position sizing) and #4 (risk.yaml ignored)")
    print("=" * 100)

    df = build_market_data()
    print(f"Market data: {len(df)} bars, {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    features = compute_features(df)
    labels = compute_labels(df)
    merged = merge_features_and_labels(features, labels)
    print(f"Merged feature/label rows: {len(merged)}")

    risk = get_risk_config()
    capped_size = risk["position_sizing"]["max_position_percent"] / 100.0
    stop = (risk.get("orders", {}).get("stop_loss_percent") or 0) / 100 or None
    target = (risk.get("orders", {}).get("take_profit_percent") or 0) / 100 or None
    print(f"risk.yaml -> position {capped_size:.0%} of equity, stop {stop}, target {target}\n")

    print("Running BEFORE (buggy: 100% of equity, no stop/target, risk.yaml ignored)...")
    before = run_config(merged, "BEFORE", position_size_pct=1.0, stop_loss_pct=None, take_profit_pct=None)

    print("\nRunning AFTER (fixed: sizing and stop/target read from risk.yaml)...")
    after = run_config(merged, "AFTER", position_size_pct=capped_size,
                       stop_loss_pct=stop, take_profit_pct=target)

    # ---- side by side, per fold ----
    for algo in ALGORITHMS:
        print("\n" + "=" * 100)
        print(f"PER-FOLD BREAKDOWN — {algo}")
        print("=" * 100)
        print(f"{'':6s} {'BEFORE (100% of equity)':>44s} | {'AFTER (5% cap + stop/target)':>44s}")
        print(f"{'fold':6s} {'return%':>9s} {'dd%':>8s} {'sharpe':>9s} {'trades':>7s} {'pass':>6s} | "
              f"{'return%':>9s} {'dd%':>8s} {'sharpe':>9s} {'trades':>7s} {'pass':>6s}")
        print("-" * 100)
        b_folds = before["algorithms"][algo]["folds"]
        a_folds = after["algorithms"][algo]["folds"]
        for b, a in zip(b_folds, a_folds):
            print(f"{b['fold']:<6d} {b['return_pct']:>9.2f} {b['drawdown_pct']:>8.2f} {b['sharpe']:>9.2f} "
                  f"{b['n_trades']:>7d} {str(b['passed']):>6s} | "
                  f"{a['return_pct']:>9.2f} {a['drawdown_pct']:>8.2f} {a['sharpe']:>9.2f} "
                  f"{a['n_trades']:>7d} {str(a['passed']):>6s}")

        bs, as_ = before["algorithms"][algo]["summary"], after["algorithms"][algo]["summary"]
        b_avg_dd = np.mean([f["drawdown_pct"] for f in b_folds])
        a_avg_dd = np.mean([f["drawdown_pct"] for f in a_folds])
        b_avg_ret = np.mean([f["return_pct"] for f in b_folds])
        a_avg_ret = np.mean([f["return_pct"] for f in a_folds])
        print("-" * 100)
        print(f"{'mean':6s} {b_avg_ret:>9.2f} {b_avg_dd:>8.2f} {'':>9s} {'':>7s} "
              f"{str(bs['overall_pass']):>6s} | {a_avg_ret:>9.2f} {a_avg_dd:>8.2f} {'':>9s} {'':>7s} "
              f"{str(as_['overall_pass']):>6s}")
        if not as_["overall_pass"] and as_.get("blocking_reason"):
            print(f"       AFTER blocked: {as_['blocking_reason']}")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("A 20x smaller position produces roughly 20x smaller returns AND drawdowns.")
    print("The BEFORE numbers are not 'better' — they describe an exposure the risk")
    print("engine would have refused to take. Any promotion decision made on them was")
    print("made on a strategy that could never have run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
