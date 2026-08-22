"""
Automatic training pipeline (Phase 9 / Section 14).

    new data -> validate -> generate features -> create labels
        -> train candidate models (walk-forward, ALL folds)
        -> validate -> test (final held-out set)
        -> compare with baseline -> save model

What is it?
    Ties together features, labels, walk-forward splitting, candidate
    models, the backtester, and the model registry into one function
    a scheduler (or `run.py train`) can call.

Why is it needed?
    So training is reproducible and leakage-safe by construction,
    without the user having to remember to do any of this by hand.

How does it work?
    For each candidate algorithm:
      1. Run every walk-forward fold: train on the fold's training
         window, predict on its validation window (after the embargo
         gap), score both classification metrics and a realistic
         backtest of the resulting signals.
      2. A fold "passes" per models/evaluation/criteria.py. The
         algorithm overall passes only if enough folds pass (Section
         15: "ALL walk-forward folds", governed by
         min_folds_passed_pct).
      3. If it passes, retrain once more on ALL data before the final
         test window, evaluate on the final held-out test set (never
         seen during folds), and register it as a CANDIDATE.
         Otherwise register it as REJECTED (kept for the record, per
         Section 24, not deleted).
    Finally, compare every CANDIDATE against the current CHAMPION (if
    any) on final-test performance and promote only if it's better —
    never automatically, and never based on accuracy alone.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy.orm import Session

from crypto_ai.backtesting.engine import BacktestEngine
from crypto_ai.config.loader import get_risk_config, get_settings
from crypto_ai.database.repositories.events_repo import log_event
from crypto_ai.features.feature_pipeline import FEATURE_VERSION, feature_columns
from crypto_ai.features.labels import LABEL_VERSION
from crypto_ai.models.evaluation.criteria import aggregate_walk_forward, evaluate_fold
from crypto_ai.models.evaluation.multiple_testing import (
    DEFAULT_MIN_DSR, collect_trial_sharpes, count_trials, deflated_sharpe,
)
from crypto_ai.models.registry import registry
from crypto_ai.models.training.models import CANDIDATE_ALGORITHMS, build_model
from crypto_ai.models.training.walk_forward import assert_no_leakage, generate_walk_forward_folds
from crypto_ai.strategies.ml_strategy import STRATEGY_VERSION, predictions_to_signals

logger = logging.getLogger(__name__)


def merge_features_and_labels(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join on timestamp. feature_df and label_df have different
    front/back-truncated rows (warm-up vs. label horizon), so this is
    where they're aligned into one chronologically-sorted frame.
    """
    merged = pd.merge(feature_df, label_df[["timestamp", "close", "label"]], on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def _predict_with_confidence(model, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    proba = model.predict_proba(X)
    classes = model.classes_
    best_idx = np.argmax(proba, axis=1)
    predicted_class = pd.Series([classes[i] for i in best_idx], index=X.index)
    confidence = pd.Series(proba[np.arange(len(proba)), best_idx], index=X.index)
    return predicted_class, confidence


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, proba: np.ndarray, classes: np.ndarray) -> dict:
    labels = ["BUY", "HOLD", "SELL"]
    metrics = {
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    try:
        if len(set(y_true)) > 1 and proba.shape[1] == len(classes):
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=list(classes))
        else:
            metrics["roc_auc_ovr"] = None
    except ValueError:
        metrics["roc_auc_ovr"] = None
    return metrics


def _run_backtest_on_slice(
    merged: pd.DataFrame,
    idx_slice: slice,
    predicted_class: pd.Series,
    confidence: pd.Series,
    min_confidence_to_trade: float,
    initial_balance: float,
    fee_pct: float,
    slippage_pct: float,
    timeframe: str,
    position_size_pct: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> dict:
    signals = predictions_to_signals(predicted_class, confidence, min_confidence_to_trade)
    bt_df = pd.DataFrame(
        {
            "timestamp": merged["timestamp"].iloc[idx_slice].values,
            "close": merged["close"].iloc[idx_slice].values,
            "signal": signals.values,
        }
    )
    engine = BacktestEngine(
        initial_balance=initial_balance, fee_pct=fee_pct, slippage_pct=slippage_pct, timeframe=timeframe,
        # Fold backtests MUST use the same position sizing and stop/target
        # rules the risk engine would actually enforce live. Backtesting at
        # 100% of equity while risk.yaml caps a real position at 5% would
        # make promotion criteria (Sharpe, drawdown, return) reflect a
        # strategy the system is never allowed to run.
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    report = engine.run(bt_df)
    return report.metrics


def train_and_evaluate_algorithm(
    session: Session,
    algorithm: str,
    merged: pd.DataFrame,
    symbol: str,
    timeframe: str,
    model_name: str,
    walk_forward_cfg: dict | None = None,
    labeling_cfg: dict | None = None,
    promotion_criteria: dict | None = None,
) -> dict:
    settings = get_settings()
    wf_cfg = walk_forward_cfg or settings.get("walk_forward", {})
    label_cfg = labeling_cfg or settings.get("labeling", {})
    criteria = promotion_criteria or settings.get("models.promotion_criteria", {})
    # Risk parameters live in risk.yaml, NOT settings.yaml (settings.yaml's
    # `risk:` block only points at the file). Reading them via
    # settings.get("risk...") silently returns the default and ignores
    # whatever the user actually configured.
    risk_cfg = get_risk_config()
    min_confidence = risk_cfg.get("position_sizing", {}).get("min_confidence_to_trade", 0.55)
    position_size_pct = risk_cfg.get("position_sizing", {}).get("max_position_percent", 5.0) / 100.0
    stop_loss_pct = (risk_cfg.get("orders", {}).get("stop_loss_percent") or 0) / 100 or None
    take_profit_pct = (risk_cfg.get("orders", {}).get("take_profit_percent") or 0) / 100 or None
    paper_balance = settings.get("paper_trading.starting_balance_usdt", 1000.0)
    fee_pct = label_cfg.get("assumed_fee_pct", 0.001)
    slippage_pct = label_cfg.get("assumed_slippage_pct", 0.0005)

    feature_cols = [c for c in merged.columns if c not in ("timestamp", "close", "label")]
    n_rows = len(merged)

    plan = generate_walk_forward_folds(
        n_rows=n_rows,
        n_folds=wf_cfg.get("n_folds", 5),
        min_train_bars=wf_cfg.get("min_train_bars", 2000),
        validation_bars=wf_cfg.get("validation_bars", 500),
        embargo_bars=wf_cfg.get("embargo_bars", 12),
        final_test_bars=wf_cfg.get("final_test_bars", 500),
    )
    assert_no_leakage(plan, embargo_bars=wf_cfg.get("embargo_bars", 12), label_horizon_bars=label_cfg.get("horizon_bars", 12))

    if not plan.folds:
        return {
            "algorithm": algorithm,
            "overall_pass": False,
            "reason": "not enough data for even one walk-forward fold",
            "walk_forward": {"n_folds": 0, "n_passed": 0, "folds": []},
        }

    fold_evaluations = []
    for fold in plan.folds:
        # NOTE: fold slices are positional (0-based, exclusive end), so
        # this MUST use .iloc — .loc on an integer RangeIndex slice is
        # inclusive of the stop value and would silently pull in one
        # extra (leaked) row per fold.
        X_train = merged.iloc[fold.train_slice][feature_cols]
        y_train = merged.iloc[fold.train_slice]["label"]
        X_val = merged.iloc[fold.val_slice][feature_cols]
        y_val = merged.iloc[fold.val_slice]["label"]

        model = build_model(algorithm, n_jobs=settings.resource_limits.max_workers)
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, clf__sample_weight=weights)

        predicted_class, confidence = _predict_with_confidence(model, X_val)
        proba = model.predict_proba(X_val)
        cls_metrics = _classification_metrics(y_val, predicted_class, proba, model.classes_)

        bt_metrics = _run_backtest_on_slice(
            merged, fold.val_slice, predicted_class, confidence, min_confidence,
            paper_balance, fee_pct, slippage_pct, timeframe,
            position_size_pct, stop_loss_pct, take_profit_pct,
        )

        fold_eval = evaluate_fold(fold.fold_index, cls_metrics, bt_metrics, criteria)
        fold_evaluations.append(fold_eval)

    wf_summary = aggregate_walk_forward(fold_evaluations, criteria)

    result = {
        "algorithm": algorithm,
        "walk_forward": wf_summary,
        "overall_pass": wf_summary["overall_pass"],
    }

    if not wf_summary["overall_pass"]:
        reject_msg = (
            wf_summary["blocking_reason"]
            or f"{algorithm} failed walk-forward criteria ({wf_summary['n_passed']}/{wf_summary['n_folds']} folds passed)"
        )
        result["reason"] = reject_msg
        log_event(
            session, component="training", event="algorithm_rejected", severity="INFO",
            message=reject_msg,
            context={"algorithm": algorithm, "model_name": model_name},
        )
        # Still register it as REJECTED so the attempt is on record
        # (Section 24: rejected models are tracked, not silently discarded).
        final_model = build_model(algorithm, n_jobs=settings.resource_limits.max_workers)
        train_slice = slice(0, plan.final_test_start)
        weights = compute_sample_weight("balanced", merged.iloc[train_slice]["label"])
        final_model.fit(merged.iloc[train_slice][feature_cols], merged.iloc[train_slice]["label"], clf__sample_weight=weights)
        version = registry.register_model_version(
            session, model_name, symbol, timeframe, algorithm,
            FEATURE_VERSION, STRATEGY_VERSION, final_model,
            hyperparameters={}, metrics={}, walk_forward_results=wf_summary,
            status=registry.STATUS_REJECTED,
        )
        result["version_label"] = version.version_label
        return result

    # Passed walk-forward: retrain on everything up to the final test
    # set, then evaluate ONCE on the never-touched final test window.
    train_slice = slice(0, plan.final_test_start)
    final_model = build_model(algorithm, n_jobs=settings.resource_limits.max_workers)
    weights = compute_sample_weight("balanced", merged.iloc[train_slice]["label"])
    final_model.fit(merged.iloc[train_slice][feature_cols], merged.iloc[train_slice]["label"], clf__sample_weight=weights)

    X_test = merged.iloc[plan.final_test_slice][feature_cols]
    y_test = merged.iloc[plan.final_test_slice]["label"]
    predicted_class, confidence = _predict_with_confidence(final_model, X_test)
    proba = final_model.predict_proba(X_test)
    final_cls_metrics = _classification_metrics(y_test, predicted_class, proba, final_model.classes_)
    final_bt_metrics = _run_backtest_on_slice(
        merged, plan.final_test_slice, predicted_class, confidence, min_confidence,
        paper_balance, fee_pct, slippage_pct, timeframe,
        position_size_pct, stop_loss_pct, take_profit_pct,
    )

    combined_metrics = {"classification": final_cls_metrics, "backtest": final_bt_metrics}

    version = registry.register_model_version(
        session, model_name, symbol, timeframe, algorithm,
        FEATURE_VERSION, STRATEGY_VERSION, final_model,
        hyperparameters={}, metrics=combined_metrics, walk_forward_results=wf_summary,
        status=registry.STATUS_CANDIDATE,
    )
    log_event(
        session, component="training", event="candidate_registered", severity="INFO",
        message=f"{algorithm} passed walk-forward ({wf_summary['n_passed']}/{wf_summary['n_folds']} folds) -> {version.version_label}",
        context={"algorithm": algorithm, "model_name": model_name, "version": version.version_label},
    )

    result["version_label"] = version.version_label
    result["final_test_metrics"] = combined_metrics
    return result


def run_training_pipeline(
    session: Session,
    symbol: str,
    timeframe: str,
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    model_name: str = "btc_direction_classifier",
    algorithms: list[str] | None = None,
    walk_forward_cfg: dict | None = None,
    labeling_cfg: dict | None = None,
    promotion_criteria: dict | None = None,
    max_training_bars: int | None = None,
) -> dict:
    settings = get_settings()
    algorithms = algorithms or settings.get("models.candidates", list(CANDIDATE_ALGORITHMS))
    merged = merge_features_and_labels(feature_df, label_df)

    # DATA_WINDOW (Section 53): cap how much history a single training run
    # uses. Without this, training on the default 2 years of 5m candles
    # (~210k bars) takes hours on a normal CPU-only laptop, which makes
    # iteration painful and the weekly retrain job unreliable. Keeping the
    # MOST RECENT window is also usually the right call for a market that
    # changes regime — older data can be less representative.
    max_bars = max_training_bars if max_training_bars is not None else settings.get(
        "resource_limits.max_training_bars"
    )
    if max_bars and len(merged) > max_bars:
        logger.info(
            "Limiting training to the most recent %s of %s bars (resource_limits.max_training_bars).",
            max_bars, len(merged),
        )
        merged = merged.iloc[-max_bars:].reset_index(drop=True)

    results = []
    for algo in algorithms:
        try:
            result = train_and_evaluate_algorithm(
                session, algo, merged, symbol, timeframe, model_name,
                walk_forward_cfg, labeling_cfg, promotion_criteria,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed for %s", algo)
            log_event(
                session, component="training", event="training_failed", severity="ERROR",
                message=str(exc), context={"algorithm": algo, "model_name": model_name},
            )
            results.append({"algorithm": algo, "overall_pass": False, "error": str(exc)})

    session.commit()

    candidates = [r for r in results if r.get("overall_pass")]

    # Multiple-testing correction (Section 15). Promoting the best of N
    # variants without deflating for N is how a worthless model gets
    # promoted on noise. This is applied to the promotion decision itself,
    # not merely reported.
    prior_variants = _count_prior_variants(session, model_name, exclude=len(results))
    n_trials = count_trials(results, prior_variants=prior_variants)
    trial_sharpes = collect_trial_sharpes(results)

    promotion = maybe_promote_champion(
        session, model_name, candidates,
        n_trials=n_trials, trial_sharpes=trial_sharpes, timeframe=timeframe,
    )
    session.commit()

    return {
        "model_name": model_name,
        "results": results,
        "promotion": promotion,
        "multiple_testing": {
            "n_trials": n_trials,
            "prior_variants": prior_variants,
            "trial_sharpes_annual": [round(x, 4) for x in trial_sharpes],
        },
    }


def _count_prior_variants(session: Session, model_name: str, exclude: int = 0) -> int:
    """
    Variants previously registered for this model. Counted so the
    correction reflects the whole multiple-testing surface across the
    project's life, not just the current run.
    """
    from crypto_ai.database.models.ml import MLModel, ModelVersion
    from sqlalchemy import select

    model = session.execute(select(MLModel).where(MLModel.name == model_name)).scalar_one_or_none()
    if model is None:
        return 0
    total = session.query(ModelVersion).filter(ModelVersion.model_id == model.id).count()
    return max(0, total - exclude)


def maybe_promote_champion(
    session: Session, model_name: str, candidate_results: list[dict],
    n_trials: int = 1, trial_sharpes: list[float] | None = None, timeframe: str = "5m",
) -> dict:
    """
    Compare CANDIDATEs against the current CHAMPION on final-test Sharpe
    ratio and promote only if a candidate is strictly better (Section
    24: "only promoted if it improves performance on unseen data").
    An empty candidate list, or no improvement, leaves the champion
    (if any) untouched — never an automatic promotion on accuracy alone.
    """
    if not candidate_results:
        return {"promoted": False, "reason": "no candidates passed walk-forward validation"}

    def sharpe_of(r: dict) -> float:
        return r.get("final_test_metrics", {}).get("backtest", {}).get("sharpe_ratio", -999)

    best = max(candidate_results, key=sharpe_of)
    best_sharpe = sharpe_of(best)

    champion = registry.get_champion(session, model_name)
    if champion is not None:
        champion_sharpe = champion.metrics.get("backtest", {}).get("sharpe_ratio", -999)
        if best_sharpe <= champion_sharpe:
            return {
                "promoted": False,
                "reason": f"best candidate sharpe {best_sharpe:.3f} did not beat champion {champion_sharpe:.3f}",
            }

    # --- multiple-testing gate -------------------------------------
    criteria = get_settings().get("models.promotion_criteria", {})
    dsr_threshold = criteria.get("min_deflated_sharpe_probability", DEFAULT_MIN_DSR)
    n_observations = (
        best.get("final_test_metrics", {}).get("backtest", {}).get("n_observations")
        or get_settings().get("walk_forward.final_test_bars", 4000)
    )
    mt = deflated_sharpe(
        observed_sharpe_annual=best_sharpe,
        n_trials=n_trials,
        n_observations=n_observations,
        timeframe=timeframe,
        trial_sharpes_annual=trial_sharpes or [],
        threshold=dsr_threshold,
    )
    if not mt.passed:
        log_event(
            session, component="training", event="promotion_blocked_by_multiple_testing",
            severity="INFO", message=mt.reason,
            context={"model_name": model_name, **mt.to_dict()},
        )
        return {
            "promoted": False,
            "reason": mt.reason,
            "multiple_testing": mt.to_dict(),
        }

    version = (
        session.query(registry.ModelVersion)
        .filter(registry.ModelVersion.version_label == best["version_label"])
        .join(registry.MLModel)
        .filter(registry.MLModel.name == model_name)
        .one()
    )
    version.metrics = {**(version.metrics or {}), "multiple_testing": mt.to_dict()}
    registry.promote_to_champion(session, version)
    log_event(
        session, component="training", event="champion_promoted", severity="INFO",
        message=f"{version.version_label} promoted to champion (sharpe={best_sharpe:.3f})",
        context={"model_name": model_name, "version": version.version_label},
    )
    return {
        "promoted": True, "version_label": version.version_label,
        "sharpe_ratio": best_sharpe, "multiple_testing": mt.to_dict(),
    }
