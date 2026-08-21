"""
Promotion criteria (Section 15): decide whether a fold — and a model
overall — is good enough to become a candidate for production.

Deliberately NOT based on accuracy alone. A model must show a plausible
edge (precision), acceptable risk (drawdown), a positive risk-adjusted
return (Sharpe), and enough trades to trust the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FoldEvaluation:
    fold_index: int
    passed: bool
    reasons: list[str] = field(default_factory=list)
    classification_metrics: dict = field(default_factory=dict)
    backtest_metrics: dict = field(default_factory=dict)


def evaluate_fold(
    fold_index: int,
    classification_metrics: dict,
    backtest_metrics: dict,
    criteria: dict,
) -> FoldEvaluation:
    reasons = []

    precision = classification_metrics.get("precision_macro", 0.0)
    min_precision = criteria.get("min_precision", 0.52)
    if precision < min_precision:
        reasons.append(f"precision {precision:.3f} < required {min_precision:.3f}")

    drawdown_pct = backtest_metrics.get("max_drawdown_pct", 100.0)
    max_drawdown_allowed = criteria.get("max_drawdown_pct", 0.20) * 100
    if drawdown_pct > max_drawdown_allowed:
        reasons.append(f"drawdown {drawdown_pct:.1f}% > allowed {max_drawdown_allowed:.1f}%")

    sharpe = backtest_metrics.get("sharpe_ratio", -999)
    min_sharpe = criteria.get("min_sharpe", 0.3)
    if sharpe < min_sharpe:
        reasons.append(f"sharpe {sharpe:.2f} < required {min_sharpe:.2f}")

    n_trades = backtest_metrics.get("n_trades", 0)
    min_trades = criteria.get("min_trades_per_fold", 20)
    if n_trades < min_trades:
        reasons.append(f"only {n_trades} trades, need >= {min_trades}")

    return FoldEvaluation(
        fold_index=fold_index,
        passed=not reasons,
        reasons=reasons,
        classification_metrics=classification_metrics,
        backtest_metrics=backtest_metrics,
    )


# A model validated on only one or two periods has not actually been
# shown to be robust across market regimes — which is the entire point of
# walk-forward validation (Section 13/15). If the available history only
# supports a single fold, "100% of folds passed" is meaningless, so the
# result must NOT count as an overall pass no matter how good it looks.
MIN_FOLDS_FOR_A_MEANINGFUL_PASS = 3


def aggregate_walk_forward(fold_evaluations: list[FoldEvaluation], criteria: dict) -> dict:
    n_folds = len(fold_evaluations)
    n_passed = sum(1 for f in fold_evaluations if f.passed)
    pct_passed = n_passed / n_folds if n_folds else 0.0
    min_folds_passed_pct = criteria.get("min_folds_passed_pct", 0.8)
    min_folds_required = criteria.get("min_folds_required", MIN_FOLDS_FOR_A_MEANINGFUL_PASS)

    enough_folds = n_folds >= min_folds_required
    overall_pass = enough_folds and pct_passed >= min_folds_passed_pct

    blocking_reason = None
    if not enough_folds:
        blocking_reason = (
            f"only {n_folds} walk-forward fold(s) available, need >= {min_folds_required} "
            f"for a meaningful robustness check — collect more history "
            f"(or lower walk_forward.min_train_bars / validation_bars) before trusting this model"
        )

    return {
        "n_folds": n_folds,
        "n_passed": n_passed,
        "pct_passed": round(pct_passed, 4),
        "min_required_pct": min_folds_passed_pct,
        "min_folds_required": min_folds_required,
        "enough_folds": enough_folds,
        "blocking_reason": blocking_reason,
        "overall_pass": overall_pass,
        "folds": [
            {
                "fold_index": f.fold_index,
                "passed": f.passed,
                "reasons": f.reasons,
                "classification_metrics": f.classification_metrics,
                "backtest_metrics": f.backtest_metrics,
            }
            for f in fold_evaluations
        ],
    }
