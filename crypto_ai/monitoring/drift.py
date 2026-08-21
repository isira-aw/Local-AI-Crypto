"""
Drift detection (Phase 15 / Section 23).

What is it?
    Two independent checks that something about the world may have
    changed since the champion model was trained:

    1. Feature drift — has the DISTRIBUTION of input features (e.g.
       volatility, volume) shifted meaningfully vs. the training
       period? Measured with the Population Stability Index (PSI), a
       standard, easy-to-explain metric.
    2. Accuracy drift — has the model's LIVE prediction accuracy
       degraded compared to what backtesting/paper trading led us to
       expect?

Why is it needed?
    Markets change. A model trained on 2023 volatility may quietly
    stop working in a very different 2025 market without ever raising
    an error — accuracy just slowly gets worse. Waiting for the next
    scheduled weekly retrain to notice this is too slow; this is the
    "check more often, flag for review" half of Section 23.

Important: drift being flagged does NOT trigger an automatic retrain
or promotion. It only raises a flag that a human/the training pipeline
should look at — see run_training_pipeline() + maybe_promote_champion()
for the actual (still-gated) promotion decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    PSI compares two distributions by binning the EXPECTED (training)
    series into `buckets` equal-frequency bins, then measuring how
    differently the ACTUAL (recent) series falls into those same bins.

    Rule of thumb: <0.10 no meaningful shift, 0.10-0.25 moderate shift
    (watch it), >0.25 significant shift (investigate).
    """
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) < buckets or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(expected.quantile(quantiles).values)
    if len(breakpoints) < 3:
        return 0.0  # not enough variation to bucket meaningfully

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = np.clip(expected_counts / max(1, expected_counts.sum()), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(1, actual_counts.sum()), 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


@dataclass
class DriftReport:
    feature_psi: dict = field(default_factory=dict)
    max_psi: float = 0.0
    feature_drift_flagged: bool = False
    accuracy_drift_flagged: bool = False
    recent_accuracy_pct: float | None = None
    expected_accuracy_pct: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def flagged_for_review(self) -> bool:
        return self.feature_drift_flagged or self.accuracy_drift_flagged


def detect_feature_drift(
    training_df: pd.DataFrame, recent_df: pd.DataFrame, feature_cols: list[str], threshold: float = PSI_SIGNIFICANT_THRESHOLD
) -> tuple[dict, bool]:
    psi_by_feature = {
        col: population_stability_index(training_df[col], recent_df[col])
        for col in feature_cols if col in training_df.columns and col in recent_df.columns
    }
    flagged = any(v > threshold for v in psi_by_feature.values())
    return psi_by_feature, flagged


def detect_accuracy_drift(
    recent_accuracy_pct: float | None, expected_accuracy_pct: float | None, degradation_threshold_pct: float = 10.0
) -> bool:
    """
    Flags drift if live accuracy has fallen more than
    `degradation_threshold_pct` percentage points below what
    backtesting/paper trading led us to expect.
    """
    if recent_accuracy_pct is None or expected_accuracy_pct is None:
        return False
    return recent_accuracy_pct < (expected_accuracy_pct - degradation_threshold_pct)


def run_drift_check(
    training_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    feature_cols: list[str],
    recent_accuracy_pct: float | None = None,
    expected_accuracy_pct: float | None = None,
) -> DriftReport:
    psi_by_feature, feature_flagged = detect_feature_drift(training_df, recent_df, feature_cols)
    accuracy_flagged = detect_accuracy_drift(recent_accuracy_pct, expected_accuracy_pct)

    reasons = []
    if feature_flagged:
        worst = max(psi_by_feature, key=psi_by_feature.get) if psi_by_feature else None
        reasons.append(f"feature distribution shifted (worst: {worst}, PSI={psi_by_feature.get(worst, 0):.3f})")
    if accuracy_flagged:
        reasons.append(
            f"live accuracy {recent_accuracy_pct:.1f}% is more than 10pp below expected {expected_accuracy_pct:.1f}%"
        )

    return DriftReport(
        feature_psi=psi_by_feature,
        max_psi=max(psi_by_feature.values()) if psi_by_feature else 0.0,
        feature_drift_flagged=feature_flagged,
        accuracy_drift_flagged=accuracy_flagged,
        recent_accuracy_pct=recent_accuracy_pct,
        expected_accuracy_pct=expected_accuracy_pct,
        reasons=reasons,
    )
