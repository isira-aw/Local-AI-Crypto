"""
Single-row inference helper (Phase 9/16).

Wraps a trained sklearn Pipeline's predict_proba so both the training
pipeline's fold evaluation and the live prediction tick
(app/services/live_pipeline.py) use the exact same "pick the highest-
probability class, use that probability as confidence" logic — one
definition, not two copies that could quietly drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_single(estimator, X: pd.DataFrame) -> tuple[str, float, dict]:
    """X: a single-row DataFrame with the model's exact training feature
    columns. Returns (predicted_class, confidence, probabilities_dict)."""
    if len(X) != 1:
        raise ValueError(f"predict_single expects exactly one row, got {len(X)}")

    proba = estimator.predict_proba(X)[0]
    classes = estimator.classes_
    best_idx = int(np.argmax(proba))
    predicted_class = str(classes[best_idx])
    confidence = float(proba[best_idx])
    probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
    return predicted_class, confidence, probabilities
