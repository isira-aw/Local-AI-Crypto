"""
Candidate model factory (Section 8: "start with simple models before
deep learning"). Each returns an sklearn Pipeline (scaler + classifier)
so features of very different scales (price distances vs. RSI 0-100)
don't dominate the model just because of their raw magnitude.

Why `gradient_boosting` uses the histogram-based implementation
-------------------------------------------------------------
sklearn ships two gradient-boosting classifiers. The classic
`GradientBoostingClassifier` evaluates exact splits over every sample,
which is O(n log n) per tree and becomes painfully slow on the tens of
thousands of bars this project trains on. `HistGradientBoostingClassifier`
bins each feature once up front and is dramatically faster on the same
data — measured on this project's own shape (20k rows x 30 features,
4-core CPU):

    GradientBoostingClassifier      84.4s per fit
    HistGradientBoostingClassifier  18.4s per fit     (~4.6x faster)

Walk-forward training does 6 fits per algorithm (5 folds + 1 final), and
the gap widens as history grows, so the exact implementation pushed a
full training run past an hour on a normal laptop — which conflicts with
Section 9's CPU-first requirement and made the weekly retrain job
impractical. sklearn's own guidance is to prefer the histogram-based
estimator above roughly 10,000 samples.

The classic implementation is still available as `gradient_boosting_exact`
for small datasets or for comparison.
"""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATE_ALGORITHMS = ("logistic_regression", "random_forest", "gradient_boosting")

ALL_ALGORITHMS = CANDIDATE_ALGORITHMS + ("gradient_boosting_exact",)


def build_model(algorithm: str, random_state: int = 42, n_jobs: int = 1):
    if algorithm == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif algorithm == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=random_state, n_jobs=n_jobs,
        )
    elif algorithm == "gradient_boosting":
        # Histogram-based: see the module docstring for the timing that
        # motivated this choice.
        clf = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, random_state=random_state,
        )
    elif algorithm == "gradient_boosting_exact":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {ALL_ALGORITHMS}. "
            f"XGBoost/LightGBM can be added later if a simple model doesn't cut it "
            f"(Section 8) — see docs/MODEL_TRAINING.md."
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
