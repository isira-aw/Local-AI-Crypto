"""
Candidate model factory (Section 8: "start with simple models before
deep learning"). Each returns an sklearn Pipeline (scaler + classifier)
so features of very different scales (price distances vs. RSI 0-100)
don't dominate the model just because of their raw magnitude.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATE_ALGORITHMS = ("logistic_regression", "random_forest", "gradient_boosting")


def build_model(algorithm: str, random_state: int = 42, n_jobs: int = 1):
    if algorithm == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif algorithm == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=random_state, n_jobs=n_jobs,
        )
    elif algorithm == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {CANDIDATE_ALGORITHMS}. "
            f"XGBoost/LightGBM can be added later if a simple model doesn't cut it "
            f"(Section 8) — see docs/MODEL_TRAINING.md."
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
