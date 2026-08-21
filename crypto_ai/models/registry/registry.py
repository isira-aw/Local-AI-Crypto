"""
Model registry (Section 24 / Phase 9).

What is it?
    Tracks every trained model artifact with a version label
    (model_001, model_002, ...) and a lifecycle status:
        TRAINING -> TESTING -> CANDIDATE -> CHAMPION
                                          -> REJECTED
        CHAMPION -> RETIRED (when a new candidate is promoted)

Why is it needed?
    "Never overwrite a working model without preserving the previous
    version" (Section 14) and "never automatically replace the
    champion without validation" (Section 24). This module is the only
    place that is allowed to change a model's status.

How does it work?
    Trained estimators are saved to disk with joblib under
    data_store/models/<model_name>/<version_label>/model.joblib —
    never deleted, only superseded. The database row is the source of
    truth for status; the file on disk is the source of truth for the
    actual weights.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.ml import MLModel, ModelVersion

STATUS_TRAINING = "TRAINING"
STATUS_TESTING = "TESTING"
STATUS_CANDIDATE = "CANDIDATE"
STATUS_CHAMPION = "CHAMPION"
STATUS_REJECTED = "REJECTED"
STATUS_RETIRED = "RETIRED"


def _models_dir() -> Path:
    return get_settings().data_store_dir / "models"


def get_or_create_model(session: Session, name: str, symbol: str, timeframe: str) -> MLModel:
    existing = session.execute(select(MLModel).where(MLModel.name == name)).scalar_one_or_none()
    if existing:
        return existing
    model = MLModel(name=name, symbol=symbol, timeframe=timeframe)
    session.add(model)
    session.flush()
    return model


def _next_version_label(session: Session, model_id: int) -> str:
    count = session.query(ModelVersion).filter(ModelVersion.model_id == model_id).count()
    return f"model_{count + 1:03d}"


def register_model_version(
    session: Session,
    model_name: str,
    symbol: str,
    timeframe: str,
    algorithm: str,
    feature_version: str,
    strategy_version: str,
    estimator,
    hyperparameters: dict,
    metrics: dict,
    walk_forward_results: dict,
    status: str = STATUS_TESTING,
) -> ModelVersion:
    model = get_or_create_model(session, model_name, symbol, timeframe)
    version_label = _next_version_label(session, model.id)

    artifact_dir = _models_dir() / model_name / version_label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.joblib"
    joblib.dump(estimator, artifact_path)

    metadata = {
        "model_name": model_name,
        "version_label": version_label,
        "algorithm": algorithm,
        "feature_version": feature_version,
        "strategy_version": strategy_version,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "walk_forward_results": walk_forward_results,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    version = ModelVersion(
        model_id=model.id,
        version_label=version_label,
        algorithm=algorithm,
        status=status,
        feature_version=feature_version,
        strategy_version=strategy_version,
        artifact_path=str(artifact_path),
        hyperparameters=hyperparameters,
        metrics=metrics,
        walk_forward_results=walk_forward_results,
    )
    session.add(version)
    session.flush()
    return version


def get_champion(session: Session, model_name: str) -> ModelVersion | None:
    model = session.execute(select(MLModel).where(MLModel.name == model_name)).scalar_one_or_none()
    if model is None:
        return None
    return (
        session.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id, ModelVersion.status == STATUS_CHAMPION)
        .one_or_none()
    )


def promote_to_champion(session: Session, version: ModelVersion) -> ModelVersion:
    """Promote `version` to champion, retiring the previous champion (if any)."""
    current_champion = (
        session.query(ModelVersion)
        .filter(ModelVersion.model_id == version.model_id, ModelVersion.status == STATUS_CHAMPION)
        .one_or_none()
    )
    if current_champion is not None and current_champion.id != version.id:
        current_champion.status = STATUS_RETIRED

    version.status = STATUS_CHAMPION
    version.promoted_at = dt.datetime.now(dt.timezone.utc)
    session.flush()
    return version


def reject_version(session: Session, version: ModelVersion, reason: str = "") -> ModelVersion:
    version.status = STATUS_REJECTED
    version.metrics = {**version.metrics, "rejection_reason": reason}
    session.flush()
    return version


def load_estimator(version: ModelVersion):
    return joblib.load(version.artifact_path)


def rollback_to_previous_champion(session: Session, model_name: str) -> ModelVersion | None:
    """
    Emergency rollback (Section 23: "keep rollback capability"). Demotes
    the current champion back to RETIRED and promotes the most recently
    retired version before it, if one exists.
    """
    model = session.execute(select(MLModel).where(MLModel.name == model_name)).scalar_one_or_none()
    if model is None:
        return None

    current_champion = (
        session.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id, ModelVersion.status == STATUS_CHAMPION)
        .one_or_none()
    )
    previous = (
        session.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id, ModelVersion.status == STATUS_RETIRED)
        .order_by(ModelVersion.promoted_at.desc().nullslast(), ModelVersion.id.desc())
        .first()
    )
    if previous is None:
        return None

    if current_champion is not None:
        current_champion.status = STATUS_RETIRED
    previous.status = STATUS_CHAMPION
    previous.promoted_at = dt.datetime.now(dt.timezone.utc)
    session.flush()
    return previous
