"""
SQLAlchemy models for every table listed in Section 11 of the spec.

Importing this module registers all models on `Base.metadata`, which is
what Alembic's env.py and `Base.metadata.create_all()` (used by tests)
rely on.
"""

from crypto_ai.database.base import Base
from crypto_ai.database.models.market import FeatureSet, MarketData
from crypto_ai.database.models.ml import MLModel, ModelVersion, TrainingRun
from crypto_ai.database.models.monitoring import HealthCheck, PerformanceMetric, SystemEvent
from crypto_ai.database.models.trading import (
    BacktestRun,
    PaperTrade,
    PortfolioSnapshot,
    Prediction,
    Signal,
)

__all__ = [
    "Base",
    "MarketData",
    "FeatureSet",
    "MLModel",
    "ModelVersion",
    "TrainingRun",
    "Prediction",
    "Signal",
    "BacktestRun",
    "PaperTrade",
    "PortfolioSnapshot",
    "PerformanceMetric",
    "SystemEvent",
    "HealthCheck",
]
