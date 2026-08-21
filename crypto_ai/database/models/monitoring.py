"""
Monitoring tables: performance_metrics, system_events, health_checks.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer

from crypto_ai.database.base import Base


class PerformanceMetric(Base):
    """Rolling performance metrics, snapshotted periodically (daily reports etc.)."""

    __tablename__ = "performance_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope: Mapped[str] = mapped_column(String(30), nullable=False)  # e.g. "paper_trading", "model_003"
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class SystemEvent(Base):
    """Structured event log (Section 34). Never contains secrets."""

    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    component: Mapped[str] = mapped_column(String(50), nullable=False)
    event: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)  # DEBUG/INFO/WARNING/ERROR/CRITICAL
    message: Mapped[str] = mapped_column(String, default="")
    context: Mapped[dict] = mapped_column(JSON, default=dict)


class HealthCheck(Base):
    __tablename__ = "health_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    component: Mapped[str] = mapped_column(String(50), nullable=False)  # database/exchange/websocket/llm
    status: Mapped[str] = mapped_column(String(10), nullable=False)  # OK/DEGRADED/DOWN
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
