"""
Shared pytest fixtures.

Tests run against an isolated in-memory SQLite database, not Postgres,
so the suite has no external dependencies and runs the same way on
any contributor's machine or in CI. Postgres-specific behaviour (JSON
columns, upserts) is exercised through the SQLAlchemy ORM layer, which
is dialect-agnostic here.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("MODE", "RESEARCH")

from crypto_ai.config import loader as config_loader  # noqa: E402
from crypto_ai.database import base as db_base  # noqa: E402
from crypto_ai.database.models import Base  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_config_cache():
    config_loader.reload_config()
    yield
    config_loader.reload_config()


@pytest.fixture()
def db_session():
    """A fresh in-memory SQLite database per test."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
