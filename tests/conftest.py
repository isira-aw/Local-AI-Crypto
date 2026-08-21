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
import shutil
import tempfile
from pathlib import Path

import pytest

_TEST_DATA_STORE = Path(tempfile.mkdtemp(prefix="crypto_ai_test_data_store_"))
# Set BEFORE importing anything from crypto_ai so every config load — including
# the reload_config() calls individual tests make — picks it up.
os.environ["DATA_STORE_DIR"] = str(_TEST_DATA_STORE)

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("MODE", "RESEARCH")

from crypto_ai.config import loader as config_loader  # noqa: E402
from crypto_ai.database import base as db_base  # noqa: E402
from crypto_ai.database.models import Base  # noqa: E402

# Tests register real model artifacts, write backups and can trip the
# emergency-stop flag file — all of which land under data_store/. Pointing
# that at a throwaway directory keeps the suite from writing into (or
# corrupting) the user's live data.
#
# This is not hypothetical: an earlier version let a test that registers a
# deliberately-malformed 2-feature model overwrite
# data_store/models/btc_direction_classifier/model_001/model.joblib, which
# then broke a real run loading that champion.
@pytest.fixture(autouse=True)
def _isolated_config():
    """Fresh config per test; data_store points at the temp dir set above."""
    config_loader.reload_config()
    for sub in ("models", "backups", "logs", "raw", "processed"):
        (_TEST_DATA_STORE / sub).mkdir(parents=True, exist_ok=True)
    yield
    config_loader.reload_config()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_STORE, ignore_errors=True)


@pytest.fixture()
def test_data_store() -> Path:
    """The temp data_store directory this test session is using."""
    return _TEST_DATA_STORE


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
