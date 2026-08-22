"""
Database engine/session setup.

What is it?
    Creates the SQLAlchemy engine and session factory used by every
    repository in the app.

Why is it needed?
    Centralizes connection handling (including basic reconnect-on-
    failure behaviour, see get_engine()) so the rest of the codebase
    just imports `SessionLocal`.

How do I start it?
    Nothing to "start" — the engine connects lazily on first use.
    Run `alembic upgrade head` first to create the tables (see
    docs/INSTALLATION.md).

How do I troubleshoot it?
    If you see "connection refused", check that Postgres is running
    (`docker compose up -d postgres`) and that DATABASE_URL in your
    .env points at the right host/port. See docs/TROUBLESHOOTING.md.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from crypto_ai.config.loader import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        url = get_settings().database.url
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(
            url,
            pool_pre_ping=True,  # detects dropped connections (Section 25: DB reconnect)
            connect_args=connect_args,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope: commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def wait_for_database(max_attempts: int = 10, base_delay_seconds: float = 2.0) -> bool:
    """
    Retry connecting to the database with exponential backoff.

    Used at startup (Section 25/26: the app must not crash permanently
    because the database is briefly unavailable, e.g. right after a
    power failure while Postgres is still starting up).
    """
    from sqlalchemy import text

    for attempt in range(1, max_attempts + 1):
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001 - we deliberately retry any error
            if attempt >= max_attempts:
                # Do NOT sleep after the final attempt — there is nothing
                # left to wait for. With the defaults that last delay is
                # 1024s, which doubled the time before the app could report
                # the database unreachable (~17 min -> ~34 min).
                logger.error("Database still unreachable after %s attempts: %s", max_attempts, exc)
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Database not ready (attempt %s/%s): %s. Retrying in %.1fs",
                attempt, max_attempts, exc, delay,
            )
            time.sleep(delay)
    return False


def reset_engine_for_tests() -> None:
    global _engine, _SessionFactory
    _engine = None
    _SessionFactory = None
