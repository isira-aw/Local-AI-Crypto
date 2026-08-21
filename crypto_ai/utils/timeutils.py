"""
Timezone normalization helpers.

Why this exists: Postgres `TIMESTAMP WITH TIME ZONE` columns round-trip
timezone-aware datetimes, but SQLite (used for tests, and a supported
option for local experimentation) silently drops tzinfo and hands back
naive datetimes. Any code that compares a value read from the database
against `datetime.now(timezone.utc)` therefore works on Postgres and
raises

    TypeError: can't compare offset-naive and offset-aware datetimes

on SQLite — a crash that only appears at runtime, on one backend.

Everywhere in this project a naive datetime means UTC, so these helpers
normalize at the boundary instead of scattering tz checks through the
business logic.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd


def as_utc(value: dt.datetime | None) -> dt.datetime | None:
    """Attach UTC to a naive datetime; leave aware datetimes untouched."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def index_as_utc(index) -> pd.DatetimeIndex:
    """
    Normalize a pandas index of timestamps to tz-aware UTC, whether it
    arrives naive (SQLite), tz-aware (Postgres), or as object dtype.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(index), utc=True))
    return idx


def series_as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)
