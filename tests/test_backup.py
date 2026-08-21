"""
Backup/restore tests (Section 33).

Section 33 is explicit that the restore path must be tested at least once
before it is relied on — "a backup that has never been restored is
unverified". These tests perform a real round trip: create a database,
back it up, destroy the original, restore, and assert the data came back.
"""

import datetime as dt
import os
import sqlite3
import tarfile
from pathlib import Path

import pytest

from crypto_ai.config import loader as config_loader


@pytest.fixture()
def sqlite_project(tmp_path, monkeypatch):
    """A self-contained project with a real on-disk SQLite database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LLM_ENABLED", "false")
    config_loader.reload_config()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from crypto_ai.database.models import Base
    from crypto_ai.database.repositories.market_data_repo import upsert_candles
    from crypto_ai.database.models.trading import PaperTrade

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    upsert_candles(session, "BTC/USDT", "5m", [
        {"timestamp": start + dt.timedelta(minutes=5 * i), "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.0 + i, "volume": 1.0}
        for i in range(10)
    ])
    session.add(PaperTrade(
        symbol="BTC/USDT", side="BUY", entry_time=start, entry_price=100.0,
        quantity=1.0, status="CLOSED", pnl=42.0, strategy_version="s1",
        exit_time=start + dt.timedelta(minutes=30), exit_price=142.0,
    ))
    session.commit()
    session.close()
    engine.dispose()

    yield {"tmp_path": tmp_path, "db_path": db_path}
    config_loader.reload_config()


def _make_model_artifact(monkeypatch, tmp_path):
    """Put a fake trained model where the backup expects to find one."""
    models = tmp_path / "data_store" / "models" / "btc" / "model_001"
    models.mkdir(parents=True, exist_ok=True)
    (models / "model.joblib").write_bytes(b"fake-model-bytes")
    (models / "metadata.json").write_text('{"algorithm": "random_forest"}')
    return models


def test_backup_creates_readable_archive(sqlite_project, tmp_path):
    from crypto_ai.app.services.backup import create_backup, verify_backup

    out = tmp_path / "backup.tar.gz"
    create_backup(output_path=out)

    assert out.exists() and out.stat().st_size > 0
    result = verify_backup(out)
    assert result["ok"], result["problems"]
    assert result["manifest"]["database_kind"] == "sqlite"


def test_backup_excludes_secrets(sqlite_project, tmp_path, monkeypatch):
    """
    Section 33: 'Do not back up secrets unnecessarily.' A backup archive is
    the most likely artifact to be copied somewhere insecure.
    """
    from crypto_ai.app.services.backup import create_backup

    # Plant a .env right next to the config the backup copies.
    settings = config_loader.get_settings()
    planted = settings.project_root / "crypto_ai" / "config" / ".env"
    planted.write_text("BINANCE_LIVE_API_SECRET=supersecret\n")
    try:
        out = tmp_path / "backup.tar.gz"
        create_backup(output_path=out)
        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        assert not any(n.endswith(".env") for n in names), f"secret leaked into archive: {names}"
        blob = out.read_bytes()
        assert b"supersecret" not in blob, "secret value found inside archive bytes"
    finally:
        planted.unlink(missing_ok=True)


def test_verify_backup_detects_a_corrupt_archive(tmp_path):
    from crypto_ai.app.services.backup import verify_backup

    bad = tmp_path / "empty.tar.gz"
    with tarfile.open(bad, "w:gz"):
        pass  # archive with no manifest

    result = verify_backup(bad)
    assert result["ok"] is False
    assert any("manifest" in p for p in result["problems"])


def test_verify_backup_missing_file_raises(tmp_path):
    from crypto_ai.app.services.backup import verify_backup

    with pytest.raises(FileNotFoundError):
        verify_backup(tmp_path / "nope.tar.gz")


def test_restore_dry_run_touches_nothing(sqlite_project, tmp_path):
    from crypto_ai.app.services.backup import create_backup, restore_backup

    out = tmp_path / "backup.tar.gz"
    create_backup(output_path=out)

    db_path = sqlite_project["db_path"]
    before = db_path.read_bytes()

    result = restore_backup(out)  # default: dry run
    assert result["applied_to_live"] is False
    assert "Dry run" in result["next_step"]
    assert db_path.read_bytes() == before, "dry run modified the live database"


def test_full_restore_round_trip_recovers_data(sqlite_project, tmp_path):
    """
    The test Section 33 actually asks for: destroy the database, restore
    from the archive, and confirm the trading history came back.
    """
    from crypto_ai.app.services.backup import create_backup, restore_backup

    out = tmp_path / "backup.tar.gz"
    create_backup(output_path=out)

    db_path = sqlite_project["db_path"]

    # Destroy the live database entirely.
    db_path.unlink()
    assert not db_path.exists()

    result = restore_backup(out, apply_to_live=True)
    assert result["applied_to_live"] is True
    assert db_path.exists(), "restore did not recreate the database"

    conn = sqlite3.connect(str(db_path))
    try:
        n_candles = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        pnl = conn.execute("SELECT pnl FROM paper_trades").fetchone()[0]
    finally:
        conn.close()

    assert n_candles == 10, f"expected 10 candles restored, got {n_candles}"
    assert pnl == 42.0, "paper-trade history did not survive the round trip"


def test_restore_keeps_a_copy_of_the_pre_restore_database(sqlite_project, tmp_path):
    from crypto_ai.app.services.backup import create_backup, restore_backup

    out = tmp_path / "backup.tar.gz"
    create_backup(output_path=out)

    db_path = sqlite_project["db_path"]
    restore_backup(out, apply_to_live=True)

    preserved = db_path.with_suffix(db_path.suffix + ".pre_restore")
    assert preserved.exists(), "restore overwrote the live DB without keeping a copy"


def test_restore_refuses_a_bad_archive(tmp_path):
    from crypto_ai.app.services.backup import restore_backup

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz"):
        pass

    with pytest.raises(ValueError, match="Refusing to restore"):
        restore_backup(bad, apply_to_live=True)


def test_prune_backups_keeps_newest(sqlite_project, monkeypatch):
    from crypto_ai.app.services.backup import _backups_dir, prune_backups

    d = _backups_dir()
    for i in range(5):
        p = d / f"crypto_ai_backup_2024010{i}T000000Z.tar.gz"
        p.write_bytes(b"x")
    removed = prune_backups(keep=2)
    remaining = sorted(x.name for x in d.glob("crypto_ai_backup_*.tar.gz"))
    assert len(remaining) == 2
    assert len(removed) >= 3
    # The two kept must be the newest by name (timestamps sort lexically).
    assert remaining == sorted(remaining, reverse=False)
