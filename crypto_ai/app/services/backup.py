"""
Backup and restore (Section 33).

What is it?
    Creates a single timestamped archive containing everything needed to
    rebuild this system's state on another machine: the database, the
    configuration, and the trained model artifacts with their metadata.

Why is it needed?
    Trading history and model provenance are the two things you cannot
    regenerate. Market data can always be re-downloaded; a record of what
    the system decided, and the exact model that decided it, cannot.

    Section 33 also insists the restore path be **tested at least once**
    before you rely on it — an untested backup is an unverified one. That
    is why `verify_backup()` exists and why `restore_backup()` can restore
    into a scratch directory rather than over your live data.

What goes in / comes out?
    In: the current DATABASE_URL, config/, and data_store/models/.
    Out: data_store/backups/crypto_ai_backup_<UTC timestamp>.tar.gz

What is deliberately NOT backed up:
    - `.env` and any credentials. Section 33: "Do not back up secrets
      unnecessarily." A backup archive is the single most likely thing to
      get copied to a USB stick or a cloud drive; putting API keys in it
      would undo the care taken everywhere else.
    - Raw market data, which is large and re-downloadable on demand.

How do I start it?      python run.py backup
How do I restore?       python run.py restore <path-to-archive>
How do I verify one?    python run.py verify-backup <path-to-archive>
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from crypto_ai.config.loader import get_settings

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
BACKUP_PREFIX = "crypto_ai_backup_"

# Names that must never end up inside an archive, however the tree changes.
_SECRET_PATTERNS = (".env", "*.key", "*.secret", "*.pem")


def _backups_dir() -> Path:
    d = get_settings().data_store_dir / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _sqlite_path(url: str) -> Path:
    # sqlite:///relative/path.db  or  sqlite:////absolute/path.db
    raw = url.split("sqlite:///", 1)[-1]
    return Path(raw)


def _dump_database(url: str, dest_dir: Path) -> str:
    """
    Dump the database into dest_dir. Returns the filename written.

    SQLite uses the online backup API (safe while the app is running);
    Postgres uses pg_dump.
    """
    if _is_sqlite(url):
        src = _sqlite_path(url)
        out = dest_dir / "database.sqlite"
        if not src.exists():
            raise FileNotFoundError(f"SQLite database not found at {src}")
        # sqlite3's backup API produces a consistent snapshot even if the
        # application is mid-write, unlike a plain file copy.
        with sqlite3.connect(str(src)) as source, sqlite3.connect(str(out)) as target:
            source.backup(target)
        return out.name

    parsed = urlparse(url.replace("postgresql+psycopg2", "postgresql"))
    out = dest_dir / "database.sql"
    cmd = [
        "pg_dump",
        "--no-owner",
        "--no-privileges",
        "-h", parsed.hostname or "localhost",
        "-p", str(parsed.port or 5432),
        "-U", parsed.username or "crypto_ai",
        "-d", (parsed.path or "/crypto_ai").lstrip("/"),
        "-f", str(out),
    ]
    env_password = parsed.password or ""
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={"PGPASSWORD": env_password, "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pg_dump failed ({result.returncode}): {result.stderr.strip()}. "
            f"Is the postgresql-client package installed and the database reachable?"
        )
    return out.name


def _contains_secret(path: Path) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(path.name, pattern) for pattern in _SECRET_PATTERNS)


def create_backup(output_path: Path | None = None) -> Path:
    """Create a backup archive and return its path."""
    settings = get_settings()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_path or (_backups_dir() / f"{BACKUP_PREFIX}{timestamp}.tar.gz")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)

        db_filename = _dump_database(settings.database.url, staging)

        config_src = settings.project_root / "crypto_ai" / "config"
        config_dst = staging / "config"
        config_dst.mkdir(parents=True, exist_ok=True)
        for item in config_src.glob("*.yaml"):
            if not _contains_secret(item):
                shutil.copy2(item, config_dst / item.name)

        models_src = settings.data_store_dir / "models"
        n_models = 0
        if models_src.exists():
            models_dst = staging / "models"
            for item in models_src.rglob("*"):
                if item.is_file() and not _contains_secret(item):
                    rel = item.relative_to(models_src)
                    target = models_dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                    if item.name == "model.joblib":
                        n_models += 1

        manifest = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "database_kind": "sqlite" if _is_sqlite(settings.database.url) else "postgresql",
            "database_file": db_filename,
            "n_model_artifacts": n_models,
            "mode": settings.mode,
            "contains_secrets": False,
            "note": (
                "Secrets (.env, keys) are deliberately excluded — see "
                "crypto_ai/app/services/backup.py and Section 33."
            ),
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w:gz") as tar:
            for item in staging.iterdir():
                tar.add(item, arcname=item.name)

    logger.info("Backup written to %s", output_path)
    return output_path


def verify_backup(archive_path: Path) -> dict:
    """
    Check an archive is readable, complete, and secret-free WITHOUT
    restoring it. This is the cheap check you can run often; the real
    proof is restore_backup() into a scratch directory.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"No such backup: {archive_path}")

    problems: list[str] = []
    manifest = None
    names: list[str] = []

    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
        if MANIFEST_NAME not in names:
            problems.append("manifest.json missing")
        else:
            manifest = json.loads(tar.extractfile(MANIFEST_NAME).read().decode())

        if manifest:
            db_file = manifest.get("database_file")
            if db_file and db_file not in names:
                problems.append(f"database dump '{db_file}' listed in manifest but absent from archive")

        leaked = [n for n in names if _contains_secret(Path(n))]
        if leaked:
            problems.append(f"archive contains secret-looking files: {leaked}")

    return {
        "archive": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "n_entries": len(names),
        "manifest": manifest,
        "problems": problems,
        "ok": not problems,
    }


def restore_backup(archive_path: Path, target_dir: Path | None = None, apply_to_live: bool = False) -> dict:
    """
    Restore an archive.

    By default this extracts into a scratch directory and reports what it
    found — that is the safe way to TEST a backup (Section 33 / Section 59's
    `backup_restore_tested`) without touching live data.

    Pass apply_to_live=True to actually overwrite the live SQLite database
    and model artifacts. For Postgres, restoring is a `psql -f` of the
    dumped SQL file, which this function prints rather than runs, so you
    stay in control of which database it lands in.
    """
    archive_path = Path(archive_path)
    verification = verify_backup(archive_path)
    if not verification["ok"]:
        raise ValueError(f"Refusing to restore a bad archive: {verification['problems']}")

    settings = get_settings()
    target_dir = Path(target_dir) if target_dir else Path(tempfile.mkdtemp(prefix="crypto_ai_restore_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_dir, filter="data")

    manifest = verification["manifest"] or {}
    result = {
        "extracted_to": str(target_dir),
        "manifest": manifest,
        "applied_to_live": False,
        "next_step": None,
    }

    if not apply_to_live:
        result["next_step"] = (
            "Dry run only — nothing live was modified. Inspect the extracted files, "
            "then re-run with apply_to_live=True (or `python run.py restore <archive> --apply`) "
            "if you want to overwrite live state."
        )
        return result

    if manifest.get("database_kind") == "sqlite":
        src = target_dir / manifest["database_file"]
        dst = _sqlite_path(settings.database.url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            # Never destroy the current database without keeping a copy.
            shutil.copy2(dst, dst.with_suffix(dst.suffix + ".pre_restore"))
        shutil.copy2(src, dst)
        result["applied_to_live"] = True
    else:
        result["next_step"] = (
            f"Postgres restore is not run automatically. Apply it yourself with:\n"
            f"  psql -h <host> -U <user> -d <database> -f {target_dir / manifest.get('database_file', 'database.sql')}"
        )

    models_src = target_dir / "models"
    if models_src.exists():
        models_dst = settings.data_store_dir / "models"
        models_dst.mkdir(parents=True, exist_ok=True)
        for item in models_src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(models_src)
                out = models_dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, out)

    return result


def list_backups() -> list[dict]:
    out = []
    for p in sorted(_backups_dir().glob(f"{BACKUP_PREFIX}*.tar.gz"), reverse=True):
        out.append({
            "path": str(p),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "modified": dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).isoformat(),
        })
    return out


def prune_backups(keep: int = 7) -> list[str]:
    """Keep the newest `keep` archives, delete the rest."""
    archives = sorted(_backups_dir().glob(f"{BACKUP_PREFIX}*.tar.gz"), reverse=True)
    removed = []
    for old in archives[keep:]:
        old.unlink()
        removed.append(str(old))
    return removed
