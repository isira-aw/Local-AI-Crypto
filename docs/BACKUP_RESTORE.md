# Backup and Restore

## What is it?

A single timestamped archive containing everything needed to rebuild this
system's state on another machine — `crypto_ai/app/services/backup.py`.

## Why is it needed?

Two things in this system cannot be regenerated:

1. **Trading history** — the record of what the system decided and what
   happened as a result.
2. **Model provenance** — the exact trained artifact that made each
   decision, plus its metrics and walk-forward results.

Market data can always be re-downloaded from the exchange. Those two
cannot. Section 33 of the design document requires backing them up, and
Section 59 lists `backup_restore_tested` as a live-trading gate item.

## What's in an archive (and what deliberately isn't)

**Included:**
- The database (SQLite via the online-backup API, or a `pg_dump` for
  Postgres)
- `crypto_ai/config/*.yaml` — settings and risk configuration
- `data_store/models/` — trained model artifacts and their metadata
- `manifest.json` — what was backed up, when, and from which mode

**Deliberately excluded:**
- **`.env` and any credentials.** Section 33: "Do not back up secrets
  unnecessarily." A backup archive is the single most likely artifact to
  end up on a USB stick or a cloud drive — putting API keys in it would
  undo the care taken everywhere else. This is enforced in code and
  covered by a test that plants a fake `.env` and asserts neither the file
  nor its contents appear in the archive.
- **Raw market data**, which is large and re-downloadable on demand.

## Commands

```bash
python run.py backup                      # create + verify an archive
python run.py list-backups                # what you have
python run.py verify-backup <archive>     # cheap integrity/secret check
python run.py restore <archive>           # DRY RUN — touches nothing
python run.py restore <archive> --apply   # actually overwrite live data
```

A backup also runs automatically once a day (scheduler job `backup`, hour
configurable via `scheduler.backup_hour_utc`), which verifies the archive
and prunes to the newest 7.

## Testing your backup — do this once, before you need it

> A backup that has never been restored is unverified.

`restore` defaults to a **dry run**: it extracts to a scratch directory and
reports what it found, without touching your live data. That's the safe way
to confirm an archive is real.

```bash
python run.py backup
python run.py restore data_store/backups/crypto_ai_backup_<timestamp>.tar.gz
```

For a genuine end-to-end test, do it on a throwaway copy of the project
rather than your live one, and use `--apply`.

When `--apply` does overwrite a SQLite database, it first copies the
existing file to `<name>.db.pre_restore` — so even a mistaken restore is
recoverable. For Postgres, the restore command is **printed rather than
executed**, so you stay in control of which database it lands in:

```
psql -h <host> -U <user> -d <database> -f <extracted>/database.sql
```

## Troubleshooting

- **`pg_dump failed`** — install the `postgresql-client` package (the
  `pg_dump` binary is not part of the Python dependencies) and confirm the
  database is reachable with the credentials in your `DATABASE_URL`.
- **`Refusing to restore a bad archive`** — the archive is missing its
  manifest or database dump, or contains secret-looking files. Run
  `python run.py verify-backup <archive>` to see exactly what's wrong.
- **Restored but the app still shows old data** — restart the application;
  it holds an open database connection.
