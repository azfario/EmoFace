> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Phase 4 local aggregate database guide

Phase 4 persists engineering counters for explicitly consented adult research calls. It does not create identity, student, attendance, expression-label, mood or mental-health records. Daily aggregates can still be identifying in a small setting, especially count one. Keep them local and restricted; this is data minimization, not irreversible anonymization. Counts represent qualifying calls, not unique people.

## Existing environment and database

Reuse the existing `.venv`. Additions are pinned in `requirements-phase4.txt`; all 149 previous distributions are constrained without upgrades. SQLAlchemy 2.0.54, Alembic 1.20.0, psycopg/psycopg-binary 3.3.6, greenlet 3.5.6 and Mako 1.4.3 are installed. PostgreSQL 18.6 binaries already existed. A separate native project cluster uses **127.0.0.1:55432**, outside OneDrive at `$env:LOCALAPPDATA\FaceAttend\phase4\postgresql\data`. The existing PostgreSQL Windows service is unchanged. No Docker/service installation/firewall opening occurred.

Database `faceattend_research` is owned by `faceattend_migrator`. Application role `faceattend_runtime` has CONNECT, public-schema USAGE and SELECT/INSERT/UPDATE only on the aggregate table. It has no table DELETE/TRUNCATE, schema/database CREATE, ownership or access to unrelated databases. Migration/retention use owner credentials, never runtime credentials. The isolated `faceattend_phase4_test` database has separate test owner/runtime roles. SCRAM authentication is required. Local restricted credentials are outside the repository; there is no automatic `.env` loader. The app never creates tables at startup.

## Start/stop the project cluster

From the workspace in PowerShell:

```powershell
$phase4PgBin = 'C:\Program Files\PostgreSQL\18\bin'
$phase4PgRoot = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase4\postgresql'
& "$phase4PgBin\pg_ctl.exe" -D "$phase4PgRoot\data" status
# Start only when status says the project cluster is stopped.
& "$phase4PgBin\pg_ctl.exe" -D "$phase4PgRoot\data" -l "$phase4PgRoot\server.log" -w start
# Stop this project instance after stopping the API and finishing requests.
& "$phase4PgBin\pg_ctl.exe" -D "$phase4PgRoot\data" -w stop -m fast
```

Do not stop the unrelated installed Windows service. This custom cluster is operator-started and is not registered for automatic startup. Its cluster administrator password was not retained; routine migrations and retention use the database owner credentials. Changes requiring cluster-wide administration would require a deliberate PostgreSQL offline administrator recovery by the owning Windows account. Keep that maintenance separate from application runtime.

## Configure and serve

The preferred helper reads the existing restricted local credential file, passes only the selected credential scope to one child process, and never prints URLs/passwords. It does **not** enable telemetry by itself.

```powershell
Set-Location 'C:\Users\User\OneDrive\Desktop\FaceAttend'
$env:FACEATTEND_DB_TELEMETRY_ENABLED = 'false'
.\.venv\Scripts\python.exe scripts\phase4_db_health.py
# Expected: {"status": "disabled"}, without any connection attempt.
.\.venv\Scripts\python.exe scripts\phase3_serve.py
# Ctrl+C stops the API.

# Explicitly enable aggregate writes for consented research operations.
$env:FACEATTEND_DB_TELEMETRY_ENABLED = 'true'
$env:FACEATTEND_DB_RETENTION_DAYS = '30'
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase4_db_health.py
# Expected: {"status": "ready"} when migrated/accessible/least-privilege.
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase3_serve.py
```

To use a user-managed secret instead of the helper, supply a complete runtime SQLAlchemy URL through an interactive hidden prompt; never paste it into shell history or source:

```powershell
$phase4Secret = Read-Host 'Runtime PostgreSQL URL (hidden)' -AsSecureString
$env:FACEATTEND_DATABASE_URL = [System.Net.NetworkCredential]::new('', $phase4Secret).Password
Remove-Variable phase4Secret
# Format: postgresql+psycopg with explicit user/password, host 127.0.0.1 and local port/database.
# Do not print the resolved environment value.
```

Only that driver/host is accepted; URL query parameters, alternate hosts, missing credentials and unsafe ports are refused. A bad enabled configuration produces coarse unavailable status while model processing continues. `.env.example` contains keys only; copying it does not enable telemetry or load secrets.

## Routes

| Method and route | Phase 4 behavior |
|---|---|
| GET `/health` | Original model/GPU readiness, independent of DB. |
| GET `/api/v1/models/info` | Original safe model metadata. |
| POST `/api/v1/face/detect` | Original consented single-face diagnostic and quality rules. |
| POST `/api/v1/face/embedding-info` | Original dimension/norm metadata, no vector. |
| POST `/api/v1/expression/predict` | Original visible-expression response, no internal-mood inference. |
| POST `/api/v1/face/compare-diagnostic` | Original request-local reference and diagnostic cosine; always unverified, null threshold, `THRESHOLD_NOT_CALIBRATED`. |
| GET `/health/db` | Only `{"status":"disabled"}`, `ready` or `unavailable`; 503 for unavailable, otherwise 200. |

`/health/db` inherits loopback Host/Origin/client/server admission and no-store headers. It is deliberately omitted from Swagger/OpenAPI so the original six-route schema contract remains exactly intact. This guide documents the additional operator probe. No public analytics endpoint exists. Existing upload bounds, CUDA-only execution, request-local encryption/deletion, no CORS/proxy trust and single-worker serving stay unchanged.

## What is counted

- Both original multipart fields must validate: `adult_consent=true` and `research_opt_in=true`.
- Origin/admission, busy, upload/schema/consent/decode failures and unavailable models before ML entry do not count. Invalid claimed IDs are also excluded.
- After decoded inputs enter the existing ML operation, success counts as `completed`; face/quality/reference-consistency rejections count as `quality_rejected`; other processing failures and cooperative inference deadlines count as `processing_error`.
- Canceled HTTP tasks do not write metrics. There are no per-request identifiers or records.
- Processing time runs from entry into the operation through its request-local cleanup. It excludes upload, decode, initial model availability checks and SQL time. Duration is rounded down to 10 ms and capped at 120,000 ms per call. Native GPU deadlines are cooperative, so the cap can understate unusually long failures; it is not a latency benchmark.
- UTC day is chosen when the counter is written, including calls crossing midnight. Only the date is persisted, never an event timestamp.
- Enabled telemetry is best effort. A busy telemetry slot drops a metric without queuing; an outage/overflow drops it and marks DB unavailable with a fixed warning. It never changes the model result. Consequently counts need not equal all HTTP attempts.

The DB receives only allowed operation/outcome strings and a bounded integer duration. It never receives request/response objects, file metadata, claimed IDs, images/crops/vectors/keys, hashes, scores, expression labels/confidences, IPs or user agents. There is no background queue retaining uploads. Sync SQL runs in bounded AnyIO workers after native inference/cleanup, with no transaction held over GPU work. Engine pool size two, no overflow, pool wait one second, connect timeout two seconds, statement timeout 1.5 seconds, lock timeout 0.8 seconds and idle-transaction timeout two seconds bound normal waits. OS/network failures and native calls are not absolute wall-clock guarantees.

## Schema, migrations and operator queries

Only `alembic_version` and `research_daily_operation_stats` exist. The aggregate columns are:

| Column | Type | Constraint |
|---|---|---|
| day_utc | DATE | Nonnull; UTC date only; first PK component. |
| operation | VARCHAR(32) | Nonnull; `face_detect`, `embedding_info`, `expression_predict`, `compare_diagnostic`; second PK component. |
| outcome | VARCHAR(24) | Nonnull; `completed`, `quality_rejected`, `processing_error`; third PK component. |
| request_count | BIGINT | Nonnull, nonnegative. |
| total_processing_ms | BIGINT | Nonnull, nonnegative. |

Revision `0001_daily_stats` creates one aggregate table, a composite primary key and four CHECK constraints. There is only the primary-key index; no extra index is needed for the small date-led workload. PostgreSQL BIGINT enforces the upper storage range. Parameterized `INSERT ... ON CONFLICT DO UPDATE ... RETURNING` atomically increments count by one and duration by the safe input; overflow refuses the write and rolls back. Repository summaries accept at most 366 ordered dates. Sessions close and transactions roll back on errors.

```powershell
# Owner-only migration; no real URL in alembic.ini.
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py owner .\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py owner .\.venv\Scripts\python.exe -m alembic current
# SQL preview has no credentials and makes no connection.
.\.venv\Scripts\python.exe -m alembic upgrade head --sql

# Local, restricted aggregate inspection; no HTTP analytics route.
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase4_db_summary.py --start 2026-10-01 --end 2026-10-10
```

Synthetic test example: 64 increments of 20 ms on the fictitious date 1900-01-01 produce `face_detect / completed / request_count=64 / total_processing_ms=1280`. That row was generated only in the disposable test database and cleared afterward; it is not a person or real production measurement. The project research database was left empty after implementation.

Migration downgrade drops aggregate data. It was tested only on the isolated test database. **Do not downgrade a populated operator database without backup and explicit approval.** For future schema changes, inspect revision SQL, stop writers, back up the local database outside the repository/OneDrive, migrate with the owner and reapply minimal table grants if tables are recreated. Restore/backups and cluster-wide privilege changes are manual operator work.

## Retention

Default retention is 30 UTC days, configurable 1–365. The command deletes only aggregate dates **strictly before** `today_utc - retention_days`; it preserves the cutoff date. It is explicit, not scheduled, and uses the owner role because the runtime cannot DELETE. First preview the cutoff, then explicitly apply:

```powershell
$env:FACEATTEND_DB_TELEMETRY_ENABLED = 'false'
$env:FACEATTEND_DB_RETENTION_DAYS = '30'
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py owner .\.venv\Scripts\python.exe scripts\phase4_db_retention.py
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py owner .\.venv\Scripts\python.exe scripts\phase4_db_retention.py --apply
```

Deletion is not secure erasure of storage, WAL, backups or original supplied photos. Private biometric files remain in their existing separate Phase 2 base and are never migrated into PostgreSQL. Database secrets are separate from biometric template keys.

## Tests and preservation

```powershell
.\.venv\Scripts\python.exe scripts\phase4_run_tests.py
# Explicit live PG opt-in; strictly guarded separate DB name and test-role pair.
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py test .\.venv\Scripts\python.exe scripts\phase4_run_tests.py --postgres-integration
# Actual CUDA integration is optional; reuse only the already authorized stills.
.\.venv\Scripts\python.exe scripts\phase4_with_local_db.py test .\.venv\Scripts\python.exe scripts\phase4_run_tests.py --postgres-integration --gpu-integration
.\.venv\Scripts\python.exe scripts\phase4_preserve.py --check
```

The wrapper invokes the existing preservation-safe Phase 3 runner in its explicit Phase 4 receipt mode. It forces normal telemetry off and removes inherited normal runtime/owner URLs from the test process; only explicitly isolated test credentials can reach the live-PG fixtures. It protects historical Phase 2 and Phase 3 GPU receipts, archives fresh evidence separately, and checks the immutable 269-file/149-package Phase 4 baseline against exact documented edits. Original earlier checkers retain their original baselines; they still flag the README update that the user authorized before Phase 4, and the Phase 3 checker also flags the planned `.gitignore` extension. The exact historical README backup matches its old digest. The Phase 2A 27-artifact checker remains clean. New GPU tests are synthetic/authorized-still technical checks, not retraining, held-out evaluation or calibration.

Do not commit/push data, credentials, dumps, source or reports from this phase without a separate user request. Phase 5, liveness, biometric calibration and operational attendance remain outside this implementation.
