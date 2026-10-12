> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Phase 4B local adult enrollment guide

Use this guide for the consented enrollment lifecycle. For a fresh machine, start with [SETUP_NEW_PC.md](../SETUP_NEW_PC.md).

## Scope and prerequisites

Use the existing Windows `.venv`, original SCRFD/ArcFace CUDA models and native PostgreSQL cluster at `127.0.0.1:55432`. Enrollment is disabled by default. This is a single local operator's consenting-adult research prototype. Registration is a consented reference record; it does not establish verified identity. There is no search, camera, frontend, expression integration or attendance in Phase 4B.

Do not reinstall packages, initialize the existing database directory, change model policy, retrain or rerun the Phase 1 held-out evaluation. Existing Phase 4 guides cover cluster startup. On this laptop:

```powershell
& 'C:\Program Files\PostgreSQL\18\bin\pg_ctl.exe' -D "$env:LOCALAPPDATA\FaceAttend\phase4\postgresql\data" status
Get-NetTCPConnection -LocalPort 55432 -State Listen | Select-Object LocalAddress,LocalPort
```

Only `127.0.0.1` should be listening. Do not start a second instance or alter the unrelated installed PostgreSQL Windows service. If the project cluster is stopped, use the original Phase 4 start command; background helpers must be hidden. Do not use PowerShell `Start-Process -Wait` for pg_ctl, since it can wait for the long-lived server's process tree.

## Database migration

The normal local database already has the Phase 4B migration and runtime grants. This idempotent owner-scoped command reconstructs them without showing credentials:

```powershell
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py owner .\.venv\Scripts\python.exe scripts\phase4b_migrate.py
```

The new `demo` schema has its own Alembic version table and revision `0002_consented_enrollment`, following `0001_daily_stats`. The original `public` table and version stay unchanged. Runtime gets only schema USAGE and SELECT/INSERT/UPDATE/DELETE on registration metadata. It cannot migrate, truncate or read migration history, and it still cannot delete aggregate telemetry.

Upgrade/downgrade roundtrips are tested only in the isolated test database. A manual metadata rollback uses the migration owner's URL and `python -m alembic -c alembic-phase4b.ini downgrade 0001_daily_stats`. **This drops registration metadata**; first stop the app, explicitly delete/revoke registrations and resolve pending ciphertext, and obtain deliberate owner approval. Downgrade does not itself remove encrypted gallery files. Do not run it as routine maintenance on the normal database. Reapply `scripts/phase4b_migrate.py` for upgrade and grants. Never use the original Phase 4 config to create or remove enrollment records.

The local PostgreSQL configuration additionally uses `log_error_verbosity=terse`, so duplicate-value error DETAIL cannot expose a chosen name. The exact prior config backup is in the existing protected PostgreSQL parent, outside the project. Preserve original statement/parameter logging restrictions. SQL engine echo and server access logs remain off.

## Explicit operator secret setup

Choose protected storage outside OneDrive and the repository, within the established private base. Keep the gallery, encryption key and password-hash file separate. Example paths below contain no participant data. Never print key contents or database URLs.

```powershell
$phase4bPrivate = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase2\private\phase4b'
$env:FACEATTEND_ENROLLMENT_GALLERY_DIR = Join-Path $phase4bPrivate 'gallery'
$env:FACEATTEND_ENROLLMENT_KEY_FILE = Join-Path $phase4bPrivate 'secrets\gallery.key'
$env:FACEATTEND_OPERATOR_AUTH_FILE = Join-Path $phase4bPrivate 'secrets\operator.secret'
.\.venv\Scripts\python.exe scripts\phase4b_admin.py setup --gallery-dir $env:FACEATTEND_ENROLLMENT_GALLERY_DIR --key-file $env:FACEATTEND_ENROLLMENT_KEY_FILE --auth-file $env:FACEATTEND_OPERATOR_AUTH_FILE
```

The command prompts twice for a password of at least 12 characters, creates restrictive local ACLs, a new independent Fernet key and a scrypt password record, and refuses to overwrite existing secrets. Setup does not enroll anyone. App startup never creates production keys. An operator-supplied key must be a valid protected Fernet key file. Protect auth/key parents and never put either in the gallery, repo, shell arguments, report, screenshot or cloud backup. This phase has not set up a retained production gallery or production operator credentials.

Start one loopback API worker with runtime credentials scoped to the child process:

```powershell
$env:FACEATTEND_ENROLLMENT_ENABLED = 'true'
$env:FACEATTEND_ENROLLMENT_RETENTION_DAYS = '30'
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase3_serve.py
```

Use `http://127.0.0.1:8000` on this laptop only. No proxy, tunnel, wildcard CORS, public bind or extra GPU worker. Existing health and six OpenAPI research routes retain their contracts; management routes below are deliberately excluded from that old schema. The operator CLI is the supported immediate workflow. A missing enrollment dependency refuses management without breaking the DB-independent health route.

## Enrollment and management

Get explicit adult research and persistent-enrollment consent before every registration or re-enrollment. Use a chosen display label/pseudonym, not a student identifier; labels currently support 1–64 ASCII letters/digits/spaces and `_.:-`, starting with a letter/digit, without leading/trailing spaces. Case-insensitive labels are unique. Capacity is 100 records. Retention is 1–30 days, bounded by the configured ceiling.

Use 3–5 distinct, clear, single-person JPEG/PNG inputs under a protected, non-synced local input directory. Each file is bounded to 8 MiB; the bounded multipart parser also enforces aggregate size, dimensions and timeouts. The existing blur minimum remains 60, with unchanged face-count, landmark, detection and embedding-consistency gates. More light, stable focus and a visible frontal face help; duplicating an image does not satisfy the distinct-image requirement. No score-based image selection or QC relaxation is performed.

The source folder supplied by the user is already under OneDrive. The integration run used restricted anonymous temporary copies outside OneDrive and left those user-owned originals untouched. The application cannot remove copies already synchronized by OneDrive. For future operator enrollment, place inputs outside synced storage first; the CLI refuses synced/repository paths. Do not include private names in example commands or share terminal history containing participant file paths.

Example with anonymous, operator-controlled protected paths:

```powershell
$phase4bInput = Join-Path $phase4bPrivate 'inputs'
.\.venv\Scripts\python.exe scripts\phase4b_admin.py enroll --images "$phase4bInput\image01.jpg" "$phase4bInput\image02.jpg" "$phase4bInput\image03.jpg" --confirm-adult-consent --confirm-persistent-enrollment
.\.venv\Scripts\python.exe scripts\phase4b_admin.py list
.\.venv\Scripts\python.exe scripts\phase4b_admin.py list --private-output
.\.venv\Scripts\python.exe scripts\phase4b_admin.py inspect --private-output
.\.venv\Scripts\python.exe scripts\phase4b_admin.py revoke
.\.venv\Scripts\python.exe scripts\phase4b_admin.py delete
.\.venv\Scripts\python.exe scripts\phase4b_admin.py reenroll --images "$phase4bInput\image01.jpg" "$phase4bInput\image02.jpg" "$phase4bInput\image03.jpg" --confirm-adult-consent --confirm-persistent-enrollment
```

Password, chosen label and selected UUID are prompted, not passed in arguments. Default output is coarse; `--private-output` deliberately reveals authorized personal registration metadata locally. Do not redirect, log, upload or publish it. The HTTP list defaults to page 1 of four pages, 25 records per page; the CLI currently lists page 1 only. Inspect checks only the selected reference's status/expiry/fingerprint and encrypted format; it does not compare a probe or identify a person. Re-enrollment uses the same selected UUID, fresh consent, the same label (case-insensitive) and a new opaque locator. Delete is idempotent. Revoke immediately blocks eligibility and removes its reference, leaving consent metadata until deletion/expiry cleanup.

Uploads are decoded in RAM. PostgreSQL stores only UUID, chosen label, opaque locator, consent/purpose/enrollment/expiry times, lifecycle state and model fingerprint. The original TemplateStore persists encrypted reference data (including normalized embedding and private sample digests); no photo, crop, EXIF, raw vector, score or expression field is written to PostgreSQL. Authenticated responses never expose locator, vector, key or original path; they remain `verified=false`, `threshold=null`, with uncalibrated-research context.

## HTTP management contract

All these routes require local Host/Origin admission; all except login require the operator cookie. Responses are private/no-store. Mutations require the session CSRF header and fixed rate limits.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/demo/session` | JSON password and `x-faceattend-login: 1`; returns CSRF, sets HttpOnly SameSite Strict cookie |
| DELETE | `/api/v1/demo/session` | Logout with `x-faceattend-csrf` |
| GET | `/api/v1/demo/enrollments?page=1` | Bounded metadata listing |
| GET | `/api/v1/demo/enrollments/{identity_id}` | Selected metadata status |
| POST | `/api/v1/demo/enrollments` | Explicit-consent multipart enrollment |
| POST | `/api/v1/demo/enrollments/{identity_id}/reenroll` | Explicit replacement under fresh consent |
| POST | `/api/v1/demo/enrollments/{identity_id}/inspect-reference` | Selected reference integrity/status only |
| POST | `/api/v1/demo/enrollments/{identity_id}/revoke` | Block then remove ciphertext |
| DELETE | `/api/v1/demo/enrollments/{identity_id}` | Block, remove scoped ciphertext and metadata |

Multipart fields: `display_name`, `adult_consent=true`, `research_opt_in=true`, `enrollment_consent=true`, `retention_days`, and 3–5 `enrollment_images` files. Login is bounded to 4 KiB/5 seconds, 5 attempts/minute; authenticated mutations 20/minute. A new login replaces the single active session; expiry is 15 minutes and restart clears it. Cookies use local HTTP (`Secure=false`), appropriate only to this loopback prototype. No credentials in query strings or recorded sample requests.

## Retention, recovery and key lifecycle

With the same enabled settings and runtime-scoped credentials:

```powershell
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase4b_admin.py retention
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase4b_admin.py retention --apply
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase4b_admin.py recover
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase4b_admin.py recover --apply
```

Without `--apply` these commands are read-only and report the number of expired registrations; recovery preview does not inventory every pending journal. They prompt for the operator password. Maintenance is explicit; no scheduler runs here. Expired entries are excluded by selected-reference inspection even before cleanup.

Writes progress through staged encrypted reference and intent, committed pending metadata, atomic file promotion, old-reference cleanup and active status. DB/file operations are not cross-resource ACID. A crash or ambiguous DB commit leaves blocked/recoverable state; explicit recovery consults the DB before completing or cleaning a known orphan older than 24 hours. File locks, an advisory DB lock and uniqueness constraints serialize mutations. Deletion commits `deletion_pending` before cleanup, and removes metadata last. A cleanup failure remains blocked and must be retried; do not claim deletion succeeded.

Keep the key and a consistent DB/gallery backup in separately protected storage only if the participant's agreed policy permits backups. Backups retain personal metadata/biometric ciphertext and require their own expiry/revocation handling. The app does not erase PostgreSQL WAL, OS caches, SSD remnants or external backups. No secure-erasure promise.

Wrong/lost keys fail reference reads and enrollment. Ordinary active ciphertext can still be deleted without decryption, but encrypted pending journals may need the original key to finish scoped old-reference cleanup; do not delete arbitrary files to bypass recovery. If no valid key/backup remains, records cannot recover their vectors. Obtain fresh consent, resolve/delete old records safely and re-enroll using a new independently managed key. There is no automatic bulk key rotation or password-reset feature. For rotation, stop mutations, retain a protected consistent backup under the consent policy, deliberately delete/re-enroll the affected registrations with fresh consent and new secrets, verify restart/deletion and retire old backups/key according to that policy. Do not switch the configured key while expecting old ciphertext to decrypt. Never print keys or silently rewrite references.

## Verification and phase boundary

```powershell
.\.venv\Scripts\python.exe scripts\phase4b_preserve.py --check
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py test .\.venv\Scripts\python.exe scripts\phase4b_run_tests.py --postgres-integration --gpu-integration
```

Adult-image integration is separately gated by `--adult-image-integration` and requires both integration flags plus prepared protected copies under the confirmed adult consent. Do not run it while replacements are still being made. No Test-folder probes or camera are used. Prior receipts are restored byte-for-byte and new evidence stays under `reports/phase4b/`.

Phase 4C and all later phases require a separate explicit approval. No approval is implied by this guide, passing tests or providing replacement photos.
