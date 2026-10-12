> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Windows PostgreSQL setup and privileges

## Actual laptop setup

PostgreSQL **18.6** already existed at `C:\Program Files\PostgreSQL\18\bin`. The existing Windows service was running but had no administrator authentication available to this session. It was not altered. FaceAttend instead reuses those binaries in a separate native cluster at `127.0.0.1:55432` with user-only LocalAppData storage, SCRAM authentication and two project databases. This installs no system software and creates no new Windows service.

The cluster is initialized/migrated and runtime grants verified. See [DB_GUIDE.md](DB_GUIDE.md) for exact start/stop, secret-scoped launcher, migrations, health, inspection, retention and test commands. The private `credentials.json` is under `$env:LOCALAPPDATA\FaceAttend\phase4\postgresql`, outside OneDrive/Git. Do not display it, copy it into the project, synchronize it or attach it to a report. It stores database passwords protected by the owning user's NTFS ACL; it is not a hardware secret vault. Administrators/malware/backups remain a separate local-security concern. Routine app processes receive only the runtime secret; owner/test secrets are separate.

Read-only inventory commands:

```powershell
Get-Service -Name '*postgres*'
& 'C:\Program Files\PostgreSQL\18\bin\psql.exe' --version
& 'C:\Program Files\PostgreSQL\18\bin\pg_ctl.exe' -D "$env:LOCALAPPDATA\FaceAttend\phase4\postgresql\data" status
Get-NetTCPConnection -LocalPort 55432 -State Listen | Select-Object LocalAddress,LocalPort
```

`psql` is installed but was not on PATH; use the absolute binary or add that directory to this terminal's PATH. The listener must show only 127.0.0.1. The custom cluster runs under the operator's Windows account and requires operator start after reboot. The bootstrap administrator password was deliberately not retained; routine database owner/runtime passwords are retained privately. Future cluster-wide administration requires deliberate offline recovery by the owning Windows account; do not grant cluster administration to the application as a workaround.

## Fresh machine/manual setup

These are **manual reconstruction instructions**, not commands that were run against the unrelated existing Windows service. If PostgreSQL binaries are absent, use the [official Windows download page](https://www.postgresql.org/download/windows/), complete the installer interactively, choose a storage directory outside OneDrive, and keep the administrator password in the operator's own secret manager. Do not enable remote access/firewall rules. Do not install Docker or upgrade the project's Python/GPU environment.

For a new empty custom cluster, use the installed tools and an interactive password prompt. Do not run initdb on an existing populated directory:

```powershell
$phase4PgBin = 'C:\Program Files\PostgreSQL\18\bin'
$phase4NewData = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase4\manual-postgresql\data'
& "$phase4PgBin\initdb.exe" -D $phase4NewData -U faceattend_admin -W -A scram-sha-256 --encoding=UTF8
```

Before starting, edit that new cluster's `postgresql.conf`: `listen_addresses='127.0.0.1'`, `port=55432` (or another unused local port), `password_encryption='scram-sha-256'`, `log_statement='none'`, `log_min_error_statement='panic'`, `log_parameter_max_length_on_error=0`. Use a `pg_hba.conf` entry `host all all 127.0.0.1/32 scram-sha-256`; remove broader hosts/IPv6 entries if IPv4-only is intended. Protect the parent folder with the operator's NTFS ACL. Do not use trust authentication.

Start this new instance with `pg_ctl -D ... -l ... -w start`. In an interactive local psql session, authenticate with its administrator password:

```powershell
& "$phase4PgBin\psql.exe" -h 127.0.0.1 -p 55432 -U faceattend_admin -d postgres -W
```

Run these SQL/psql commands **only in that identified new project cluster**. Passwords are entered using `\password`, not SQL literals or shell history:

```sql
CREATE ROLE faceattend_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
CREATE ROLE faceattend_runtime LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
CREATE ROLE faceattend_test_owner LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
CREATE ROLE faceattend_test_runtime LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
\password faceattend_migrator
\password faceattend_runtime
\password faceattend_test_owner
\password faceattend_test_runtime

CREATE DATABASE faceattend_research OWNER faceattend_migrator;
CREATE DATABASE faceattend_phase4_test OWNER faceattend_test_owner;
REVOKE ALL ON DATABASE postgres FROM PUBLIC;
REVOKE ALL ON DATABASE template0 FROM PUBLIC;
REVOKE ALL ON DATABASE template1 FROM PUBLIC;
REVOKE ALL ON DATABASE faceattend_research FROM PUBLIC;
REVOKE ALL ON DATABASE faceattend_phase4_test FROM PUBLIC;
GRANT CONNECT ON DATABASE faceattend_research TO faceattend_migrator,faceattend_runtime;
GRANT CONNECT ON DATABASE faceattend_phase4_test TO faceattend_test_owner,faceattend_test_runtime;

\connect faceattend_research
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO faceattend_runtime;
\connect faceattend_phase4_test
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO faceattend_test_runtime;
\quit
```

If reconstructing on an existing shared server instead, do not revoke permissions on unrelated databases or assume permission to change server configuration. Have its operator set least privilege and approved project databases. This implementation intentionally avoided that dependency by using its own local cluster.

Supply the migration owner's URL using a hidden prompt; the runtime URL must be a different role:

```powershell
$phase4Secret = Read-Host 'Migration owner PostgreSQL URL (hidden)' -AsSecureString
$env:FACEATTEND_MIGRATION_DATABASE_URL = [System.Net.NetworkCredential]::new('', $phase4Secret).Password
Remove-Variable phase4Secret
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
Remove-Item Env:\FACEATTEND_MIGRATION_DATABASE_URL
```

After migration, connect as the project owner using interactive psql `-W` and grant only the new table permissions:

```sql
GRANT SELECT, INSERT, UPDATE ON research_daily_operation_stats TO faceattend_runtime;
```

The test database grants `SELECT, INSERT, UPDATE` to `faceattend_test_runtime` after migration; the opt-in test fixture does this itself. Runtime roles have no access to `alembic_version`. Grant no DELETE/TRUNCATE/schema CREATE/database CREATE/role memberships. Migration-table privileges remain with the owner. Set test URLs in hidden environment prompts only for the isolated test database; the tests reject the normal research DB or wrong role names before downgrade.

Keep backups/dumps/credential files outside the repository and OneDrive; `.gitignore` excludes common dump formats and local state. Retention deletion is a separate explicit owner operation and is not secure erasure. Credential rotation requires updating the private secret source and restarting the API; never show the resolved URL in a report.

Implementation references: [PostgreSQL initdb](https://www.postgresql.org/docs/current/app-initdb.html), [connection/listener settings](https://www.postgresql.org/docs/current/runtime-config-connection.html), [SQLAlchemy PostgreSQL](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html), [Alembic tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html).
