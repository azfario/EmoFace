#!/bin/sh
set -eu
# Refuse unreadable/empty files instead of creating roles without passwords.
for secret_file in /run/secrets/db_owner_password /run/secrets/db_runtime_password; do
    if [ ! -r "$secret_file" ] || [ ! -s "$secret_file" ]; then
        echo 'Compose database secret files are unreadable or empty.' >&2
        exit 1
    fi
done
# Secret substitution stays inside psql, with all raw diagnostics suppressed.
if ! psql --no-psqlrc --username "$POSTGRES_USER" --dbname postgres --set ON_ERROR_STOP=1 >/dev/null 2>&1 <<'SQL'
\set owner_password `cat /run/secrets/db_owner_password`
\set runtime_password `cat /run/secrets/db_runtime_password`
CREATE ROLE faceattend_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS PASSWORD :'owner_password';
CREATE ROLE faceattend_runtime LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS PASSWORD :'runtime_password';
CREATE DATABASE faceattend_research OWNER faceattend_migrator;
REVOKE ALL ON DATABASE faceattend_research FROM PUBLIC;
GRANT CONNECT ON DATABASE faceattend_research TO faceattend_migrator,faceattend_runtime;
\connect faceattend_research
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO faceattend_runtime;
SQL
then
    echo 'Compose database initialization unavailable.' >&2
    exit 1
fi
echo 'Isolated Compose roles and database ready.'
