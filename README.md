# EmoFace: Early Student Emotion Detection

## Project Overview

**EmoFace** is an end-to-end machine learning project designed to automate student attendance through facial recognition while classifying facial expressions using a custom fine-tuned MobileNetV3 deep learning model.

This project explores how daily school check-ins could support early awareness of students' emotional wellbeing and encourage timely intervention. Facial-expression predictions are intended as supportive indicators, not as a means of diagnosing depression or determining a student's actual emotional state.

## Quick start (Windows + Docker)

Run the app with Docker; Python, Node.js and PostgreSQL are supplied inside the containers.

**Prerequisites:** Git, access to this private repository, and Docker Desktop running with Linux containers, WSL2 and NVIDIA GPU support. Model inference requires a compatible NVIDIA GPU and driver. See [Docker's GPU setup](https://docs.docker.com/desktop/features/gpu/).

### 1. Clone the repository

```powershell
git clone https://github.com/azfario/EmoFace.git
Set-Location EmoFace
```

### 2. Supply the model files

Privately copy the three authorized model files into these paths:

```text
models/faceattend_best.pt
models/external/insightface/models/buffalo_l/det_10g.onnx
models/external/insightface/models/buffalo_l/w600k_r50.onnx
```

Model weights are supplied separately and are excluded from Git. Third-party face weights have separate [InsightFace model-use terms](https://github.com/deepinsight/insightface#license).

### 3. Start the app

Open Docker Desktop and wait for its engine to be running, then run:

```powershell
.\start.ps1
```

The launcher builds the containers, checks CUDA model inference, creates or reuses protected secrets, sets up PostgreSQL and waits for healthy services. First use downloads dependencies and builds large CUDA images; allow sufficient disk space and time. It opens [http://127.0.0.1:18080/app/](http://127.0.0.1:18080/app/) when ready. **No login or password prompt is required.**

Choose your camera, click **Start camera**, and confirm adult consent to enable analysis. Saved enrollments are available to anyone who can access the app on this PC.

To stop while keeping your data and secrets:

```powershell
.\start.ps1 -Stop
```

For the next run, start Docker Desktop and run `.\start.ps1` again. If an older app stack is already running, stop it before launching an updated build. Custom paths/project names must be reused when stopping or restarting; see [Advanced development](#advanced-development).

## Application and architecture

The current application is a local, consenting-adult research prototype. It supports encrypted face enrollment, optional candidate matching and continuous visible-expression prediction. Automated student attendance and school deployment remain future goals.

```text
Browser camera / image → React + TypeScript → local FastAPI
                                              ├─ SCRFD face detection → ArcFace candidate diagnostics
                                              ├─ face crop → MobileNetV3-Small expression prediction
                                              └─ PostgreSQL consent metadata + encrypted local references
```

- Camera analysis starts after the operator starts the preview and confirms adult consent. A face box carries `Emotion: <category>` and an uncalibrated-prediction note.
- Each browser analysis loop waits for its current request before starting the next, with at most 20 starts per second and five-minute analysis sessions. The backend serializes model access. Actual speed depends on capture, encoding and inference; camera FPS has not been benchmarked.
- Expression prediction, candidate matching and face outlines have separate controls. Blurry-frame expression analysis warns of uncertainty and skips identity matching; enrollment retains its original quality gates.
- The app opens without a login. All local users can manage saved enrollments; temporary browser sessions still enforce CSRF protection and request limits.
- Stop camera, Clear workspace, consent withdrawal and hidden tabs stop capture. Images and prediction results remain transient; the application records no video or individual expression history.

The backend uses CUDA PyTorch and ONNX Runtime GPU, FastAPI and SQLAlchemy/Alembic. The frontend uses React, TypeScript and Vite. Runtime APIs, migrations and model interfaces are retained alongside the ML research code.

## Model development and results

Two ImageNet-pretrained MobileNetV3-Small variants were fine-tuned on seven RAF-DB expression categories: surprise, fear, disgust, happy, sad, angry and neutral. Seed-42 stratification produced **10,430 training / 1,841 validation / 3,068 held-out** images. Both variants used the same manifests, seven-logit head, 224×224 RGB preprocessing, ImageNet normalization, AdamW, CUDA AMP and validation-based early stopping. The augmented variant added modest affine and brightness/contrast transformations.

| Experiment | Best epoch | Validation accuracy | Validation macro F1 | Held-out accuracy | Held-out macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained baseline | 16 | 76.75% | 0.6795 | 77.35% | 0.6647 |
| Pretrained + augmentation | 27 | 79.52% | 0.7254 | **79.66%** | **0.7035** |

The augmented checkpoint was selected by validation macro F1, with accuracy and loss as tie-breakers, **before held-out evaluation**. Checkpoint hashes, split fingerprints, resumable optimizer/RNG state and cached evaluation receipts support reproducibility. These are preserved historical results from one run per variant, not newly repeated experiments or webcam accuracy estimates.

The selected model's lowest held-out per-class F1 values were disgust **0.4498** (160 images) and fear **0.5674** (74 images). Exact file overlap across splits was checked; original per-image annotations/checksums were unavailable, approximate duplicate candidates remain, and identity-level leakage cannot be ruled out. A single seed does not establish statistical significance.

Inspect the [baseline training notebook](notebooks/02_M0_Pretrained_Baseline_Training.ipynb), [augmented training notebook](notebooks/03_M1_Pretrained_Augmented_Training.ipynb) and [comparison/export notebook](notebooks/04_Model_Comparison_and_Export.ipynb). Published notebooks contain source without captured outputs. To reproduce research, supply an authorized dataset at the paths in [the configuration](configs/common.yaml), inspect its provenance/consent settings for your own data, then run notebooks 01–04 in order. This setup does not grant dataset permissions. [The result validator](scripts/validate_results.py) reconciles retained local receipts without repeating held-out inference.

## Privacy and limitations

Identity matching remains **Unknown / unverified**: candidate similarity has no validated acceptance threshold. Expression scores are uncalibrated visible-category predictions and do not establish actual feelings, wellbeing or diagnosis. This prototype is not a clinical, liveness or automated attendance decision system.

Enrollment requires adult consent and encrypted-reference retention consent, with a maximum 30-day retention period and revoke/delete controls. The default app has no operator login: anyone who can access it on this PC can view, enroll, revoke or delete retained records. It is intended for a trusted local PC, with loopback and same-origin restrictions. Temporary browser sessions expire after 15 minutes; the interface reconnects automatically after clearing temporary state. Cookies, CSRF checks and rate limits protect requests, but do not authenticate a person. Optional password mode is available with `FACEATTEND_OPERATOR_LOGIN_REQUIRED=true` (override the Compose API setting for Docker). Retained names/consent metadata and encrypted references stay on the operator's device. Administrators, compromised devices and backups remain outside that protection.

Participant images, templates, credentials, keys, datasets, weights, database files, local documentation, handoffs and reports are excluded from publication. Original research artifacts remain preserved locally. This repository cleanup changes presentation and file tracking, not runtime behavior or the implemented project scope.

## Advanced development

The sections below cover native development, database administration and detailed Docker operations. They are optional for the Docker quick start.

<details>
<summary>Native Python/Node environment, PostgreSQL configuration and tests</summary>

### Native installation and build

Use this workflow when developing or training directly on Windows. To run the app through Docker, use [Quick start](#quick-start-windows--docker).

The verified stack is **Windows, Python 3.12.14, PyTorch 2.13.0+cu126, ONNX Runtime GPU 1.20.2, Node 22.13.0/npm 10.9.2**, with an NVIDIA RTX 3060 Laptop GPU. Other hardware/installations must be verified independently; model inference requires CUDA.

From an existing checkout, create the native development environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -c constraints.txt insightface==1.0.1
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name faceattend-gpu --display-name 'Python (FaceAttend GPU)'
.\.venv\Scripts\python.exe scripts\verify_gpu.py
npm.cmd --prefix frontend ci --ignore-scripts --no-fund
npm.cmd --prefix frontend run build
```

InsightFace is installed separately to avoid pulling in CPU `onnxruntime`; the existing environment uses `onnxruntime-gpu`. Its metadata can therefore produce an expected `pip check` complaint about the CPU package. Check [PyTorch wheels](https://pytorch.org/get-started/previous-versions/) and [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for the documented stack.

[requirements.txt](requirements.txt) installs app/ML dependencies; [requirements-dev.txt](requirements-dev.txt) adds tests and notebook tools. Both use the same tested pins in [constraints.txt](constraints.txt). For an app-only environment, use `-r requirements.txt` in place of `-r requirements-dev.txt`; the CUDA and separate InsightFace steps still apply. `pyproject.toml` retains the Python project/package configuration.

Privately transfer the existing authorized model assets into these paths:

```text
models/faceattend_best.pt
models/external/insightface/models/buffalo_l/det_10g.onnx
models/external/insightface/models/buffalo_l/w600k_r50.onnx
```

Weights, datasets and database storage are excluded from Git. Third-party face weights have separate [InsightFace model-use terms](https://github.com/deepinsight/insightface#license). Cloning transfers source, not those assets, participant data or permissions.

### Native PostgreSQL and protected storage

Use an isolated local PostgreSQL cluster, bound to `127.0.0.1`; the verified laptop uses port **55432**. Install PostgreSQL separately if needed. For a new empty cluster, initialize with a password prompt and SCRAM authentication; never initialize an existing database directory:

```powershell
$faceattendPgBin = 'C:\Program Files\PostgreSQL\18\bin' # Adjust to your installation.
$faceattendPgData = Join-Path $env:LOCALAPPDATA 'FaceAttend\postgresql\data'
& "$faceattendPgBin\initdb.exe" -D $faceattendPgData -U faceattend_admin -W -A scram-sha-256 --encoding=UTF8
```

Before starting, set this new cluster's `postgresql.conf` to `listen_addresses='127.0.0.1'`, `port=55432`, `password_encryption='scram-sha-256'`, `log_statement='none'`, `log_min_error_statement='panic'` and `log_parameter_max_length_on_error=0`. Restrict `pg_hba.conf` to local SCRAM connections (`host all all 127.0.0.1/32 scram-sha-256`) and protect storage with your NTFS ACL. Use an unused port and do not alter an unrelated/shared server.

```powershell
& "$faceattendPgBin\pg_ctl.exe" -D $faceattendPgData -l (Join-Path (Split-Path $faceattendPgData) 'postgresql.log') -w start
& "$faceattendPgBin\psql.exe" -h 127.0.0.1 -p 55432 -U faceattend_admin -d postgres -W
```

Inside this identified project cluster, create separate owner/runtime roles. Enter passwords with `\password`:

```sql
CREATE ROLE faceattend_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
CREATE ROLE faceattend_runtime LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
\password faceattend_migrator
\password faceattend_runtime
CREATE DATABASE faceattend_research OWNER faceattend_migrator;
REVOKE ALL ON DATABASE faceattend_research FROM PUBLIC;
GRANT CONNECT ON DATABASE faceattend_research TO faceattend_migrator,faceattend_runtime;
\connect faceattend_research
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO faceattend_runtime;
\quit
```

In PowerShell, enter the migration URL through a hidden prompt. Use `postgresql+psycopg://faceattend_migrator:<URL-encoded password>@127.0.0.1:55432/faceattend_research` with your own password; do not place credentials in commands or Git:

```powershell
$env:FACEATTEND_MIGRATION_DATABASE_URL = [System.Net.NetworkCredential]::new('', (Read-Host 'Migration-owner URL' -AsSecureString)).Password
.\.venv\Scripts\python.exe -m alembic -c alembic.ini upgrade head
.\.venv\Scripts\python.exe scripts\phase4b_migrate.py
Remove-Item Env:\FACEATTEND_MIGRATION_DATABASE_URL
```

The enrollment helper grants access only to its `demo` metadata table. Connect to `faceattend_research` as `faceattend_migrator` using interactive `psql -W` and grant aggregate access separately:

```sql
GRANT SELECT, INSERT, UPDATE ON public.research_daily_operation_stats TO faceattend_runtime;
```

The runtime role must have no aggregate DELETE/TRUNCATE, schema CREATE or migration-table privileges. Anonymous aggregate telemetry remains disabled by default. Database integration tests require separately configured disposable test roles/database; ordinary source tests do not use your retained database.

Create fresh operator credentials and an encryption key outside the checkout and OneDrive:

```powershell
$faceattendPrivate = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase2\private\phase4b'
$env:FACEATTEND_ENROLLMENT_GALLERY_DIR = Join-Path $faceattendPrivate 'gallery'
$env:FACEATTEND_ENROLLMENT_KEY_FILE = Join-Path $faceattendPrivate 'secrets\gallery.key'
$env:FACEATTEND_OPERATOR_AUTH_FILE = Join-Path $faceattendPrivate 'secrets\operator.secret'
.\.venv\Scripts\python.exe scripts\phase4b_admin.py setup --gallery-dir $env:FACEATTEND_ENROLLMENT_GALLERY_DIR --key-file $env:FACEATTEND_ENROLLMENT_KEY_FILE --auth-file $env:FACEATTEND_OPERATOR_AUTH_FILE
```

The helper prompts for your chosen password, protects the files and refuses to overwrite existing secrets. [.env.example](.env.example) documents settings; the app does not automatically load `.env`.

### Native launch and tests

In the same shell, enter your runtime URL using `faceattend_runtime`, then start the single-worker loopback app:

```powershell
$env:FACEATTEND_ENROLLMENT_DATABASE_URL = [System.Net.NetworkCredential]::new('', (Read-Host 'Runtime URL' -AsSecureString)).Password
$env:FACEATTEND_ENROLLMENT_ENABLED = 'true'
$env:FACEATTEND_FRONTEND_ENABLED = 'true'
$env:FACEATTEND_FACE_ANALYSIS_ENABLED = 'true'
$env:FACEATTEND_CANDIDATES_ENABLED = 'true'
$env:FACEATTEND_DB_TELEMETRY_ENABLED = 'false'
.\.venv\Scripts\python.exe scripts\phase3_serve.py
```

Open `http://127.0.0.1:8000/app/`, select a camera and confirm adult consent. Login is disabled by default when enrollment is enabled. Optional password mode uses `FACEATTEND_OPERATOR_LOGIN_REQUIRED=true` and the configured operator record. Local API exploration is available at `/docs`; its bundled Swagger assets remain part of the application. No camera is opened by installation or tests.

Run source-checkout checks:

```powershell
.\.venv\Scripts\python.exe scripts\run_checks.py --frontend
```

The source-only publication checks passed **501 Python tests**, with **59 optional tests skipped**, and **50 frontend tests**, plus TypeScript and the production build. The original environment also completed **560 Python tests** with explicitly configured GPU/database/private-input integrations. Fresh-PC installation has not been independently tested. The check launcher disables private-input/database/saved-model integration flags, creates ignored local output directories and needs no historical handoffs.

</details>

<details>
<summary>Docker options, deployment details, backups, troubleshooting and integration tests</summary>

### Docker deployment details and verification

GPU Docker deployment is implemented and tested on the original laptop, including an isolated PostgreSQL database, protected operator/encryption secrets, CUDA-only model inference and a localhost browser interface. The Docker acceptance suite passed **14 tests**. The owner also confirmed the Camo → OBS Virtual Camera feed, adult-consent gating, successive face-box/Emotion updates, Stop camera and End session through the Docker-served page.

The physical-camera run included three HTTP refusals and two overlapping HTTP handlers; their exact causes were not established. Existing client serialization and backend model-gate tests passed, but this does not prove perfect request serialization across the observed run. No camera FPS or prediction-accuracy claim is made. A fresh installation on a second PC remains unverified.

Docker is an alternative runtime; it does not modify the native Windows environment or its database. Use Docker Desktop with **Linux containers, WSL2 and NVIDIA GPU support**. The tested laptop has a 6 GB RTX 3060 GPU. Plan for **40 GB of free disk headroom for a first CUDA image build**, in addition to existing Docker storage, and monitor space during export. The successful laptop retry began with 41.2 GiB free and already cached CUDA layers; this is not a measured fresh-install minimum. Build downloads require internet; the running application blocks new outbound connections and DNS. See [Docker's GPU prerequisites](https://docs.docker.com/desktop/features/gpu/) and [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).

#### Launcher options and retained data

Clone the repository using an account with access, start Docker Desktop's Linux engine, and supply the three authorized model files at the [model paths above](#2-supply-the-model-files). From the checkout run:

```powershell
.\start.ps1
```

The [launcher](start.ps1) requires no local Python, Node, PyTorch or PostgreSQL installation. It builds the app image, checks actual CUDA model inference, creates the five secret files without a password prompt through a temporary container, builds the frontend and waits for the app to become healthy at `http://127.0.0.1:18080/app/`. The browser then opens; camera permission, activation and adult consent remain manual. First use downloads the pinned container dependencies and can take time.

The login-free update passed 546 backend regression tests (64 optional integrations skipped), 44 final focused checks, 53 frontend tests, TypeScript and the production build. An isolated Docker startup and synthetic-image acceptance check confirmed no password prompt, automatic local access, CUDA readiness, consent enforcement, CSRF/Origin checks and workspace clearing. The changed login-free build has not been rechecked with a physical camera; the earlier camera acceptance remains historical evidence.

The legacy operator record is retained for compatibility and optional password mode; fresh launcher setup generates a random dormant record. Existing secrets and encrypted data are preserved.

Default secrets are retained under `%LOCALAPPDATA%\FaceAttend\phase6\private\operator_secrets`, outside the checkout and OneDrive with restricted Windows ACLs. Subsequent runs reuse those secrets and the same Compose data volumes. Existing incomplete/insecure secrets, missing keys beside retained data, remote Docker contexts, unavailable models/GPU and conflicting ports are refused. No automatic key replacement, CPU fallback, data reset or volume deletion occurs. If the selected project is already running, the launcher leaves it running and reports its URL; stop the whole stack before rebuilding.

For existing installations or model files stored elsewhere, use explicit paths and the **same project name and matching secrets**:

```powershell
.\start.ps1 -ModelsDirectory 'D:\EmoFaceModels' -SecretsDirectory 'C:\Private\EmoFaceSecrets' -ProjectName 'emoface_local' -NoBrowser
```

`EMOFACE_MODELS_DIR`, `EMOFACE_SECRETS_DIR` and `COMPOSE_PROJECT_NAME` are also honored when their corresponding parameters are omitted. A successful launch leaves those variables set in the current PowerShell process, so `docker compose down` in that window also works. To stop from a new window, use `./start.ps1 -Stop` with the same project/path parameters as your launch; no volumes or secrets are removed. An unsuccessful launch keeps created secrets/volumes for recovery and refuses to call a partially started stack ready. Stop that selected stack before retrying. The launcher checks local setup on PC 1; independent second-PC reproducibility still needs validation.

```text
Browser → 127.0.0.1:18080 → Nginx frontend
                               │ shared Compose network namespace
                               ├─ 127.0.0.1:8000 FastAPI, one GPU worker
                               └─ 127.0.0.1:5432 isolated PostgreSQL 18
                                  │
                           metadata volume + encrypted-reference volume
Operator secret files → read-only mounts → private temporary key copies
Authorized model files → read-only mount; absent from images
```

Only Nginx publishes a host port, bound to IPv4 loopback. The shared namespace preserves the backend's existing loopback, Host/Origin and CSRF checks without trusting forwarded client headers. A one-shot network guard has `NET_ADMIN` solely inside this namespace, drops new outbound IPv4/IPv6 traffic and embedded DNS, and allows loopback and established replies. The API checks the guard's namespace marker before startup. This is local application isolation, not protection from an administrator controlling Docker or the host.

The [Compose manifest](compose.yaml) creates separate database, encrypted-reference and network-state volumes. It never mounts the native PostgreSQL cluster at port 55432. Only the API reserves a GPU; frontend/API run as non-root, and the database initialization and namespace guard use narrowly scoped elevated container privileges. Images use digest-pinned bases, a [separate complete Linux package lock](docker/requirements-linux-lock.txt), read-only runtime filesystems and explicit source-only build inputs. The Linux loader path includes the installed CUDA/cuDNN libraries; Windows constraints remain unchanged. Face-model CPU fallback remains disabled, and health requires actual model and registration-database readiness.

#### Manual Compose workflow

The launcher above is the Docker-only path. For operators who prefer the existing manual workflow, the host secret helper requires the native Python dependencies described earlier. From the checkout, create **new** protected Compose secrets; the helper prompts for your chosen operator password and refuses an existing directory:

```powershell
$env:EMOFACE_MODELS_DIR = Join-Path (Get-Location) 'models'
$env:EMOFACE_SECRETS_DIR = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase6\private\operator_secrets'
.\.venv\Scripts\python.exe docker\setup_secrets.py --directory $env:EMOFACE_SECRETS_DIR
docker version
docker compose version
docker compose config --quiet
docker compose build api frontend
docker compose up -d frontend network_guard db
docker compose run --rm --no-deps api python docker/gpu_check.py
docker compose up -d --wait --wait-timeout 180
```

The GPU check runs the mounted saved models on synthetic tensors and requires exclusive CUDA computation. Stop if it fails; do not substitute a CPU service. One-shot migrations use the owner credential; the API receives only the restricted runtime database password, operator record and encryption key. Keep all five generated secret files outside the checkout and OneDrive, under their protected ACL. No default production password is supplied. `--qa` creates a separately scoped fictional test fixture only; it is not an operator setup shortcut.

Open `http://127.0.0.1:18080/app/`; no login is required. Start the camera yourself and confirm adult consent; the existing automatic expression/outline workflow then operates as before. The container tests do not request camera permission. Enrollment requires 3–5 distinct acceptable images and encrypted-reference consent. Candidate names remain possible matches, and expressions remain uncalibrated predictions.

To stop, preserve data and restart later:

```powershell
docker compose down
docker compose up -d --wait --wait-timeout 180
```

Use a whole-stack `down`/`up` when changing or rebuilding the frontend namespace owner. Do not restart or recreate the frontend alone while backend/database services are retained in its old namespace. After a source update, run `docker compose down`, `docker compose build api frontend`, then `docker compose up -d --wait --wait-timeout 180`. An API-only restart retains data and invalidates in-memory operator sessions.

**Backup and retention:** stop the stack before making a consistent, access-controlled backup of its named database and encrypted-reference volumes. Back up the protected host secret directory independently; key loss makes templates unreadable. The API's key copies are temporary memory files, while non-secret credential fingerprints prevent accidental rotation. Restoring requires the matching database, encrypted references and secrets as one coordinated set, the same compatible model assets, and a fresh GPU/readiness check. Backups can retain deleted identities; apply the same consent and maximum 30-day retention policy to them. The app provides revoke/delete and expiry cleanup, but does not promise secure erasure from SSDs or backups. Secret rotation and backup restoration are manual operations, not automatic features.

`docker compose down --volumes` would erase Compose-owned data; it is deliberately not part of setup, tests or normal shutdown. Do not use pruning as a troubleshooting shortcut. PostgreSQL logs omit statements and bind values, and HTTP access logging is disabled. Models, photographs, keys, credentials, handoffs and reports do not enter the build context or final images.

For troubleshooting, use `docker compose ps -a` and coarse service health. Verify the phone/OBS source separately if browser preview fails. If CUDA fails, check WSL2, Docker's GPU device access, driver and the pinned CUDA 12/cuDNN 9 environment using the synthetic GPU checker. If database initialization/migrations fail, verify the scoped secret mounts and volume ownership; preserve volumes and never point Compose at the native cluster. If port 18080 is occupied, stop the unrelated conflict only with its owner's approval; changing the port requires updating both Compose and the proxy's exact Host/Origin allowlist.

Ordinary source tests include static container-contract checks. Live Compose tests in [tests/phase6](tests/phase6/) are explicitly opt-in and restricted to the disposable `emoface_phase6qa` project. To run those checks, stop the normal stack first so port 18080 is free, then create a new QA secret directory and isolated project:

```powershell
$env:COMPOSE_PROJECT_NAME = 'emoface_phase6qa'
# First QA setup only; later runs must reuse the matching existing secret directory.
$env:EMOFACE_SECRETS_DIR = Join-Path $env:LOCALAPPDATA ('FaceAttend\phase6\private\qa_' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe docker\setup_secrets.py --directory $env:EMOFACE_SECRETS_DIR --qa
docker compose build api frontend
docker compose up -d --wait --wait-timeout 180
$env:EMOFACE_PHASE6_LIVE = '1'
.\.venv\Scripts\python.exe -m pytest -q tests/phase6 -o junit_family=legacy
docker compose down
Remove-Item Env:\EMOFACE_PHASE6_LIVE
Remove-Item Env:\COMPOSE_PROJECT_NAME
```

The adult-image test skips unless `EMOFACE_PHASE6_INPUT_MANIFEST` points to a protected local JSON file containing `adult_consent: true`, `persistent_enrollment_consent: true`, a `references` array of 3–5 image paths and a `probe` image path. Supply only explicitly consented adult inputs, keep this manifest outside OneDrive/source control, and clear the variable after testing. The test deletes its isolated registrations. Never use production credentials or volumes. Restore `EMOFACE_SECRETS_DIR` to your original operator directory before returning to the normal stack. No held-out evaluation or retraining occurs during container acceptance.

</details>
