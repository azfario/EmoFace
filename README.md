# EmoFace: Early Student Emotion Detection

## Project Overview

**EmoFace** is an end-to-end machine learning project designed to automate student attendance through facial recognition while classifying facial expressions using a custom fine-tuned MobileNetV3 deep learning model.

This project explores how daily school check-ins could support early awareness of students' emotional wellbeing and encourage timely intervention. Facial-expression predictions are intended as supportive indicators, not as a means of diagnosing depression or determining a student's actual emotional state.


## Application and architecture

The current application is a local, consenting-adult research prototype. It supports encrypted face enrollment, optional candidate matching and continuous visible-expression prediction. Automated student attendance and school deployment remain future goals.

```text
Browser camera / image → React + TypeScript → local FastAPI
                                              ├─ SCRFD face detection → ArcFace candidate diagnostics
                                              ├─ face crop → MobileNetV3-Small expression prediction
                                              └─ PostgreSQL consent metadata + encrypted local references
```

- Camera analysis starts after the operator starts the preview and confirms adult consent. A face box carries `Emotion: <category>` and an uncalibrated-prediction note.
- One request runs at a time, with at most 20 starts per second and five-minute analysis sessions. Actual speed depends on capture, encoding and inference; camera FPS has not been benchmarked.
- Expression prediction, candidate matching and face outlines have separate controls. Blurry-frame expression analysis warns of uncertainty and skips identity matching; enrollment retains its original quality gates.
- Stop camera, End session, consent withdrawal and hidden tabs stop capture. Images and prediction results remain transient; the application records no video or individual expression history.

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

## Install and build

The verified stack is **Windows, Python 3.12.14, PyTorch 2.13.0+cu126, ONNX Runtime GPU 1.20.2, Node 22.13.0/npm 10.9.2**, with an NVIDIA RTX 3060 Laptop GPU. Other hardware/installations must be verified independently; model inference requires CUDA.

Clone using your own GitHub account and recreate the environment:

```powershell
git clone https://github.com/azfario/EmoFace.git
Set-Location EmoFace
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
models/external/insightface/buffalo_l/det_10g.onnx
models/external/insightface/buffalo_l/w600k_r50.onnx
```

Weights, datasets and database storage are excluded from Git. Third-party face weights have separate [InsightFace model-use terms](https://github.com/deepinsight/insightface#license). Cloning transfers source, not those assets, participant data or permissions.

## Configure PostgreSQL and the operator

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

## Run and test

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

Open `http://127.0.0.1:8000/app/`, sign in with your chosen operator password, select a camera and confirm adult consent. Local API exploration is available at `/docs`; its bundled Swagger assets remain part of the application. No camera is opened by installation or tests.

Run source-checkout checks:

```powershell
.\.venv\Scripts\python.exe scripts\run_checks.py --frontend
```

The source-only publication checks passed **501 Python tests**, with **59 optional tests skipped**, and **50 frontend tests**, plus TypeScript and the production build. The original environment also completed **560 Python tests** with explicitly configured GPU/database/private-input integrations. Fresh-PC installation has not been independently tested. The check launcher disables private-input/database/saved-model integration flags, creates ignored local output directories and needs no historical handoffs.

## Privacy and limitations

Identity matching remains **Unknown / unverified**: candidate similarity has no validated acceptance threshold. Expression scores are uncalibrated visible-category predictions and do not establish actual feelings, wellbeing or diagnosis. This prototype is not a clinical, liveness or automated attendance decision system.

Enrollment requires adult consent and encrypted-reference retention consent, with a maximum 30-day retention period and revoke/delete controls. Operator sessions expire after 15 minutes; same-origin cookies and CSRF protection protect local operations. Retained names/consent metadata and encrypted references stay on the operator's device. Administrators, compromised devices and backups remain outside that protection.

Participant images, templates, credentials, keys, datasets, weights, database files, local documentation, handoffs and reports are excluded from publication. Original research artifacts remain preserved locally. This repository cleanup changes presentation and file tracking, not runtime behavior or the implemented project scope.
