# Continue development on another PC

This guide targets **Windows with an NVIDIA CUDA-capable GPU**, matching the tested project. Create a new environment on the new PC; the original `.venv` is tied to its base interpreter and cannot be moved by cloning Git. Other platforms have not been validated for this complete application.

The repository contains code, tests, configuration, migrations, dependency files, setup guides and notebook source. Local handoffs, reports, executed notebook outputs, datasets, checkpoints, biometric references, passwords, native database storage and caches are excluded. A clone can be used to edit and test the source; running saved-model inference also requires the separately retained model assets and local setup below.

## 1. Clone into your own folder

Install Git, Python 3.12, Node.js (the verified version is 22.13.0) and an appropriate NVIDIA driver. PostgreSQL is needed for the enrollment application and optional database tests. Authenticate Git normally using your own GitHub account because the repository is private.

```powershell
git clone https://github.com/azfario/FaceAttend-Early-Studen-Emotion-Detection.git
Set-Location FaceAttend-Early-Studen-Emotion-Detection
py -3.12 -m venv .venv
```

Paths used by the application are relative to this checkout or your own LocalAppData. Do not copy the original laptop's user-directory paths into your setup.

## 2. Recreate the Python environment

The original environment uses torch 2.13.0+cu126, torchvision 0.28.0+cu126 and ONNX Runtime GPU 1.20.2. Check driver/GPU compatibility before recreating that stack. The CUDA wheel command is documented on the [official PyTorch version page](https://pytorch.org/get-started/previous-versions/); ORT's CUDA/cuDNN compatibility is documented in the [official CUDA provider guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

Run these commands in the **new** virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -c requirements-phase5-constraints.txt -r requirements-phase2-insightface.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name faceattend-gpu --display-name 'Python (FaceAttend GPU)'
```

InsightFace is installed separately to avoid pulling in CPU `onnxruntime`. This matches the original environment's documented metadata exception: `pip check` can report that InsightFace requires `onnxruntime` even though the application uses `onnxruntime-gpu`. Do not install both runtimes to silence that message. No GUI/face3d extras are required.

Verify the new environment:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe scripts\verify_gpu.py
```

The project requires CUDA for model inference; missing GPU support is a setup issue, not permission to switch the models to CPU. The install procedure is based on the observed package versions and official wheel documentation; dependency installation on a different physical PC must still be verified there.

## 3. Install and build the interface

```powershell
npm.cmd --prefix frontend ci --ignore-scripts --no-fund
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
```

`node_modules` and `frontend/dist` are recreated locally. They are not needed in Git.

## 4. Run development checks without historical reports

```powershell
.\.venv\Scripts\python.exe scripts\run_checks.py
# Or include frontend tests/build:
.\.venv\Scripts\python.exe scripts\run_checks.py --frontend
```

This command creates empty local output directories and runs the existing pytest suite with private-data/database/saved-model integration flags disabled. It does not need the original laptop's preservation manifests or credentials. Optional integration cases are skipped unless separately configured; synthetic CUDA checks can run if CUDA is available. The original all-opt-in 560-pass result belongs to the tested original workspace, not a promise that every optional test runs immediately after cloning.

The prepared source-only export was tested using the existing environment: **501 Python tests passed, 59 optional tests skipped, and 50 frontend tests passed**. TypeScript checks and the frontend build also passed. This verifies the exported source without the original handoffs or reports; dependency installation and hardware setup on another physical PC remain to be checked there.

Generated reports and historical preservation launchers are not part of this source publication. Run `run_checks.py` for ordinary development. Some notebook and research helper commands generate ignored local `reports/` or `outputs/` directories; that is expected and does not commit those records.

## 5. Supply model assets separately

To reuse the trained expression model, privately transfer your existing `models/faceattend_best.pt` into the same relative location in the new checkout. Do not retrain or repeat the historical held-out evaluation merely to move the project.

The face-model paths configured in `configs/phase2_verification.yaml` are:

```text
models/external/insightface/buffalo_l/det_10g.onnx
models/external/insightface/buffalo_l/w600k_r50.onnx
```

Transfer your already-authorized local model assets separately, or follow the existing model acquisition instructions under their applicable research terms. Third-party pretrained face weights are governed separately from the Python code; see [InsightFace's official repository](https://github.com/deepinsight/insightface). No participant photos or encrypted face templates are needed to run ordinary synthetic source tests.

For training/data exploration, supply your authorized dataset separately at the paths in `configs/common.yaml`. Dataset permissions and consent do not come from cloning this repository. Original checkpoints, experimental records and dataset archives remain on the original PC unless you explicitly transfer them through an appropriate private channel.

## 6. Configure fresh local PostgreSQL and operator credentials

Use the **Fresh machine/manual setup** section of [the Windows PostgreSQL guide](phase4/POSTGRES_SETUP_WINDOWS.md) to create an empty loopback project cluster, owner/runtime/test roles, migrations and least-privilege grants. Never initialize an existing populated database directory. Do not copy passwords or database storage into Git.

The application does not automatically load `.env`; `.env.example` lists variable names and disabled defaults. Set variables explicitly in your local shell. Enter sensitive URLs with hidden prompts, not literal passwords in commands:

```powershell
$env:FACEATTEND_MIGRATION_DATABASE_URL = [System.Net.NetworkCredential]::new('', (Read-Host 'New local migration-owner URL' -AsSecureString)).Password
.\.venv\Scripts\python.exe -m alembic -c alembic.ini upgrade head
.\.venv\Scripts\python.exe scripts\phase4b_migrate.py
Remove-Item Env:\FACEATTEND_MIGRATION_DATABASE_URL
```

The migration helper expects database `faceattend_research` and role `faceattend_migrator`. Apply the Phase 4 runtime grants from the PostgreSQL guide as the owner; the enrollment helper adds the separate `demo` grants. Owner credentials are not used by the running application.

Create fresh operator secrets outside the checkout and OneDrive:

```powershell
$faceattendPrivate = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase2\private\phase4b'
$env:FACEATTEND_ENROLLMENT_GALLERY_DIR = Join-Path $faceattendPrivate 'gallery'
$env:FACEATTEND_ENROLLMENT_KEY_FILE = Join-Path $faceattendPrivate 'secrets\gallery.key'
$env:FACEATTEND_OPERATOR_AUTH_FILE = Join-Path $faceattendPrivate 'secrets\operator.secret'
.\.venv\Scripts\python.exe scripts\phase4b_admin.py setup --gallery-dir $env:FACEATTEND_ENROLLMENT_GALLERY_DIR --key-file $env:FACEATTEND_ENROLLMENT_KEY_FILE --auth-file $env:FACEATTEND_OPERATOR_AUTH_FILE
```

The helper prompts for your new operator password, protects the local files and refuses to overwrite existing secrets. This creates no enrollment. Do not reuse the original temporary-demo password as a retained credential.

Start the local app using only the runtime database URL:

```powershell
$env:FACEATTEND_ENROLLMENT_DATABASE_URL = [System.Net.NetworkCredential]::new('', (Read-Host 'New local runtime URL' -AsSecureString)).Password
$env:FACEATTEND_ENROLLMENT_ENABLED = 'true'
$env:FACEATTEND_FRONTEND_ENABLED = 'true'
$env:FACEATTEND_FACE_ANALYSIS_ENABLED = 'true'
$env:FACEATTEND_CANDIDATES_ENABLED = 'true'
$env:FACEATTEND_DB_TELEMETRY_ENABLED = 'false'
.\.venv\Scripts\python.exe scripts\phase3_serve.py
```

Open `http://127.0.0.1:8000/app/`. Sign in with the operator password you just chose. Select your camera and confirm adult consent before starting analysis. The code, model paths and UI do not depend on the original laptop username or temporary demo database.

## 7. Continue work and commit only source

Select the new `.venv` interpreter and the FaceAttend notebook kernel in VS Code. Notebook copies in Git contain source without prior outputs. Clear outputs before committing notebooks that display participant/dataset images. The ignore rules exclude local reports/handoffs, assets, secrets, database files and generated build/cache folders.

The original project is complete through Phase 5. This guide only supports continuing that source on another PC; it adds no deployment, Docker or subsequent-phase components.
