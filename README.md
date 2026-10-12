# FaceAttend: Early Studen Emotion Detection

## Project Overview

**FaceAttend** is an end-to-end machine learning project designed to automate student attendance through facial recognition while classifying facial expressions using a custom fine-tuned MobileNetV3 deep learning model.

This project explores how daily school check-ins could support early awareness of students' emotional wellbeing and encourage timely intervention. Facial-expression predictions are intended as supportive indicators, not as a means of diagnosing depression or determining a student's actual emotional state.

## Current Progress

**Last updated: 12 October 2026.** FaceAttend currently runs as a local, consenting-adult research prototype. The machine learning, API, database and interface implementation is complete through Phase 5, including the confirmed live-camera and Stop/End session checks. Automated student attendance and school deployment are future project goals.

| Phase | Implemented work | Status |
| --- | --- | --- |
| 1 | RAF-DB experiments, MobileNetV3-Small fine-tuning, validation-based model selection and export | Complete |
| 2 / 2A | CUDA face detection, encrypted verification diagnostics and local webcam capture | Complete |
| 3 | Local FastAPI model-serving endpoints | Complete |
| 4 | PostgreSQL integration and anonymous aggregate telemetry | Complete |
| 4B | Consented adult enrollment, encrypted references, expiry, revocation and deletion | Complete |
| 4C | Optional candidate matching with unverified results | Complete |
| 4D | Combined per-face detection, candidate diagnostics and expression prediction | Complete |
| 5 | React/TypeScript interface and continuous camera analysis | Complete |

Phase 6 has not started.

## What Works Now

- Local operator sign-in, service readiness and adult enrollment management.
- SCRFD face detection and ArcFace verification/candidate diagnostics using the existing laptop CUDA environment.
- MobileNetV3-Small visible-expression predictions for seven categories: surprise, fear, disgust, happy, sad, angry and neutral.
- Automatic camera analysis after the user starts the camera and confirms adult consent. Each new frame follows the previous completed analysis, with no five-second pause.
- A detected face box with **`Emotion: <category>`** above it and an **uncalibrated prediction** note.
- One analysis request at a time, capped at 20 starts per second and five-minute sessions. Actual update speed depends on capture, encoding and GPU inference.
- Optional expression analysis on blurry frames when a usable face remains detectable. The interface shows an uncertainty warning and skips identity matching on those frames.
- Separate choices for expression prediction, possible identity matches and the face outline. Uploaded images use manual analysis.

The camera operates on one voluntarily participating adult at a time. Stop controls, consent withdrawal, navigation, hidden tabs, camera loss and session end stop new captures. The interface records no video, attendance decisions or individual prediction history.

## Results and Validation

The selected **M1 pretrained + augmentation** model achieved **79.66% held-out accuracy** and **0.7035 macro F1** on the RAF-DB experiment. Model selection was frozen using validation results before held-out evaluation. These historical dataset results do not establish webcam performance. See [experiment results and limitations](docs/RESULTS.md).

The latest verified implementation has:

- **560 passing Python tests**, including all 528 previous baseline cases and 32 Phase 5 cases.
- **50 passing frontend tests**.
- Passing TypeScript checks and production frontend build.
- Passing actual CUDA/PostgreSQL functional checks, including the expression-only blur exception and live-analysis requests.

The user confirmed that the live camera works, Stop camera turns off the preview and End session signs out. Temporary demo state was cleaned up and normal PostgreSQL data and privileges were verified unchanged. Camera FPS has not been benchmarked; the frontend's update ceiling is not a measured camera FPS result.

## Interpretation and Privacy

Identity results remain **Unknown / unverified**. Candidate similarity has no validated acceptance threshold, and a high-ranked name does not establish identity.

The on-screen `Emotion:` label is a visible-expression model prediction. Scores are uncalibrated and do not determine a person's actual emotional state, wellbeing or diagnosis. FaceAttend is not a clinical assessment or liveness system.

Encrypted enrollment references have explicit consent and a bounded retention period. Operator sessions last 15 minutes. The application uses local processing, same-origin authentication and CSRF protection; anonymous telemetry is off by default.

Participant photos, biometric templates, keys, credentials, datasets, model weights, database storage, local handoffs and reports are kept out of this repository. Notebook publication copies contain source without captured outputs; the original executed notebooks and experimental artifacts remain preserved locally.

## Technology and Local Setup

The verified development environment is Windows, Python 3.12.14, PyTorch 2.13.0+cu126, ONNX Runtime GPU 1.20.2 and an NVIDIA RTX 3060 Laptop GPU. The interface uses React, TypeScript and Vite with Node 22.13.0/npm 10.9.2. The backend uses FastAPI and PostgreSQL.

Inference requires the configured CUDA environment and authorized local model assets. A source checkout does not include the trained checkpoint, third-party face weights, dataset or operator credentials.

To clone and continue development on another Windows/NVIDIA PC, start with the [new-PC setup guide](docs/SETUP_NEW_PC.md). It covers recreating `.venv`, installing the pinned dependencies, source-checkout tests, local model assets and fresh database/operator setup. Copy your retained model checkpoint separately; cloning does not transfer it, your dataset or your existing database.

After installing dependencies, the development check command is:

```powershell
.\.venv\Scripts\python.exe scripts\run_checks.py --frontend
```

This command works without the original laptop's handoffs or preservation reports. Private-image, saved-model and database integration checks require their own local setup; they are not enabled by this command.

The prepared source-only export was checked using the existing environment: **501 Python tests passed, 59 optional tests skipped, and 50 frontend tests passed**, with TypeScript checks and the frontend build passing. Installation and hardware setup on another physical PC still need to be verified there.

Additional setup references:

- [GPU setup](docs/GPU_SETUP.md)
- [Face-model and verification setup](docs/phase2/README.md)
- [FastAPI guide](docs/phase3/API_GUIDE.md)
- [PostgreSQL setup on Windows](docs/phase4/POSTGRES_SETUP_WINDOWS.md)
- [Enrollment lifecycle](docs/phase4b/ENROLLMENT_GUIDE.md)
- [Local interface and camera guide](docs/phase5/UI_GUIDE.md)

Build the frontend from the repository root:

```powershell
npm.cmd --prefix frontend ci --ignore-scripts --no-fund
npm.cmd --prefix frontend run build
```

The enabled FastAPI application serves the built interface at `/app/` on its loopback origin. Follow the interface guide to configure protected local credentials and start the service.

## Repository Layout

```text
configs/             Experiment and face-model configuration
notebooks/           Training and verification notebook source
src/                 ML pipelines, FastAPI services and database modules
frontend/            React/TypeScript local interface
migrations*/         PostgreSQL migrations
scripts/             Setup, execution, validation and lifecycle helpers
tests/               Backend and frontend integration support
docs/                New-PC setup, feature guides and experiment context
requirements-dev.txt Fresh source-checkout Python dependency entry point
```

Phase 5 acceptance and temporary-demo cleanup are complete. Subsequent phases require separate approval; this progress update does not claim the complete attendance system is finished.
