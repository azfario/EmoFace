> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# FaceAttend Phase 3 local API guide

The Phase 3 API serves the existing SCRFD, ArcFace and optional M1 visible-expression model on the laptop GPU. It accepts operator-selected still images; it never opens a camera. Comparison is diagnostic only: `verified=false`, threshold null and `THRESHOLD_NOT_CALIBRATED` are fixed response semantics.

## Start and stop

In the existing workspace's VS Code PowerShell terminal:

```powershell
# Already installed in the existing .venv; needed only to reproduce this setup.
.venv\Scripts\python.exe -m pip install -r requirements-phase3.txt

# Preferred launcher: 127.0.0.1:8000, one worker, no reload/proxies/access logs.
.venv\Scripts\python.exe scripts\phase3_serve.py
```

Startup loads the existing local models once and performs synthetic CUDA readiness checks. It never downloads model weights. Loading can take several seconds. Open **http://127.0.0.1:8000/docs** after startup. `/openapi.json` contains the generated request/response schemas. Swagger JS/CSS are served locally from the vendored official assets; no CDN or remote schema validator is needed. Swagger is documentation, not a new application frontend. Stop with **Ctrl+C**, allowing any active request to finish cleanup.

Equivalent explicit Uvicorn invocation:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache\matplotlib'
.venv\Scripts\python.exe -m uvicorn faceattend_api.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log --no-proxy-headers --limit-concurrency 8
```

Use the preferred launcher to keep its transport settings together. Do not bind to `0.0.0.0`, enable reload/multiple workers, use tunnels or publish the server. The application additionally checks loopback client/server addresses, allowed Host and same-origin browser requests. No CORS middleware or wildcard permission is enabled.

## Routes and multipart fields

| Route | Input | Result |
|---|---|---|
| `GET /health` | None | Truthful per-feature readiness and GPU-required/GPU-ready fields; 503 if an enabled feature is unavailable. |
| `GET /api/v1/models/info` | None | Architecture names, classifier label order, readiness and GPU label; no local paths or model contents. |
| `POST /api/v1/face/detect` | `image`, `adult_consent=true`, `research_opt_in=true` | Single usable face, quality metadata and CUDA device status. |
| `POST /api/v1/face/embedding-info` | Same fields | Dimension 512, L2 norm and quality metadata; no vector. |
| `POST /api/v1/expression/predict` | Same fields | Visible expression label, uncalibrated softmax confidence and interpretation limitation. |
| `POST /api/v1/face/compare-diagnostic` | `claimed_id`, 3–5 repeated `enrollment_images` parts, distinct `probe`, both consent/opt-in fields | Selected-reference cosine, quality metadata and fixed uncalibrated refusal of operational verification. |

Consent fields are exactly the lowercase string `true` in multipart form data. Their scope is the route's documented research use: identity/detection processing for face routes and visible-expression processing for expression prediction. The operator must confirm consent from every adult pictured. The field records an assertion, not proof of consent. No previous enrollment is assumed or retained. The claimed ID must match `volunteer_[a-z0-9_]{3,40}`; it is intentionally not reflected in the response.

## Operator examples

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/models/info

$img = Read-Host 'Authorized adult research image path'
curl.exe --silent --show-error -X POST 'http://127.0.0.1:8000/api/v1/face/detect' -F "image=@$img" -F 'adult_consent=true' -F 'research_opt_in=true'
curl.exe --silent --show-error -X POST 'http://127.0.0.1:8000/api/v1/face/embedding-info' -F "image=@$img" -F 'adult_consent=true' -F 'research_opt_in=true'
curl.exe --silent --show-error -X POST 'http://127.0.0.1:8000/api/v1/expression/predict' -F "image=@$img" -F 'adult_consent=true' -F 'research_opt_in=true'
```

Single-request diagnostic comparison, selecting valid authorized reference captures and a distinct probe:

```powershell
$claimedId = Read-Host 'Pseudonymous volunteer ID'
$ref1 = Read-Host 'First authorized reference image path'
$ref2 = Read-Host 'Second authorized reference image path'
$ref3 = Read-Host 'Third authorized reference image path'
$probe = Read-Host 'Distinct authorized probe image path'
curl.exe --silent --show-error -X POST 'http://127.0.0.1:8000/api/v1/face/compare-diagnostic' -F "claimed_id=$claimedId" -F "enrollment_images=@$ref1" -F "enrollment_images=@$ref2" -F "enrollment_images=@$ref3" -F "probe=@$probe" -F 'adult_consent=true' -F 'research_opt_in=true'
```

Shell process arguments contain the locally selected paths while curl runs; keep the laptop/session private. Do not paste real paths, photos or response scores into public logs. Curl normally infers JPEG/PNG MIME from the extension; if a valid file has a different extension, explicitly append `;type=image/jpeg` or `;type=image/png` to that file form value. Filename extension alone never authorizes decoding.

Illustrative **synthetic** comparison response, not a measured volunteer score:

```json
{
  "status": "DIAGNOSTIC_COMPARISON_COMPLETED",
  "face_count": 1,
  "quality": {"blur_variance": 80.0, "brightness": 120.0, "face_width": 100.0, "face_height": 120.0},
  "similarity_metric": "cosine",
  "cosine_similarity": 0.5,
  "operational_verification": {"verified": false, "threshold": null, "reason": "THRESHOLD_NOT_CALIBRATED"},
  "disclaimer": "Research diagnostic only. Not identity verification or attendance approval."
}
```

## Limits, rejections and cleanup

Defaults: **8 MiB per image**, **40 MiB per whole multipart request**, **10 million decoded pixels per image**, one image for ordinary inference, and 3–5 references plus one probe for comparison. Multipart parsing is incremental and bounded in RAM, with an 8-KiB per-part header bound and 128-byte form-field bound. It does not use UploadFile disk spooling. Signature/MIME agreement, Pillow content verification and dimensions are checked before OpenCV decoding/inference; animated PNG and unsupported encodings are rejected.

The original face policy still requires exactly one detected face before QC, minimum image side 160, face side 80, valid unclipped box/five landmarks, blur variance at least **60** and brightness 35–220. No largest-face selection or automatic portrait crop is introduced. Expression prediction applies this face gate, then the full original encoded image follows Phase 1's Pillow RGB decoding, resize and normalization, including its unchanged EXIF behavior; the face-aligned ArcFace crop is not substituted into the expression model. This distinction preserves Phase 1 even when OpenCV's face-input decoder honors an EXIF orientation tag.

| HTTP | Typical safe status |
|---|---|
| 400 | `INVALID_IMAGE`, `BLACK_FRAME_REJECTED` |
| 403 | `CONSENT_REQUIRED`, `LOCAL_ONLY` |
| 408 | `REQUEST_TIMEOUT` |
| 413 | `PAYLOAD_TOO_LARGE` |
| 415 | `UNSUPPORTED_MEDIA_TYPE` |
| 422 | `INVALID_REQUEST`, `INVALID_ID`, `NO_FACE`, `MULTIPLE_FACES`, `LOW_QUALITY_IMAGE`, duplicate/inconsistent reference rejection |
| 503 | `CUDA_UNAVAILABLE`, `MODEL_NOT_READY`, `SERVICE_BUSY`, `STORAGE_ERROR` |
| 504 | `INFERENCE_TIMEOUT` |
| 500 | Sanitized `INTERNAL_ERROR` |

Example rejection: `{"status":"NO_FACE","message":"No usable face was detected."}` with HTTP 422. Uploaded values, filenames, vectors and exception details are never echoed in errors. Responses have `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

One upload/inference request is admitted at a time; a second gets `SERVICE_BUSY` without a GPU queue. Health/metadata remain available. Upload timeout is 30 seconds. Inference deadline is 60 seconds, **cooperative**: a running native GPU call cannot safely be killed from Python. The gate remains held until work/reference cleanup completes, including direct HTTP-task cancellation; a timeout is reported after that completion. A hung driver can therefore exceed the deadline.

Each comparison uses a unique temporary directory in the already established non-synced private base, an in-memory Fernet key passed directly to the existing store, and core enrollment/deletion. No environment secret is changed. References are encrypted on disk only for that request. Uploaded photos/vectors remain in RAM and are never stored in the project. Directory/template cleanup runs in `finally`; key references are dropped. This is not secure erasure and cannot guarantee cleanup after abrupt process/OS termination. No ID gallery, persistent enrollment or operational `/face/verify` route exists.

## Tests and smoke commands

```powershell
# Full preservation-safe suite; private-image Phase 3 integration disabled.
.venv\Scripts\python.exe scripts\phase3_run_tests.py

# Opt in to the existing authorized private stills and actual CUDA integration.
.venv\Scripts\python.exe scripts\phase3_run_tests.py --gpu-integration

# Real temporary Uvicorn server; synthetic input only; stops its server afterward.
.venv\Scripts\python.exe scripts\phase3_api_smoke.py --start-server

# Read-only integrity checks.
.venv\Scripts\python.exe scripts\phase2_snapshot.py --check
.venv\Scripts\python.exe scripts\phase2a_preserve.py --check
.venv\Scripts\python.exe scripts\phase3_preserve.py --check
```

Stop any separately running server before the temporary smoke command; it refuses to reuse an occupied port. The full-test runner preserves earlier receipts and separates unit/mock, actual Phase 3 GPU and earlier tests in its additive JSON/JUnit evidence. Actual private-image integration is disabled by default; unavailable authorized files must not be replaced with fabricated results.

Settings are a validated dataclass, injectable through `create_app(settings, model_dependencies)`. Expression serving can be disabled with `Settings(expression_enabled=False)` in an explicit app factory. No new YAML config is necessary. Keep the established private base outside OneDrive, the project and network shares; it must already exist for comparison. There is no remote authentication design or deployment authorization.

Implementation references: [FastAPI file request contracts](https://fastapi.tiangolo.com/tutorial/request-files/), [self-hosted Swagger assets](https://fastapi.tiangolo.com/how-to/custom-docs-ui-assets/), and the official [Swagger UI v5.17.14 source/license](https://github.com/swagger-api/swagger-ui/tree/v5.17.14). The custom low-level parser adapts those multipart contracts to the project's RAM-only upload requirement.
