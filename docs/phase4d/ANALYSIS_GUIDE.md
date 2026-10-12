> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Phase 4D local per-face research guide

The new manual endpoint predicts a **visible facial-expression category** for one voluntarily analyzed adult. Candidate names are optional and remain **Possible match (unverified)**; identity remains Unknown, with `verified=false` and `threshold=null`. Expression softmax is uncalibrated, `not_a_mood_or_diagnosis=true`, and crop-domain accuracy is unmeasured. There is no history, attendance action, mental-state inference or camera capture.

Use the existing Windows `.venv`, saved M1 checkpoint, strict SCRFD/ArcFace CUDA runtime and native PostgreSQL at `127.0.0.1:55432`. No package installation, new training, model download or migration is needed. The existing protected operator auth/key/gallery configuration comes from [Phase 4B's guide](../phase4b/ENROLLMENT_GUIDE.md). Nothing in Phase 4D automatically creates production operator credentials or retains enrollment. [Phase 4C's guide](../phase4c/CANDIDATE_GUIDE.md) describes optional candidate ranking and current identity limits.

## Start and submit explicitly

With existing operator settings configured outside synced storage:

```powershell
$env:FACEATTEND_ENROLLMENT_ENABLED = 'true'
$env:FACEATTEND_FACE_ANALYSIS_ENABLED = 'true'
# Required only when requesting optional candidate ranking:
$env:FACEATTEND_CANDIDATES_ENABLED = 'true'
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase3_serve.py
```

The launcher uses one worker, `127.0.0.1:8000`, no access logs/reload/public proxy. Face analysis is disabled by default; invalid enable values fail closed. The original full-image expression route retains its original defaults and exact contract. GPU models load through the original startup lifecycle. Existing health remains DB-independent. The new route requires the local operator session even for anonymous expression, but the anonymous path does not query the gallery or need a working gallery service once operator auth is available. If candidate lookup is explicitly requested and its gallery/DB fails, the entire request fails safely instead of exposing a partial expression result.

Choose a single-face JPEG/PNG belonging to the consenting adult, under protected non-synced storage. No automatic camera opens. Example contains no participant name:

```powershell
$phase4dProbe = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase2\private\inputs\probe.jpg'
.\.venv\Scripts\python.exe scripts\phase4d_analyze.py --image $phase4dProbe --confirm-adult-consent --confirm-visible-expression-research
.\.venv\Scripts\python.exe scripts\phase4d_analyze.py --image $phase4dProbe --confirm-adult-consent --confirm-visible-expression-research --private-output
.\.venv\Scripts\python.exe scripts\phase4d_analyze.py --image $phase4dProbe --confirm-adult-consent --confirm-visible-expression-research --include-candidates --top-k 3 --private-output
```

The CLI prompts for a password, logs in, submits once and logs out. Default output contains only coarse status/face count and warnings. `--private-output` deliberately reveals the transient prediction and, if additionally requested, authorized unverified names locally. Do not redirect, log, screenshot, publish or save those individual results. CLI paths refuse synced/network/repository/reparse locations; reads are bounded to 8 MiB and JPEG/PNG magic. Server streaming bounds, original QC and timeout policy still apply. Inputs, crops and tensors remain transient; reference cleanup is best effort, with no secure-erasure guarantee.

## Separate HTTP contract

Use existing `POST /api/v1/demo/session`, with `x-faceattend-login: 1` and JSON password. The result sets an HttpOnly SameSite Strict cookie and returns CSRF. Keep both private and ephemeral. Then:

```text
POST http://127.0.0.1:8000/api/v1/demo/analyze-face?include_candidates=false
x-faceattend-csrf: <session token>
x-faceattend-expression-opt-in: 1
Content-Type: multipart/form-data; boundary=<generated boundary>

image: one consenting-adult JPEG/PNG
adult_consent: true
research_opt_in: true
```

Optional `include_candidates=true&top_k=3` additionally requires `x-faceattend-candidate-opt-in: 1`, the candidate enable flag and working protected reference service. `top_k` is 1–5. There is no unauthenticated name or expression diagnostic. CSRF, existing mutation rate limit, loopback Host/Origin, upload/inference timeout and final session checks apply. Timeout remains HTTP 504 after native work cleanup; concurrent requests refuse with 503. Responses are no-store, with no private access/body logging. New routes stay excluded from the six-route historical OpenAPI contract.

A fictional abbreviated response, not an actual adult prediction:

```json
{
  "face_count": 1,
  "faces": [{
    "face_index": 0,
    "identity_diagnostic": {
      "identity_status": "Unknown (unverified)",
      "possible_candidates": [],
      "verified": false,
      "threshold": null
    },
    "visible_expression_prediction": {
      "label": "neutral",
      "softmax_score": 0.2,
      "score_calibrated": false,
      "not_a_mood_or_diagnosis": true,
      "domain_accuracy_validated": false,
      "crop_version": "scrfd-bbox-rgb-m1-v1"
    }
  }],
  "persistent_inference_history": false
}
```

The full output adds warnings and descriptive entropy/top-score gap. These create no decision cutoff or probability calibration. One accepted detection is shared between expression and optional candidate embedding, with face index zero. No coordinates, crop bytes or raw vector is returned. No face returns `faces=[]`, `face_count=0`; multiple faces and failed QC refuse before classification or gallery search. An unenrolled voluntary adult or empty gallery can still receive anonymous expression classification. Candidate names and expression appear together only transiently in the authorized response; no persistent join or person-linked history exists in DB, telemetry or reports.

## Reproduce the completed evidence

Use only the same confirmed consenting adults and separate-session Test photos for this inspected functionality cohort, not an independent accuracy evaluation. The existing preparation helper copies anonymous inputs into protected temporary storage, including prior regression prerequisites:

```powershell
$phase4dSource = '<operator-owned directory containing Enrollment and Test>'
.\.venv\Scripts\python.exe scripts\phase4c_prepare_inputs.py --source-root $phase4dSource --confirm-adult-consent --confirm-persistent-enrollment --confirm-separate-probe-sessions
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py test .\.venv\Scripts\python.exe scripts\phase4d_run_tests.py --postgres-integration --gpu-integration --adult-image-integration
.\.venv\Scripts\python.exe scripts\phase4c_prepare_inputs.py --cleanup
.\.venv\Scripts\python.exe scripts\phase4d_preserve.py
```

Earlier legacy GPU checks require their original protected authorized input prerequisites too. The runner preserves the completed 460 cases and all historical receipts with new evidence under `reports/phase4d`. PostgreSQL integration is guarded to the disposable test DB, with no normal schema reset or enrollment. Without opt-ins, integrations skip and do not establish adult/GPU acceptance. `--gpu-integration` also exercises synthetic CUDA startup and crop-classifier checks independently of adult inputs.

The crop policy and its new domain are documented in the preprocessing report (local record, excluded from Git). Do not quote the old RAF-DB 79.66% as this endpoint's accuracy. Original models, historical reports and six endpoints remain preserved; Phase 5 requires separate explicit approval.
