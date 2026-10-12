> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Phase 4C local candidate research guide

Phase 4C adds one manual, authenticated single-face candidate route. All identities remain **Unknown (unverified)**. Candidate display labels say **Possible match (unverified)**; cosine values are diagnostics, not probabilities. There is no accepted identity, attendance decision, expression integration, camera capture or calibrated threshold.

Use the existing Windows `.venv`, SCRFD/ArcFace CUDA models and native PostgreSQL at `127.0.0.1:55432`. No dependency installation, schema migration, new DB grants or model replacement is needed. The existing Phase 4B operator setup and explicit enrollment workflow remain in [the enrollment guide](../phase4b/ENROLLMENT_GUIDE.md). That completed guide's Phase 4C approval gate is historical; the user approved this phase on 12 October 2026. Phase 4D still requires separate approval.

## Enable a manual local request

First deliberately configure Phase 4B protected gallery/key/auth files outside synced storage and enroll consenting adults with 3–5 distinct acceptable photographs. Setup alone creates no enrollment. The integration tests retained no production gallery or credentials. Never copy a key into the gallery, repository, source arguments or reports.

With those existing operator settings configured:

```powershell
$env:FACEATTEND_ENROLLMENT_ENABLED = 'true'
$env:FACEATTEND_CANDIDATES_ENABLED = 'true'
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase3_serve.py
```

The service binds only `127.0.0.1:8000`, one worker, no access logs or reload. Candidate search is disabled by default. Invalid enable values fail closed; valid search additionally requires working enrollment storage and operator auth. Existing DB-independent health and six research OpenAPI contracts are preserved. The separate management/candidate route is deliberately excluded from that historical schema.

In a second terminal, place the voluntarily supplied probe under a protected non-synced directory. Do not use a bystander photograph. Example variables below have no participant names:

```powershell
$phase4cProbe = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase2\private\inputs\probe.jpg'
.\.venv\Scripts\python.exe scripts\phase4c_candidates.py --image $phase4cProbe --top-k 3 --confirm-adult-consent --confirm-research-search
.\.venv\Scripts\python.exe scripts\phase4c_candidates.py --image $phase4cProbe --top-k 3 --confirm-adult-consent --confirm-research-search --private-output
```

The CLI prompts for the operator password, logs in, submits one request and logs out. Default output omits names, UUIDs and scores. `--private-output` explicitly shows authorized diagnostic results locally; do not redirect, screenshot, publish or log them. The CLI refuses repository/synced/network/reparse paths, invalid encodings and files over 8 MiB. The server also enforces bounded streaming multipart parsing, dimensions, timeouts and original QC. Memory cleanup is best effort; there is no secure RAM/SSD erase claim.

## HTTP contract and fictional sample

Authenticate with `POST /api/v1/demo/session` using `x-faceattend-login: 1` and a JSON password. Reuse its HttpOnly SameSite Strict cookie and returned CSRF token without saving either. Submit:

```text
POST http://127.0.0.1:8000/api/v1/demo/candidates?top_k=3
x-faceattend-csrf: <session token>
x-faceattend-candidate-opt-in: 1
Content-Type: multipart/form-data; boundary=<generated boundary>

image: one JPEG or PNG file
adult_consent: true
research_opt_in: true
```

`top_k` is 1–5, default 3; at most 100 gallery metadata rows are eligible for consideration. Search reads the whole bounded catalog across its old 25-record pages without truncating it. Over-capacity catalogs refuse rather than silently searching a subset. Results contain no source paths, raw vectors, template locators or sample digests. Responses use `Cache-Control: no-store`. Request/body/DB logging stays disabled; no scores or person-linked histories are written to PostgreSQL or anonymous aggregate telemetry.

This is an illustrative fictional excerpt, not an actual participant's result:

```json
{
  "face_detected": true,
  "candidate_count": 1,
  "possible_candidates": [{
    "display_name": "Fictional Adult",
    "label": "Possible match (unverified)",
    "cosine_similarity": 0.12,
    "verified": false
  }],
  "identification_status": "UNCALIBRATED_DIAGNOSTIC",
  "identity_status": "Unknown (unverified)",
  "verified": false,
  "threshold": null,
  "reason": "THRESHOLD_NOT_CALIBRATED"
}
```

The full response includes a research warning, eligible/excluded counts, exact-tie flag and numerical top-score gap. These fields define no acceptance or ambiguity cutoff. An absent subject can receive ranked names even with a high cosine value. The original 1:1 route still refuses operational verification when its threshold is uncalibrated.

No face produces Unknown with no catalog access. Empty/all-inactive metadata produces Unknown/no eligible references. Multiple faces and failed QC refuse before gallery search. Invalid or corrupt individual references are excluded; an entirely unreadable eligible gallery, DB outage, missing/wrong key or unsafe path fails generically without names. Auth/CSRF/local admission and final-session checks protect candidate exposure. Revocation, deletion, replacement, expiry and fingerprint mismatch exclude affected references.

## Reproduce and clean functionality inputs

Use only consenting adults and confirmed separate-session Test photos. Filename alphabetic owner labels must match exactly between `Enrollment` and `Test`; three groups are supported by this supplied cohort fixture. This grouping uses filenames only. Each enrollment group needs 3–5 files and each probe group 1–5. The helper prepares protected anonymous copies, including the completed Phase 4B regression inputs, and refuses existing targets. It never changes originals.

```powershell
$phase4cSource = '<operator-owned directory containing Enrollment and Test>'
.\.venv\Scripts\python.exe scripts\phase4c_prepare_inputs.py --source-root $phase4cSource --confirm-adult-consent --confirm-persistent-enrollment --confirm-separate-probe-sessions
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py test .\.venv\Scripts\python.exe scripts\phase4c_run_tests.py --postgres-integration --gpu-integration --adult-image-integration
.\.venv\Scripts\python.exe scripts\phase4c_prepare_inputs.py --cleanup
.\.venv\Scripts\python.exe scripts\phase4c_preserve.py
```

Use the original protected test-input preparation for earlier legacy GPU cases if those copies are absent. The runner restores previous mutable GPU receipts byte-for-byte and writes new evidence under `reports/phase4c`. Real PostgreSQL fixtures guard the known disposable database/roles and refuse unexplained existing registrations. No normal database downgrade, reset or enrollment occurs. Without explicit integration flags, those cases skip; a skipped run does not prove CUDA/adult functionality. The final reported full run enables every applicable integration flag.

Accepted input selection uses the first three enrollment QC passes and first probe QC pass in deterministic filename order. It never selects by a matching score. Keep this now-inspected cohort out of any future independent calibration/evaluation set. See calibration limitations (local record, excluded from Git).
