> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Phase 5 local UI guide

Use the existing laptop `.venv`, saved face models and M1 checkpoint. The interface is a consenting-adult research prototype. Identity stays **Unknown / unverified** even when candidate names are shown. Visible-expression scores are **uncalibrated**, not probabilities of correctness, actual feelings, moods or diagnoses. Exactly one consenting adult is supported; there is no attendance recording or individual prediction history.

## Build the frontend

From the existing workspace in PowerShell:

```powershell
Set-Location 'C:\Users\User\OneDrive\Desktop\FaceAttend'
node.exe --version
npm.cmd --version
npm.cmd --prefix frontend ci --ignore-scripts --no-fund
npm.cmd --prefix frontend run build
```

The verified runtime is Node 22.13.0 and npm 10.9.2. Packages are pinned in `frontend/package-lock.json`. Python packages were not changed. `frontend/dist` is served by the existing API at `/app/`; no development server, external CDN, second origin, CORS changes or public bind is required. Rebuild and refresh the browser after source changes. `dist` and `node_modules` are excluded from Git.

## Reuse the project PostgreSQL cluster

```powershell
$phase5PgBin = 'C:\Program Files\PostgreSQL\18\bin'
$phase5PgRoot = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase4\postgresql'
& "$phase5PgBin\pg_ctl.exe" -D "$phase5PgRoot\data" status
# Run start only if the preceding status says this project cluster is stopped.
& "$phase5PgBin\pg_ctl.exe" -D "$phase5PgRoot\data" -l "$phase5PgRoot\server.log" -w start
& "$phase5PgBin\pg_isready.exe" -h 127.0.0.1 -p 55432
Get-NetTCPConnection -LocalPort 55432 -State Listen | Select-Object LocalAddress,LocalPort
```

The only permitted project listener is `127.0.0.1:55432`. Do not initialize the existing data directory, change the unrelated Windows PostgreSQL service, migrate at API startup, or print/copy `credentials.json`. Background launches use `Start-Process -WindowStyle Hidden`; do not use `Start-Process -Wait` for the long-lived server tree.

## Deliberate retained operator setup

No retained production gallery, key or operator password was created during Phase 5. Reuse your own existing setup if already configured. Otherwise, follow the protected-path setup in [the Phase 4B guide](../phase4b/ENROLLMENT_GUIDE.md). This prompts for the password locally and refuses to overwrite existing secrets:

```powershell
$phase5Private = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase4b\private'
$env:FACEATTEND_ENROLLMENT_GALLERY_DIR = Join-Path $phase5Private 'gallery'
$env:FACEATTEND_ENROLLMENT_KEY_FILE = Join-Path $phase5Private 'secrets\gallery.key'
$env:FACEATTEND_OPERATOR_AUTH_FILE = Join-Path $phase5Private 'secrets\operator.secret'
.\.venv\Scripts\python.exe scripts\phase4b_admin.py setup --gallery-dir $env:FACEATTEND_ENROLLMENT_GALLERY_DIR --key-file $env:FACEATTEND_ENROLLMENT_KEY_FILE --auth-file $env:FACEATTEND_OPERATOR_AUTH_FILE
```

Keep these files outside OneDrive and Git, under the operator's restrictive NTFS ACL. The UI never creates or resets credentials, stores no password/token in browser storage, and uses the existing HttpOnly, SameSite cookie plus a CSRF token held only in memory. An administrator or compromised local device remains outside this protection.

## Start the existing API with the UI

With your protected operator paths configured in that PowerShell session:

```powershell
$env:FACEATTEND_ENROLLMENT_ENABLED = 'true'
$env:FACEATTEND_ENROLLMENT_RETENTION_DAYS = '30'
$env:FACEATTEND_CANDIDATES_ENABLED = 'true'
$env:FACEATTEND_FACE_ANALYSIS_ENABLED = 'true'
$env:FACEATTEND_FRONTEND_ENABLED = 'true'
$env:FACEATTEND_DB_TELEMETRY_ENABLED = 'false'
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py runtime .\.venv\Scripts\python.exe scripts\phase3_serve.py
```

Open **http://127.0.0.1:8000/app/** in a browser on the laptop. This uses one loopback API worker and the existing runtime role. Do not use `0.0.0.0`, a tunnel, a proxy, a different origin or an extra GPU worker. Missing CUDA refuses processing; there is no CPU fallback. Optional enrollment/candidates/expression integration and the frontend remain disabled by default unless explicitly enabled. Telemetry stays off by default.

## Requested blurry-frame behavior

The visible-expression option now permits blurry frames when one sufficiently large, complete face is still detectable and adequately lit. If the original blur minimum is not met, the UI warns that the expression result may be unreliable and skips identity matching. Enrollment, candidate-only preview and original API defaults still require the original quality gates (minimum 60). No sharpening or changes to model weights/configuration were made. Severe blur that prevents face detection, clipped/multiple faces, black frames and bad brightness still cannot produce a per-face prediction.

## Automatic camera behavior

The user requested automatic detection and analysis. Automatic sampling, expression prediction and the face outline now default on; adult consent must still be confirmed. When you Start camera and its live preview is ready, analysis begins automatically. Results and the latest face outline appear over the preview. Candidate names remain an optional separate choice. The next fresh frame starts as soon as the prior inference finishes, with a ceiling of 20 updates per second and a five-minute/6,000-frame bound. Actual speed depends on the GPU. The latest face box carries `Emotion: <category>` and an uncalibrated-prediction note above it; this is not independently tracked video or a measured emotional state. Live calls use a separately limited opt-in bucket; manual/enrollment requests keep their existing limits. Stop analysis stays stopped until Resume or an explicit automatic-mode restart. No background capture or recording is added.

## Operate the interface

1. Check **Overview**, then sign in with your local operator password. The password input clears on submission. The session ends after 15 minutes, on End session, or when the tab becomes hidden; returning to a hidden tab requires a fresh sign-in.
2. In **Enrollments**, choose the adult's chosen display name/pseudonym, select 1–30 days, supply 3–5 distinct JPEG/PNG photos or manually capture references, and obtain both adult research consent and encrypted-reference retention consent. Submit Save encrypted reference. The existing blur minimum **60** and all face-count, clipping, landmark and consistency gates remain unchanged. Improve focus/light or replace inputs if rejected.
3. The gallery shows authorized metadata only. Re-enroll requires fresh photos and consent. Revoke excludes matching and removes the reference while retaining consent metadata; Delete removes the selected metadata and scoped encrypted reference. Confirm the selected operation deliberately. Original user files, synced copies and backups are unaffected.
4. In **Research camera**, choose Refresh cameras, select a source, and click Start camera yourself. Allow the browser's camera permission only for this local page. If camera names are hidden before permission, start once to allow access, stop the camera, refresh cameras, then choose the named source and start it. No microphone is requested. Phone users can choose OBS Virtual Camera with the existing Camo source selected in OBS. Direct Camo/OpenCV black-frame history does not establish how a particular browser source behaves.
5. Confirm the subject's adult voluntary research consent and wait for a live preview. With automatic mode enabled, face detection, optional expression and the outline update automatically; there is no Analyze click. You can change expression/outline options or separately opt into possible candidate names. Uploaded images, or a camera with automatic mode disabled, retain the manual Analyze frame button. There is no SPACE key capture in this browser workflow.
6. Live analysis permits one in-flight request. The next fresh frame starts when inference finishes, with at most 20 starts per second, and stops after five minutes or 6,000 frames. Stop analysis clears the result and prevents new capture; Resume automatic analysis starts another bounded session when ready/consented. Submitted GPU work can finish cleanup after client cancellation. Stop camera releases tracks. Navigation, unmount, tab hide, ended camera permission, consent withdrawal and session end stop relevant capture and discard private state.

Latest results and photo previews are transient browser memory/object URLs. URLs are revoked when removed or the component is destroyed. The client requests no media recording, filesystem downloads, external analytics, localStorage, sessionStorage or IndexedDB. There is no service worker. The static response has a same-origin CSP, camera-only Permissions Policy, no-store and anti-framing headers. Encrypted templates and chosen-name consent metadata persist only through explicit enrollment; no expression history is persisted.

## Source-checkout checks

For a new clone, use `.\.venv\Scripts\python.exe scripts\run_checks.py --frontend`. This needs no original-laptop handoffs or preservation reports. Optional real-model/database/private-input checks require separate local setup. See [the new-PC guide](../SETUP_NEW_PC.md).

## Temporary live acceptance fixture

For a local check without creating retained production credentials:

```powershell
New-Item -ItemType Directory -Path reports\phase5,.cache -Force | Out-Null
.\.venv\Scripts\python.exe scripts\phase4b_with_local_db.py test .\.venv\Scripts\python.exe scripts\phase5_fixture_server.py --mode live --port 8015 --stop-after 7200
```

Open **http://127.0.0.1:8015/app/**. Its operator password is the public disposable fixture constant **Fictional test operator password**. It is never a production credential. This uses real existing CUDA models and isolated test SQL, with empty enrollment initially. Do not use the fixture for retained enrollment. Exit removes its temporary key/auth/gallery and guarded test demo schema; it refuses unexplained pre-existing test registrations. It stops automatically after the chosen lifetime (the example now uses two hours; sessions still expire after 15 minutes). A page left open after the service stops cannot sign in; restart the local process and refresh. You can request an earlier graceful stop from a second workspace terminal:

```powershell
New-Item -ItemType File -Path .cache\phase5_stop_live -Force | Out-Null
```

The `--mode fictional` fixture deliberately substitutes models and registration storage, displays a prominent simulated-data banner, and exists only for UI testing. Screenshots in Phase 5 contain fictional data. No real participant photos or individual prediction labels/scores are saved in reports.

Phase 6 requires separate explicit approval. No subsequent phase is started by these commands.
