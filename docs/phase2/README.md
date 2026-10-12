> For another PC, follow [the source-checkout setup guide](../SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# FaceAttend Phase 2 research workflow

This is an adult, voluntary, pseudonymous **1:1 verification prototype**. A claimed ID selects exactly one encrypted template. It contains no expression inference, attendance marking, identity search, server, database or liveness feature. **NOT READY FOR UNSUPERVISED ATTENDANCE.**

Start with the six notebooks under `notebooks/phase2/`. Select **Python (FaceAttend GPU)** manually in VS Code. Existing `.vscode/settings.json` already points to `.venv`; GUI selection has not been observed. Ordinary Run All starts with real faces disabled and never activates the camera. The explicitly authorized recorded execution includes actual model CUDA profiles and the supplied-image diagnostic; enrollment/calibration limits are labeled.

## Installed environment and reproduction

The existing `.venv` was extended without changing its 124 existing package versions. Python 3.12.14, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA 12.6, cuDNN 9.10.2, NVIDIA RTX 3060 Laptop GPU (6 GiB), driver 576.52. ORT GPU **1.20.2** was selected against the [official CUDA/cuDNN matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html); newer releases use CUDA 12.8 or 13. InsightFace **1.0.1** supplies pure-Python SCRFD/ArcFace wrappers without optional GUI, person-analysis or liveness components. The installed official source is inspected directly for API behavior. `import torch` preloads the CUDA/cuDNN DLLs before sessions.

From the preserved Phase 1 environment, the additive install is:

```powershell
.\.venv\Scripts\python.exe -m pip install -c requirements-phase2-constraints.txt -r requirements-phase2.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-phase2-insightface.txt
.\.venv\Scripts\python.exe scripts\phase2_environment_report.py
.\.venv\Scripts\python.exe scripts\phase2_gpu_smoke.py
```

`requirements-phase2-lock.txt` records the full observed versions, including CUDA-specific PyTorch wheels. It is an inventory, not a portable replacement for the original CUDA-aware Phase 1 setup. InsightFace's base metadata requests CPU `onnxruntime`; the [official GPU installation procedure](https://github.com/deepinsight/insightface/tree/master/python-package#nvidia-cuda) substitutes `onnxruntime-gpu`. Only the GPU distribution is installed. `pip check` has one documented metadata warning (`insightface 1.0.1 requires onnxruntime, which is not installed.`); imports and actual CUDA convolution execution succeed. Do not install the CPU distribution to silence it, because both write the same module. The environment audit rejects any additional warning or existing-package change.

`cuda_session` sets `session.disable_cpu_ep_fallback=1`, verifies CUDA is the primary effective provider and disables Python runtime fallback. ORT may still list CPU in its registered providers; that does not mean CPU compute occurred. Profiles must contain CUDA computational events and no other provider. Actual detector and recognizer sessions must each pass this check after acquisition; synthetic runtime evidence alone is insufficient.

Both actual model profiles now pass this check. The downloaded graphs' dynamic detector dimensions and recognition batch symbol initially required CPU shape nodes, and strict initialization failed. The wrapper fixes input metadata to batch one, detector 640x640 and recognition 112x112 **in memory**; downloaded files and every weight remain unchanged. ORT can then fold shape bookkeeping and use CUDA exclusively. This follows the [official fixed-shape approach](https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html). The pipeline fingerprint versions this input contract. Batch sizes or resolutions outside those contracts are rejected.

## Confirmations and official model acquisition

The user has accepted the non-commercial research terms, confirmed adult consent, and explicitly supplied three images. Model acquisition succeeded after that data-availability confirmation. Both official models executed on CUDA and their weight hashes/effective providers/profiles are recorded. The earlier image-availability approval rejection is resolved.

The requested private image folder was created at `%LOCALAPPDATA%\FaceAttend\phase2\private\captures`, outside OneDrive and the project, with a user-only NTFS ACL. Sources use pseudonymous filenames. Two screenshots have separate main-photograph inputs excluding gallery thumbnails; the originals remain unchanged. Images and vectors are not copied into public reports/notebooks or uploaded to GitHub. No real embeddings, templates or keys were persisted.

The current collection has one photo per person. Inputs 1 and 2 fail the unchanged blur gate (7.45 and 20.73 versus minimum 60); input 3 passes (104.72) and produces a unit-norm 512-dimensional embedding. Collect 3–5 distinct clear original photos **per person**, plus independent-session probes, before enrollment. Image variants or copies are not additional captures. A tiny cohort still cannot satisfy the default low-FMR calibration policy.

### Latest ten-image retest

The original three-image paragraph above describes the initial collection. The later collection contains five user-labeled photos for each of two subjects; the user confirmed they are all from **one capture session**. Private pseudonymous copies and the name mapping remain in `%LOCALAPPDATA%\FaceAttend\phase2\private\batch_02`, outside OneDrive/GitHub. `reports/phase2/batch02_results.json` is the latest diagnostic artifact.

Seven of ten photos pass. Profiles 3, 5 and 6 fail the landmark-within-box gate; the QC policy is unchanged. Three photos per person were selected before score inspection for an encrypted enrollment smoke (1/2/4 and 7/8/9), with 3/5 and 6/10 reserved as probes. Both enrollments, encryption round-trip, duplicate-ID refusal and removal pass. The key exists only in process memory/environment, and the temporary real templates are explicitly removed before exit. No permanent registration or recoverable key was created by this **test**.

Photo 10's same-session diagnostic score is 0.6839 against the supplied-label reference, 0.3067 against the other reference. No cutoff or verification decision is inferred from those scores. All four actual claimed-ID calls refuse with `THRESHOLD_NOT_CALIBRATED`, because an adequate independent-session calibration does not exist. FMR/FNMR/threshold remain null. All-pair descriptive summaries include enrollment captures and dependent comparisons; they are not held-out accuracy.

The local runner is `scripts/phase2_labeled_batch_smoke.py`, accepting `--source-dir`, `--private-dir` and `--confirm-consented-adult-data`. It reads precisely the ten supplied numbered attachments, copies them privately, runs official CUDA inference and removes the transient encrypted store/key. Do not rename same-session captures as separate sessions, generate augmentations as independent probes, or choose a cutoff between these two participants' scores.

### Later single-subject cross-session retest

The user then supplied ten photos for an added subject and confirmed **each photo was taken in a different session**. The private name/ID mapping and originals remain under the protected LocalAppData `batch_03` folder. Dates were not supplied, so provenance is recorded as a user confirmation rather than inferred from image backgrounds or clothing.

Photos 1/2/7 were fixed as enrollment; 3/4/5/6/8/9/10 are probes. Eight pass unchanged QC; 6/10 fail the size/clipping gate. The new three-image encrypted enrollment and the two previously authorized control enrollments succeed transiently, then are removed with no key persistence. Five usable genuine probes score 0.529–0.835, versus 0.022–0.268 for ten explicitly labeled negative comparisons. These are dependent diagnostic scores, not a calibration cutoff or population accuracy measurement. Exact dates/chronology across different subjects remain unconfirmed.

`scripts/phase2_cross_session_smoke.py` runs this test with source/private paths, consent/session-confirmation flags and optional prior controls. It does not correct arbitrary rotation, crop away other faces, relax quality/consistency gates or enable matching. All actual API attempts return THRESHOLD_NOT_CALIBRATED. `batch03_results.json` is the latest artifact; earlier retests above remain historical. More independently sampled identities and subject-disjoint development/test data are still required for calibration.

After both confirmations, acquire only from official tooling:

```powershell
.\.venv\Scripts\python.exe scripts\phase2_acquire_models.py --accept-noncommercial-research-terms --confirm-consented-adult-data-available
.\.venv\Scripts\python.exe scripts\phase2_model_audit.py
$env:FACEATTEND_RUN_REAL_MODEL_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest tests\phase2\test_gpu_inference.py -q
```

The installed InsightFace version obtains `buffalo_l.zip` from its [official v0.7 release asset](https://github.com/deepinsight/insightface/releases/tag/v0.7). The [current official zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md) documents SCRFD-10GF plus ResNet50@WebFace600K. Only `det_10g.onnx` and `w600k_r50.onnx` are initialized; the remaining pack files are not used. The model audit records exact SHA-256 hashes, provider lists, input/output contracts and computational profiles. Initialization never auto-downloads. Code is [MIT licensed](https://github.com/deepinsight/insightface#license); publicly supplied weights are restricted to non-commercial research absent separate authorization. Do not redistribute weights. Institutional/commercial use needs a fresh licensing assessment.

## Prepare private captures

Use consenting adults and IDs such as `volunteer_001`. Explain purpose, retention, deletion, lack of liveness and the ability to refuse. Do not encode real names, emails, student/employee IDs or private participant details in published artifacts. The syntactic ID gate cannot verify pseudonymity by itself; the operator must honor it.

Collect 3–5 distinct enrollment captures per participant. Later independent capture sessions supply probes. Never use an enrollment frame as a probe, adjacent frames as independent evidence, or RAF-DB images as identity enrollments. Assign development/test **subjects** before computing scores. Each participant belongs to one partition; do not split a person's frames across partitions. The default conservative 1% FMR / one-sided 95% bound needs 299 independent impostor comparisons involving at least 598 development subjects; a small volunteer pilot stays `INSUFFICIENT_DATA`. That is an honest usable state, not a reason to insert an arbitrary 0.5 cutoff.

Copy `docs/phase2/consent_manifest.example.json` into `data/private/consent_manifest.json`, add private image paths/session IDs, and change adult/identity consent flags only after confirmation. Paths are relative to the private manifest. The example is deliberately fictional and has consent disabled. Add more participants; its two illustrative identities are insufficient for evaluation. Different enrollment/probe session labels must refer to actual separate collection sessions. Exact decoded hashes reject identical frames, but metadata cannot prove consent/session independence or catch all near-duplicates. Review collection manually.

This project lives under OneDrive. Before real enrollment, exclude/pause synchronization or move the entire checkout to a non-synced local location; paths remain portable. Restrict `data/private/` to the operator's NTFS account, e.g. from the project root:

```powershell
New-Item -ItemType Directory -Force data\private | Out-Null
$phase2Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
icacls data\private /inheritance:r /grant:r "${phase2Identity}:(OI)(CI)F"
```

These manual ACL and sync steps have **not** been applied or verified. Administrators and anyone holding the key can access biometrics. The prototype is single-operator, not a hardened multi-user service.

For the supplied three-image diagnostic, the separate LocalAppData private folder's user-only ACL was applied and checked. The project's `data/private/` template directory remains unused; the manual steps above apply if an operator later chooses that synced project location. Prefer using `TemplateStore(PRIVATE_INPUTS / 'templates', key_env=config.key_env, retention_days=config.retention_days)` for future enrollment so real templates also remain outside OneDrive.

## Encryption key and enrollment

Use a user-controlled secret manager or an interactive unsaved environment variable. Never write keys beside templates or in a notebook, committed `.env`, output, screenshot or log. A Fernet key must be generated once and retained securely; regenerating it makes existing templates unreadable. To generate a new session key without printing or writing it:

```powershell
$env:FACEATTEND_TEMPLATE_KEY = & .\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Save that key in your chosen secret manager before the session ends; this command only places it in the current process environment and its child processes. Notebook 02 reads the environment but never prints it. Adult and identity consent are explicit keyword arguments to enrollment and verification. Detection/alignment interfaces assume the caller already obtained permission. Pairwise enrollment cosine ≥0.6 is an unvalidated consistency QC heuristic, distinct from the calibrated verification threshold. A metadata/model-policy fingerprint prevents mixing weights or preprocessing across enrollment, calibration and verification.

Delete only the participant's selected record using `delete_enrollment(user_id, store)`, then recollect to re-enroll. No background deletion or implicit replacement occurs. Templates contain consent and sample digests inside authenticated encryption; filenames are hashes of predictable pseudonyms and are not anonymous. Deletion removes the active file, not backups, synchronized replicas, SSD remnants, other process memory or any separately retained input photos. Operator retention policy must cover those copies; no raw captures are saved by the core.

`retention_days` defaults to 30 in configuration (valid range 1–365). Pass it to `TemplateStore`; the documented callers do so. Expired records return `TEMPLATE_EXPIRED` and cannot be used for verification. There is no unattended deletion job: the operator must remove expired records and handle private input photos and backups under the agreed retention policy. Fresh consent is required before re-enrollment.

## Calibration, held-out evaluation and latency

```powershell
.\.venv\Scripts\python.exe scripts\phase2_evaluate.py --manifest data\private\consent_manifest.json --confirm-consented-adult-data
.\.venv\Scripts\python.exe scripts\phase2_verify.py --user-id volunteer_001 --image data\private\fresh_probe.jpg --confirm-adult-verification-consent
```

The evaluator produces private in-memory embeddings and comparisons, publishes only aggregate reports/plots, and creates the trusted local calibration artifact. It never exports embeddings or participant pair lists. A caller able to edit the calibration artifact can forge its metadata: it is **not cryptographically signed**. Protect source/configuration/reports against unauthorized modification. The loader checks status, real-data declaration, model fingerprint, counts, target/confidence and uncertainty before allowing acceptance; synthetic calibration never authorizes matching.

Threshold selection uses development only. Held-out subjects and images must be disjoint. FMR/FNMR denominators include their respective pair types; errors and denominators are reported separately. Scores use cosine ≥ cutoff for acceptance. Exact interval assumptions and independent subset limits are documented in `reports/phase2/threshold_report.md`. No demographic guarantee, confidence of identity for an individual, or production threshold is claimed. Evaluation currently has zero real pairs and null metrics.

Held-out evaluation checks the frozen development-protocol digest and labels a small final-test set `INSUFFICIENT_DATA`; descriptive rates do not become reliability evidence. The two-sided held-out interval has a different sample requirement from the one-sided development policy (368 versus 299 zero-error independent impostor comparisons at these defaults).

Notebook 05 provides a bounded one-frame webcam capture with SPACE/ESC/Q, stop on window close and timeout. It remains disabled in Run All. Component timing ran on the passing real input with 20 warmups and 100 repetitions: median detector 16.38 ms, aligned-crop embedding 5.83 ms, and quality/detection/alignment/embedding pipeline 28.26 ms. `latency_report.json` gives mean/median/p95 and explicit scopes. The pipeline timing excludes template decryption, threshold loading and a match decision, so full calibrated verification remains unmeasured. Camera capture and image decoding are excluded; no camera test occurred. Real-input diagnostics also ran from a clean interpreter, without notebook state.

## Verification and safe notebook execution

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .cache\phase2_full_tests --junitxml reports\phase2\tests.junit.xml
.\.venv\Scripts\python.exe scripts\phase2_execute_notebooks.py
.\.venv\Scripts\python.exe scripts\phase2_snapshot.py --check
```

To reproduce the explicitly consented three-image diagnostic using the already prepared private folder:

```powershell
$phase2PrivateInputs = Join-Path $env:LOCALAPPDATA 'FaceAttend\phase2\private'
.\.venv\Scripts\python.exe scripts\phase2_private_image_smoke.py --confirm-consented-adult-data --private-dir $phase2PrivateInputs --skip-latency
.\.venv\Scripts\python.exe scripts\phase2_execute_notebooks.py --consented-image-smoke --private-inputs $phase2PrivateInputs
```

The consented-image executor validates the private consent manifest and activates only the diagnostic/refusal branches suitable for this collection. It does not enroll a person from one photo, fabricate a threshold or open a camera. The private input manifest is intentionally separate from the full enrollment/probe protocol manifest.

The notebook executor launches a fresh kernel per Phase 2 file, does not rebuild notebooks or run Phase 1 notebooks, and records honest gated sections. Avoid rerunning the scaffold builder after editing private manual cells: it recreates only Phase 2 notebooks and clears their outputs. Clear any private notebook output before sharing; current published outputs contain no real participant data.

GPU/model failures stop inference rather than substitute CPU. The real-model integration test is opt-in; a skip is not a pass. See `reports/phase2/PHASE2_RESULTS_AND_LIMITATIONS.md` for observed counts, exact blockers and deferred work.
