"""Build six bounded Phase 2 notebooks with explicit real-data execution gates."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'notebooks/phase2'
DEST.mkdir(parents=True,exist_ok=True)

SETUP = '''
from pathlib import Path
import sys, json, os, subprocess
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/phase2_verification.yaml').exists())
sys.path.insert(0,str(ROOT/'src'))
os.environ['MPLCONFIGDIR'] = str(ROOT/'.cache/matplotlib')
from face_verification import *
config = load_config(root=ROOT)
REPORTS = ROOT/'reports/phase2'
# Real faces are opt-in. Set these manually only after confirming each participant's consent.
RUN_REAL_FACES = False
ADULT_CONSENT = False
IDENTITY_CONSENT = False
PRIVATE_MANIFEST = ROOT/'data/private/consent_manifest.json'
print('Interpreter:',sys.executable)
print('Real-face workflow enabled:',RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT)
'''

def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())
def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())
def save(name,cells):
    notebook = nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'faceattend-gpu','display_name':'Python (FaceAttend GPU)','language':'python'},
        'language_info':{'name':'python','version':'3.12.14'}})
    nbf.validate(notebook)
    nbf.write(notebook,DEST/name)

save('00_GPU_and_Model_Environment.ipynb',[
md('''# Phase 2 — GPU and model environment
## Goal
Verify the existing laptop environment without touching Phase 1. The synthetic ONNX smoke proves CUDA execution in ONNX Runtime. Actual SCRFD/ArcFace loading is a separate check and may remain blocked.

The user confirmed non-commercial research terms and adult consent. Image-data availability and a local folder path remain pending. This handoff requires authorized model use and consented-data availability before acquiring weights. **NOT READY FOR UNSUPERVISED ATTENDANCE.**
'''),md('## Setup'),code(SETUP),
md('''## Runtime versions and hardware
The existing PyTorch CUDA 12.6/cuDNN 9 DLLs are preloaded by importing torch. ORT 1.20.2 matches the [official CUDA 12/cuDNN 9 matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). Newer CUDA 12.8/13 defaults were not installed. No Phase 1 package was upgraded.'''),
code('''import torch, onnxruntime as ort, cv2, insightface, cryptography
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
    'onnxruntime':ort.__version__,'opencv':cv2.__version__,'insightface':insightface.__version__,
    'cryptography':cryptography.__version__,'gpu':torch.cuda.get_device_name(0),
    'available_providers':ort.get_available_providers()},indent=2))
assert 'CUDAExecutionProvider' in ort.get_available_providers()
'''),md('## Fresh-process GPU smoke'),
code('''smoke = subprocess.run([sys.executable,str(ROOT/'scripts/phase2_gpu_smoke.py')],cwd=ROOT,capture_output=True,text=True,timeout=120)
print(smoke.stdout)
assert smoke.returncode == 0, smoke.stderr
'''),
md('''## Model contract and acquisition gate
[Official model zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md): buffalo_l contains SCRFD-10GF and ResNet50@WebFace600K recognition. Only these two ONNX files are loaded. Age/gender, expression and liveness are excluded. InsightFace's [MIT code license and research-only weights](https://github.com/deepinsight/insightface#license) differ.

`initialize_models` never downloads. After both confirmations, use the explicit acquisition script documented in the Phase 2 guide. Effective CUDA provider checks plus `session.disable_cpu_ep_fallback=1` fail closed; ORT may list CPU in its provider registry despite disabling graph fallback. Computational profile events must all be CUDA.'''),
code('''from face_verification.gpu import inspect_profile
try:
    models = initialize_models(config,profile=True)
    import numpy as np
    models.detector.detect(np.zeros((640,640,3),np.uint8),input_size=(640,640),max_num=0)
    models.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
    model_smoke = {'status':'ACTUAL_MODEL_SYNTHETIC_INPUT_PASS','model_id':models.model_id,
        'profiles': [inspect_profile(m.session.end_profiling()) for m in (models.detector,models.recognizer)]}
except VerificationError as exc:
    model_smoke = {'status':exc.status.value,'message':str(exc),'actual_models_tested':False}
print(json.dumps(model_smoke,indent=2))
'''),md('## Checks'),code('''integrity = subprocess.run([sys.executable,str(ROOT/'scripts/phase2_snapshot.py'),'--check'],cwd=ROOT,capture_output=True,text=True,timeout=120)
print(integrity.stdout)
assert integrity.returncode == 0, integrity.stderr
print(json.loads((REPORTS/'model_manifest.json').read_text()))
'''),md('''## Next steps
The synthetic CUDA success is runtime evidence, not detector/recognizer validation. Select **Python (FaceAttend GPU)** in VS Code manually. The interpreter settings and existing six Phase 1 notebooks remain unchanged. Weight acquisition and real-data experiments remain gated. Base InsightFace metadata requests CPU `onnxruntime`; the documented GPU substitution produces one expected `pip check` warning. Both runtimes must never coexist.''')])

save('01_SCRFD_Face_Detection_and_Alignment.ipynb',[
md('''# Phase 2 — SCRFD detection and official alignment
## Goal
Require exactly one face before computing an ArcFace embedding. Count all SCRFD detections above the confidence gate before quality filtering; never select the largest face from a crowd. No real images or identities are displayed or saved.'''),md('## Setup'),code(SETUP),
md('''## Quality policy
Research defaults: confidence 0.6, input minimum 160 pixels, face box minimum 80 pixels, face-region Laplacian variance ≥60, brightness 35–220, untruncated box and valid five-point landmarks. These are engineering gates, not calibrated biometric thresholds. They need adjustment using development captures only; their changes invalidate the model/policy fingerprint. Severe occlusion and unusual pose can still evade these heuristics. No liveness guarantee.'''),
code('''print({k:v for k,v in vars(config).items() if k.startswith(('min_','max_','detection_'))})
'''),md('## Synthetic error-path checks (no face-model inference)'),
code('''from types import SimpleNamespace
import numpy as np
session = SimpleNamespace(get_providers=lambda:['CUDAExecutionProvider'],disable_fallback=lambda:None)
image = np.random.default_rng(42).integers(0,255,(200,200,3),dtype=np.uint8)
for count in [0,2]:
    fake = SimpleNamespace(config=config,detector=SimpleNamespace(session=session,
        detect=lambda *args, count=count, **kwargs:(np.zeros((count,5)),None)))
    print('Injected detections:',count,detect_single_face(image,fake).to_dict())
fake = SimpleNamespace(config=config)
print('Unreadable:',detect_single_face(None,fake).to_dict())
print('Low resolution:',detect_single_face(np.zeros((50,50,3),np.uint8),fake).to_dict())
'''),md('''## Consented sample inference — manual opt-in
Set the three consent switches and create the private manifest from the documented schema first. This cell uses the first enrollment capture solely to inspect aggregate quality and alignment dimensions; it emits no portrait, pseudonym, bounding box or landmark coordinates into saved outputs.'''),
code('''if RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT:
    from face_verification.protocol import load_protocol
    import cv2
    protocol = load_protocol(PRIVATE_MANIFEST,consent_confirmed=True)
    models = initialize_models(config)
    image_path = protocol['participants'][0]['enrollment'][0]['_resolved_path']
    face = detect_single_face(cv2.imread(str(image_path)),models)
    print(face.to_dict())
    if face.status == Status.OK:
        print('Official norm_crop output:',face.aligned.shape,'five landmarks:',face.landmarks.shape)
else:
    print('NOT_RUN: real SCRFD detection/alignment awaits consented adult images and licensed model acquisition.')
'''),md('''## Checks and next steps
Synthetic NO_FACE, MULTIPLE_FACES, INVALID_IMAGE and LOW_QUALITY_IMAGE branches executed. They do not establish real detector accuracy. A valid consented real image is required to validate five-point alignment on this machine.''')])

save('02_Consent_Based_Face_Enrollment.ipynb',[
md('''# Phase 2 — consent-based enrollment
## Goal
Store one template per voluntary adult pseudonym. Normalize each of 3–5 distinct captures, require pairwise consistency, average and re-normalize. Templates and capture hashes are authenticated-encrypted; no raw images are saved by enrollment.

This checkout is under **OneDrive**. Before storing real biometrics, pause/exclude cloud synchronization or relocate the research checkout to a genuinely local folder, restrict NTFS permissions, and keep the key in a user-controlled secret source. Deleting a local file cannot revoke synced copies/backups.'''),md('## Setup'),code(SETUP),
md('''## Synthetic enrollment and encrypted storage check
This uses deliberately injected synthetic vectors and no face inference. An ephemeral key is kept only in process memory; no key is printed or saved. The temporary synthetic store is deleted when the cell ends.'''),
code('''from tempfile import TemporaryDirectory
from cryptography.fernet import Fernet
from types import SimpleNamespace
from face_verification.schemas import FaceResult
import numpy as np
with TemporaryDirectory(dir=ROOT/'.cache/tmp') as temporary:
    synthetic_store = TemplateStore(Path(temporary)/'templates',key=Fernet.generate_key())
    fake_models = SimpleNamespace(config=config,model_id='synthetic_unit_test',embedding_dim=3)
    images = [np.full((200,200,3),100+i,np.uint8) for i in range(3)]
    fake_detect = lambda image,models:FaceResult(Status.OK,aligned=image)
    fake_embed = lambda face,models:np.array([1.,0.,0.])
    missing_consent = enroll_user('volunteer_001',images,synthetic_store,fake_models,detector=fake_detect,embedder=fake_embed)
    result = enroll_user('volunteer_001',images,synthetic_store,fake_models,adult_consent=True,identity_consent=True,detector=fake_detect,embedder=fake_embed)
    duplicate = enroll_user('volunteer_001',images,synthetic_store,fake_models,adult_consent=True,identity_consent=True)
    encrypted = next(synthetic_store.directory.glob('*.fernet')).read_bytes()
    print({'synthetic_only':True,'missing_consent':missing_consent.status.value,'enrollment':result.status.value,
        'duplicate_id':duplicate.status.value,'norm':float(np.linalg.norm(synthetic_store.load('volunteer_001')['embedding'])),
        'plaintext_embedding_field_present':b'embedding' in encrypted,'plaintext_id_present':b'volunteer_001' in encrypted})
    print('Participant removal:',delete_enrollment('volunteer_001',synthetic_store).status.value)
'''),
md('''## Real enrollment — manual opt-in
The key must exist in `FACEATTEND_TEMPLATE_KEY` before this cell. Never paste it into a saved cell. The manifest contains pseudonyms, adult identity-verification consent, image paths, and capture-session labels. Keep it private. Do not overwrite duplicate enrollment; delete then recollect and re-enroll.'''),
code('''if RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT:
    from face_verification.protocol import load_protocol
    import cv2
    protocol = load_protocol(PRIVATE_MANIFEST,consent_confirmed=True)
    models = initialize_models(config)
    store = TemplateStore(config.path(config.store_dir),key_env=config.key_env,retention_days=config.retention_days)
    person = protocol['participants'][0]
    enrollment_images = [cv2.imread(str(record['_resolved_path'])) for record in person['enrollment']]
    result = enroll_user(person['user_id'],enrollment_images,store,models,adult_consent=True,identity_consent=True)
    print({'status':result.status.value,'sample_count':result.sample_count})
    del enrollment_images
else:
    print('NOT_RUN: no real enrollment; explicit adult consent, private captures, model weights and user-controlled key required.')
'''),md('''## Checks and next steps
Encryption round-trip, missing consent, duplicate ID and deletion are demonstrated with synthetic data. The consistency cutoff 0.6 is an enrollment QC heuristic, never a verification threshold. Embeddings remain sensitive even when encrypted. Windows ACL setup, key management and participant removal from backups remain operator responsibilities.''')])

save('03_ArcFace_One_to_One_Verification.ipynb',[
md('''# Phase 2 — ArcFace 1:1 verification
## Goal
Compare a fresh probe with only the explicitly claimed pseudonym. No matching before adequate real development calibration; no identity search. Scores are unit-normalized cosine similarities in [-1,1]. A match does not establish liveness or authorize attendance.'''),md('## Setup'),code(SETUP),
md('## Synthetic similarity checks'),code('''from face_verification.quality import cosine_similarity
print({'identical':cosine_similarity([1,0],[2,0]),'orthogonal':cosine_similarity([1,0],[0,1]),'opposite':cosine_similarity([1,0],[-1,0])})
'''),md('''## Uncalibrated refusal with injected synthetic vectors
The isolated configuration has no threshold artifact. It must refuse before calling the detector; a mocked perfect embedding cannot bypass calibration.'''),
code('''from tempfile import TemporaryDirectory
from cryptography.fernet import Fernet
from types import SimpleNamespace
from dataclasses import replace
from face_verification.schemas import FaceResult
import numpy as np
with TemporaryDirectory(dir=ROOT/'.cache/tmp') as temporary:
    isolated = replace(config,root=Path(temporary))
    store = TemplateStore(isolated.path(isolated.store_dir),key=Fernet.generate_key())
    fake_models = SimpleNamespace(config=isolated,model_id='synthetic_test',embedding_dim=3)
    samples = [np.full((200,200,3),100+i,np.uint8) for i in range(3)]
    enroll_user('volunteer_001',samples,store,fake_models,adult_consent=True,identity_consent=True,
        detector=lambda image,models:FaceResult(Status.OK,aligned=image),embedder=lambda *a:np.array([1,0,0]))
    def must_not_infer(*args):
        raise AssertionError('Uncalibrated inference was attempted.')
    result = verify_claimed_identity('volunteer_001',None,store,fake_models,adult_consent=True,identity_consent=True,detector=must_not_infer)
    print(json.dumps(result.to_dict(),indent=2))
    assert result.status == Status.THRESHOLD_NOT_CALIBRATED and not result.verified
'''),md('## Real claimed-ID verification — manual opt-in'),
code('''CLAIMED_ID = 'volunteer_001'  # Explicit participant selection; never inferred.
PROBE_PATH = ROOT/'data/private/fresh_probe.jpg'
if RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT:
    import cv2
    models = initialize_models(config)
    store = TemplateStore(config.path(config.store_dir),key_env=config.key_env,retention_days=config.retention_days)
    result = verify_claimed_identity(CLAIMED_ID,cv2.imread(str(PROBE_PATH)),store,models,adult_consent=True,identity_consent=True)
    # Saved output excludes pseudonym and similarity; inspect details only in a private session if needed.
    print({'status':result.status.value,'verified':result.verified})
else:
    print('NOT_RUN: real same-person and different-person trials require consented fresh probes and calibrated development evidence.')
'''),md('''## Checks and next steps
Unknown IDs and corrupt/missing templates fail closed; model/preprocessing fingerprints must match enrollment and calibration. A probe identical to an enrollment frame is rejected. Raw scores, embeddings and faces are not logged by the library. The CLI can intentionally emit a decision score for a local caller; keep that output private.''')])

save('04_Threshold_Calibration_and_Evaluation.ipynb',[
md('''# Phase 2 — threshold calibration and held-out evaluation
## Goal
Select a threshold from development captures only, then freeze it before subject-disjoint final evaluation. **There is no real calibration dataset yet, so the published outcome is INSUFFICIENT_DATA and the operational threshold remains null.**

FMR = accepted impostor comparisons / all impostor comparisons. FNMR = rejected genuine comparisons / all genuine comparisons. These are verification metrics, not expression-classifier Macro F1.'''),md('## Setup'),code(SETUP),
md('''## Protocol and prespecified tradeoff
Adult volunteers use pseudonyms with 3–5 enrollment images and independent-session probes. Development and test subjects, image paths and decoded content hashes must not overlap. No RAF-DB identity assumptions. Repeated pair matrices are dependent: development selection uses a deterministic disjoint-subject impostor subset selected without inspecting its scores.

Default target FMR is 1% with a one-sided 95% order-statistic tolerance bound. With zero errors this needs **299 independent impostor comparisons**, each using two different subjects without re-use. A handful of volunteers cannot satisfy this policy. This conservative research gate intentionally keeps acceptance disabled for small cohorts. Sample size does not establish representativeness, fairness or production readiness.'''),
code('''from face_verification.calibration import calibrate_development
report = calibrate_development([], 'NO_REAL_MODEL_CALIBRATION', config)
print(json.dumps(report,indent=2))
assert report['status'] == 'INSUFFICIENT_DATA' and report['threshold'] is None
'''),md('''## Synthetic score visualization — algorithm demonstration only
The following deterministic toy scores demonstrate plotting and confusion denominators. They do not come from faces and cannot authorize any verification threshold. Curves describe scores; they are not evidence of real biometric accuracy.'''),
code('''from face_verification.calibration import Pair,confusion_metrics
from face_verification.plotting import plot_scores
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
toy_pairs = []
for i in range(60):
    a,b = f'toy_{i}_a',f'toy_{i}_b'
    hashes = tuple(f'{a}_enrollment_{j}' for j in range(3))
    toy_pairs.append(Pair(float(np.clip(rng.normal(.68,.14),-1,1)),True,a,a,hashes,f'{a}_probe','enrollment','probe'))
    toy_pairs.append(Pair(float(np.clip(rng.normal(.30,.18),-1,1)),False,a,b,hashes,f'{b}_probe','enrollment','probe'))
toy_calibration = calibrate_development(toy_pairs,'synthetic',config,data_kind='synthetic')
print('Synthetic calibration status:',toy_calibration['status'],'threshold:',toy_calibration['threshold'])
fig = plot_scores(toy_pairs,REPORTS/'synthetic_score_demo.png',title='SYNTHETIC TEST DATA — no face-model accuracy claim')
plt.show()
'''),
md('''## Real protocol evaluation — manual opt-in
The reusable evaluator validates metadata, generates normalized embeddings without saving them, calibrates development, and evaluates the frozen cutoff on held-out subjects. Independent-subject exact intervals are reported alongside all-pair descriptive metrics. If data are inadequate, it writes a null threshold to disable verification. Test scores never influence selection.'''),
code('''if RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT:
    command = [sys.executable,str(ROOT/'scripts/phase2_evaluate.py'),'--manifest',str(PRIVATE_MANIFEST),'--confirm-consented-adult-data']
    completed = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=3600)
    print(completed.stdout)
    assert completed.returncode == 0, 'Private evaluation failed; inspect the local error without publishing participant paths.'
else:
    print('INSUFFICIENT_DATA: real development/test comparison counts = 0; FMR/FNMR/threshold are null.')
'''),md('''## Checks and next steps
Exact duplicate captures are rejected; session independence relies on truthful capture metadata, and near-duplicate videos need manual review. Broad demographic, lighting, camera and pose coverage has not been measured. Do not infer subgroup guarantees, anti-spoof performance, or production reliability from toy plots or small private cohorts.''')])

save('05_Webcam_Inference_and_Export_Smoke_Test.ipynb',[
md('''# Phase 2 — manual capture, latency and clean-process smoke
## Goal
Demonstrate reuse outside Jupyter and provide an explicit one-capture webcam action. **Webcam, real full verification and model latency have not been tested.** No automatic camera activation or attendance writes occur during Run All.'''),md('## Setup'),code(SETUP),
md('## Fresh-process import/model smoke'),
code('''completed = subprocess.run([sys.executable,str(ROOT/'scripts/phase2_verify.py'),'--smoke-no-data'],cwd=ROOT,capture_output=True,text=True,timeout=120)
print(completed.stdout)
assert completed.returncode == 0, completed.stderr
print(json.loads((REPORTS/'latency_report.json').read_text()))
'''),md('''## User-initiated camera capture — manual cell
Confirm adult consent, set `START_ONE_CAPTURE=True`, then run this cell yourself. A visible preview shows **SPACE to capture once, ESC/Q to stop**. Closing the window or the 60-second timeout stops capture. Nothing is saved. Keep unconsenting people out of the frame. No liveness assurance.

If OpenCV preview is unavailable, use a fresh consented local image. Codex has not opened the camera.'''),
code('''START_ONE_CAPTURE = False
captured = None
if START_ONE_CAPTURE and RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT:
    from face_verification.webcam import capture_one_frame
    captured = capture_one_frame(adult_consent=True,identity_consent=True,user_initiated=True)
    print('Capture result:', 'one in-memory frame' if captured is not None else 'stopped without capture')
else:
    print('WEBCAM_NOT_RUN: requires a deliberate participant action.')
'''),md('''## Full verification benchmark — manual opt-in
Use a valid consented fresh probe and previously enrolled claimed pseudonym after real calibration. Warmup 20, measured repetitions 100. Report mean/median/p95 for SCRFD, embedding from aligned crop, and full verification. CPU image decoding/capture are excluded; preprocessing, transfers and postprocessing are included as documented. Each synchronous `session.run` returns CPU output after GPU work completes.'''),
code('''RUN_LATENCY = False
CLAIMED_ID = 'volunteer_001'
PROBE_PATH = ROOT/'data/private/fresh_probe.jpg'
if RUN_LATENCY and RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT:
    import cv2
    from face_verification.benchmark import benchmark_verification
    models = initialize_models(config)
    store = TemplateStore(config.path(config.store_dir),key_env=config.key_env,retention_days=config.retention_days)
    probe = captured if captured is not None else cv2.imread(str(PROBE_PATH))
    latency = benchmark_verification(probe,CLAIMED_ID,store,models,adult_consent=True,identity_consent=True)
    (REPORTS/'latency_report.json').write_text(json.dumps(latency,indent=2))
    print(latency)
else:
    print('LATENCY_NOT_RUN: no valid consented probe, enrollment and calibrated threshold; no measurements invented.')
'''),md('''## Clean interpreter with private real data — manual opt-in
The encryption key must already be in the environment. This command loads its own models/templates and does not depend on variables in earlier notebooks. Keep CLI stdout private.'''),
code('''RUN_FRESH_REAL_VERIFY = False
if RUN_FRESH_REAL_VERIFY and RUN_REAL_FACES and ADULT_CONSENT and IDENTITY_CONSENT:
    result = subprocess.run([sys.executable,str(ROOT/'scripts/phase2_verify.py'),'--user-id',CLAIMED_ID,
        '--image',str(PROBE_PATH),'--confirm-adult-verification-consent'],cwd=ROOT,capture_output=True,text=True,timeout=120)
    decision = json.loads(result.stdout)
    print({'status':decision['status'],'verified':decision.get('verified',False)})
else:
    print('REAL_FRESH_PROCESS_VERIFY_NOT_RUN: consented inputs and calibration pending.')
'''),md('''## Checks and next steps
The test suite covers successful/rejected synthetic comparisons, unknown ID, no/multiple faces, quality, duplicate enrollment, removal, missing models, CUDA rejection and uncalibrated refusals. Synthetic outcomes are not real-face trials. No weights or biometric samples are exported publicly. Phase 3 still requires liveness/anti-spoofing, security review and a separate attendance design. **NOT READY FOR UNSUPERVISED ATTENDANCE.**''')])

from phase2_update_notebooks import update
update()
print(f'Created six Phase 2 notebooks in {DEST.relative_to(ROOT)}.')
