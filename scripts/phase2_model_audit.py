"""Audit actual local model hashes and CUDA compute using synthetic tensor inputs."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from face_verification import load_config,initialize_models
from face_verification.detector import file_hash
from face_verification.gpu import inspect_profile

config = load_config(root=ROOT)
bundle = initialize_models(config,profile=True)
bundle.detector.detect(np.zeros((640,640,3),np.uint8),input_size=(640,640),max_num=0)
bundle.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
path = ROOT/'reports/phase2/model_manifest.json'
report = json.loads(path.read_text())
report.update(status='ACTUAL_MODELS_CUDA_SYNTHETIC_INPUT_VALIDATED', model_id=bundle.model_id)
for key,model in [('detector',bundle.detector),('recognizer',bundle.recognizer)]:
    session = model.session
    report[key].update(sha256=file_hash(Path(model.model_file)),actual_session_validated=True,
        effective_providers=session.get_providers(), profile=inspect_profile(session.end_profiling()))
    report[key]['inputs'] = [{'name':i.name,'shape':i.shape,'type':i.type} for i in session.get_inputs()]
    report[key]['outputs'] = [{'name':i.name,'shape':i.shape,'type':i.type} for i in session.get_outputs()]
report['audit_input'] = 'Synthetic blank tensors; this audit does not process real faces.'
report.setdefault('real_face_validation','NO_REAL_IMAGE_REPORT')
path.write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'model_id':bundle.model_id,
    'detector_compute':report['detector']['profile'],'recognizer_compute':report['recognizer']['profile']},indent=2))
