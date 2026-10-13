"""Actual model CUDA compute on synthetic tensors; no private photographs."""
from pathlib import Path
import json,time
import numpy as np
import torch
from face_verification import load_config,initialize_models
from face_verification.gpu import inspect_profile
from faceattend.inference import load_model

ROOT=Path('/app')

def check():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    config=load_config(root=ROOT)
    bundle=initialize_models(config,profile=True)
    bundle.detector.detect(np.zeros((640,640,3),np.uint8),input_size=(640,640),max_num=0)
    bundle.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
    profiles={}
    for name,model in [('detector',bundle.detector),('recognizer',bundle.recognizer)]:
        profiles[name]=inspect_profile(model.session.end_profiling())['node_events_by_provider']
    model,_,_=load_model(ROOT/'models/faceattend_best.pt',torch.device('cuda:0'))
    sample=torch.zeros((1,3,224,224),device='cuda:0')
    with torch.inference_mode():
        model(sample);torch.cuda.synchronize();start=time.perf_counter();out=model(sample);torch.cuda.synchronize()
    assert out.is_cuda and tuple(out.shape)==(1,7)
    return {'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'exclusive_ort_compute':profiles,'expression_logits_shape':[1,7],'synthetic_expression_forward_ms':round((time.perf_counter()-start)*1000,2),'input':'synthetic tensors; no accuracy/identity benchmark'}

if __name__=='__main__':
    try:print(json.dumps(check()))
    except Exception:
        print(json.dumps({'status':'GPU_MODEL_CHECK_FAILED','cpu_fallback':False}));raise SystemExit(1)
