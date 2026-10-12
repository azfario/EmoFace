import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

def test_fresh_interpreter_module_smoke():
    completed = subprocess.run([sys.executable,str(ROOT/'scripts/phase2_verify.py'),'--smoke-no-data'],
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    assert completed.returncode == 0,completed.stderr
    result = json.loads(completed.stdout)
    assert result['status'] in ('MODEL_ERROR','REAL_MODELS_SYNTHETIC_INPUT_SMOKE')

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Synthetic ONNX integration requires a local NVIDIA CUDA device.')
def test_synthetic_gpu_in_fresh_process():
    completed = subprocess.run([sys.executable,str(ROOT/'scripts/phase2_gpu_smoke.py')],
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    assert completed.returncode == 0,completed.stderr
    result = json.loads(completed.stdout)
    assert result['status'] == 'CUDA_CONFIRMED'
    assert result['node_events_by_provider'].keys() == {'CUDAExecutionProvider'}
    assert result['cpu_fallback_disabled'] == '1'
