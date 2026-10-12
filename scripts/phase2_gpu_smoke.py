"""Fresh-process CUDA proof using a synthetic Conv graph, never a real face."""
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from face_verification import load_config
from face_verification.gpu import cuda_session, inspect_profile

def main():
    config = load_config(root=ROOT)
    folder = ROOT / '.cache/phase2'
    folder.mkdir(parents=True, exist_ok=True)
    weights = np.ones((4,3,3,3), dtype=np.float32) / 27
    graph = helper.make_graph([helper.make_node('Conv', ['image','weight'], ['features'])], 'synthetic_cuda_check',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1,3,32,32])],
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, [1,4,30,30])],
        [numpy_helper.from_array(weights, 'weight')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',13)])
    model.ir_version = 8
    path = folder / 'synthetic_cuda.onnx'
    onnx.save(model, path)
    session = cuda_session(path, config, folder / 'profiles/synthetic')
    output = session.run(None, {'image': np.ones((1,3,32,32), dtype=np.float32)})[0]
    np.testing.assert_allclose(output, 1, atol=1e-6)
    result = inspect_profile(session.end_profiling())
    import torch
    import onnxruntime as ort
    result.update(input='synthetic all-one tensor; no real images', effective_providers=session.get_providers(),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        onnxruntime=ort.__version__, gpu=torch.cuda.get_device_name(0),
        cpu_fallback_disabled=session.get_session_options().get_session_config_entry('session.disable_cpu_ep_fallback'))
    (ROOT / 'reports/phase2/gpu_runtime_smoke.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
