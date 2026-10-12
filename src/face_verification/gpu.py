"""Strict CUDA sessions and auditable profiling; no CPU inference fallback."""
import json
from pathlib import Path
from uuid import uuid4
from .schemas import Status, VerificationError

def require_cuda(session):
    if not session.get_providers() or session.get_providers()[0] != 'CUDAExecutionProvider':
        raise VerificationError(Status.GPU_UNAVAILABLE, 'The effective model session is not CUDA-primary. Check ORT, driver and DLL versions.')
    session.disable_fallback()

def fixed_model_bytes(path, input_shape):
    """Fix the documented batch-one input contract, preserving every weight.

    Static input dimensions let ORT fold shape bookkeeping instead of assigning
    it to CPU. The original downloaded model is never rewritten.
    """
    import onnx
    graph = onnx.load(str(path))
    if len(graph.graph.input) != 1:
        raise VerificationError(Status.MODEL_ERROR, 'Expected a single model input.')
    dimensions = graph.graph.input[0].type.tensor_type.shape.dim
    if len(dimensions) != len(input_shape):
        raise VerificationError(Status.MODEL_ERROR, 'Model input rank mismatch.')
    for dimension, fixed in zip(dimensions, input_shape):
        if dimension.dim_value > 0 and dimension.dim_value != fixed:
            raise VerificationError(Status.MODEL_ERROR, 'Fixed model input dimensions disagree with the configured contract.')
        dimension.ClearField('dim_param')
        dimension.dim_value = int(fixed)
    return graph.SerializeToString()

def cuda_session(model, config, profile_prefix=None, *, input_shape=None):
    try:
        import torch  # Official ORT preload mechanism for the CUDA/cuDNN DLLs.
        import onnxruntime as ort
        if not torch.cuda.is_available() or 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise VerificationError(Status.GPU_UNAVAILABLE, 'CUDA is unavailable; CPU inference is disabled.')
        options = ort.SessionOptions()
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        options.log_severity_level = 3
        if profile_prefix:
            Path(profile_prefix).parent.mkdir(parents=True, exist_ok=True)
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix) + '_' + uuid4().hex[:12]
        source = fixed_model_bytes(model, input_shape) if input_shape else str(model)
        session = ort.InferenceSession(source, sess_options=options,
            providers=[('CUDAExecutionProvider', {'device_id': config.device_id, 'cudnn_conv_algo_search': 'HEURISTIC'})])
        require_cuda(session)
        return session
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(Status.GPU_UNAVAILABLE, 'CUDA session creation failed. Check matching CUDA 12/cuDNN 9 DLLs, model graph, and driver; CPU fallback is forbidden.') from exc

def inspect_profile(path):
    events = json.loads(Path(path).read_text())
    nodes = [e for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider')]
    counts = {}
    for event in nodes:
        provider = event['args']['provider']
        counts[provider] = counts.get(provider, 0) + 1
    if not nodes or not counts.get('CUDAExecutionProvider') or any(p != 'CUDAExecutionProvider' for p in counts):
        raise VerificationError(Status.GPU_UNAVAILABLE, 'Profiling did not confirm exclusive CUDA computational execution.')
    return {'status': 'CUDA_CONFIRMED', 'node_events_by_provider': counts, 'profile_file': Path(path).name}
