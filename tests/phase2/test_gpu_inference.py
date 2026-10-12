import os
from types import SimpleNamespace
import pytest
from face_verification import initialize_models, load_config, Status, VerificationError
from face_verification.gpu import require_cuda

@pytest.mark.parametrize('providers', [[], ['CPUExecutionProvider'], ['CPUExecutionProvider','CUDAExecutionProvider']])
def test_cpu_fallback_rejected(providers):
    with pytest.raises(VerificationError) as exc:
        require_cuda(SimpleNamespace(get_providers=lambda: providers))
    assert exc.value.status == Status.GPU_UNAVAILABLE

def test_missing_models(config):
    with pytest.raises(VerificationError) as exc:
        initialize_models(config)
    assert exc.value.status == Status.MODEL_ERROR

@pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_REAL_MODEL_TESTS') != '1', reason='Official model acquisition gated on consented adult data; not run by default.')
def test_real_models_cuda():
    import numpy as np
    from face_verification.gpu import inspect_profile
    models = initialize_models(load_config(),profile=True)
    models.detector.detect(np.zeros((640,640,3),np.uint8),input_size=(640,640),max_num=0)
    models.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
    for model in (models.detector,models.recognizer):
        require_cuda(model.session)
        assert inspect_profile(model.session.end_profiling())['status'] == 'CUDA_CONFIRMED'
