"""A QC refusal cannot invent CUDA evidence or hide CPU/missing-success profiles."""
import json
import pytest
from face_verification.schemas import Status,VerificationError
from .test_gpu import profile_for_receipt

def profile(tmp_path,providers):
    path=tmp_path/'synthetic_profile.json'
    path.write_text(json.dumps([{'cat':'Node','args':{'provider':p}} for p in providers]))
    return path

def test_unused_recognizer_is_not_claimed_cuda_confirmed(tmp_path):
    result=profile_for_receipt('recognizer',profile(tmp_path,[]),False)
    assert result['status']=='NOT_EXERCISED' and result['node_events_by_provider']=={}

@pytest.mark.parametrize('name,completed,providers',[
    ('recognizer',True,[]),('detector',False,[]),
    ('recognizer',False,['CPUExecutionProvider']),
    ('recognizer',True,['CUDAExecutionProvider','CPUExecutionProvider'])])
def test_missing_success_evidence_and_any_cpu_execution_are_refused(tmp_path,name,completed,providers):
    with pytest.raises(VerificationError) as error:
        profile_for_receipt(name,profile(tmp_path,providers),completed)
    assert error.value.status==Status.GPU_UNAVAILABLE

def test_exclusively_cuda_nodes_keep_original_profile_validation(tmp_path):
    result=profile_for_receipt('recognizer',profile(tmp_path,['CUDAExecutionProvider']),True)
    assert result['status']=='CUDA_CONFIRMED' and result['node_events_by_provider']=={'CUDAExecutionProvider':1}
