import pytest
from faceattend_analysis.errors import AnalysisError
from faceattend_api.errors import APIError
from faceattend_candidates.service import CandidateService
from tests.phase4c.conftest import MemoryCatalog,add_reference
from .conftest import invoke

def test_anonymous_expression_only_no_embedding_or_gallery(analyzer,context,images):
    result=invoke(analyzer,context,images)
    assert result['face_count']==1 and result['faces'][0]['face_index']==0
    identity=result['faces'][0]['identity_diagnostic']
    assert identity['candidate_count']==0 and identity['identity_status']=='Unknown (unverified)'
    assert result['not_a_mood_or_diagnosis'] and result['verified'] is False
    assert context.bundle.detector.calls==1 and context.bundle.recognizer.calls==0

def test_candidates_and_expression_share_single_qc_detection(analyzer,context,images,gallery):
    add_reference(gallery,name='Fictional private reference')
    analyzer.candidate_service=CandidateService(gallery.service,catalog=MemoryCatalog(gallery.repository))
    result=invoke(analyzer,context,images)
    face=result['faces'][0]
    assert face['identity_diagnostic']['candidate_count']==1 and face['identity_diagnostic']['threshold'] is None
    assert face['visible_expression_prediction']['score_calibrated'] is False
    assert context.bundle.detector.calls==1 and context.bundle.recognizer.calls==1

@pytest.mark.parametrize('field',['adult_consent','research_opt_in','expression_opt_in'])
def test_all_optins_required_before_models(analyzer,context,images,field):
    flags=dict(adult_consent=True,research_opt_in=True,expression_opt_in=True);flags[field]=False
    with pytest.raises(AnalysisError):analyzer.analyze(images[0],context.bundle,**flags)
    assert context.bundle.detector.calls==0 and analyzer.expression.calls==0

@pytest.mark.parametrize('mode',['multiple','clipped','exception'])
def test_no_prediction_for_bystanders_or_invalid_face(analyzer,context,images,mode):
    context.bundle.detector.mode=mode
    with pytest.raises(APIError):invoke(analyzer,context,images)
    assert analyzer.expression.calls==0

def test_no_face_empty_and_no_expression(analyzer,context,images):
    context.bundle.detector.mode='none';result=invoke(analyzer,context,images)
    assert result['faces']==[] and result['face_count']==0 and not result['face_detected']
    assert analyzer.expression.calls==0 and context.bundle.recognizer.calls==0

def test_request_face_inputs_are_isolated_no_stale_crop(analyzer,context,images):
    invoke(analyzer,context,images);invoke(analyzer,context,[images[1]])
    import numpy as np
    assert not np.array_equal(analyzer.expression.inputs[0],analyzer.expression.inputs[1])
    assert analyzer.expression.calls==2 and context.bundle.detector.calls==2
