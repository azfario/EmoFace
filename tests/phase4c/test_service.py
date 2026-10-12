from dataclasses import replace
from datetime import datetime,timezone,timedelta
from uuid import UUID
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pytest
from cryptography.fernet import Fernet
from faceattend_candidates.errors import CandidateError
from faceattend_api.errors import APIError
from face_verification.schemas import VerificationError
from faceattend_enrollment.errors import EnrollmentError
from .conftest import add_reference,invoke

def test_empty_gallery_is_unknown_without_candidates(search,context,images):
    result=invoke(search,context,images)
    assert result['face_detected'] and result['possible_candidates']==[]
    assert result['identity_status']=='Unknown (unverified)' and result['verified'] is False
    assert result['threshold'] is None and result['reason']=='NO_ELIGIBLE_REFERENCES'

def test_rank_cosine_and_top_k_without_identity_acceptance(gallery,search,context,images):
    add_reference(gallery,np.ones(512),name='Fictional A')
    add_reference(gallery,np.eye(1,512).flatten(),name='Fictional B')
    add_reference(gallery,-np.ones(512),name='Fictional C')
    result=invoke(search,context,images,top_k=2)
    assert [r['display_name'] for r in result['possible_candidates']]==['Fictional A','Fictional B']
    assert result['candidate_count']==2 and result['eligible_reference_count']==3
    assert result['identification_status']=='UNCALIBRATED_DIAGNOSTIC' and result['verified'] is False
    assert all(r['label']=='Possible match (unverified)' and r['verified'] is False for r in result['possible_candidates'])
    assert result['top_score_gap']>0 and 'No acceptance threshold' in result['warning']
    assert not any(word in json.dumps(result) for word in ('embedding','template_locator','sample_hashes','recorded_at'))

def test_unknown_probe_still_has_unverified_candidates_not_a_forced_identity(gallery,search,context,images):
    add_reference(gallery,-np.ones(512),name='Fictional unrelated reference')
    result=invoke(search,context,images)
    assert result['possible_candidates'][0]['cosine_similarity']==pytest.approx(-1)
    assert result['identity_status']=='Unknown (unverified)' and result['threshold'] is None

def test_near_duplicate_ties_are_stable_by_uuid_and_not_verified(gallery,search,context,images):
    b=add_reference(gallery,name='Fictional B',identity=UUID('00000000-0000-4000-8000-000000000002'))
    a=add_reference(gallery,name='Fictional A',identity=UUID('00000000-0000-4000-8000-000000000001'))
    result=invoke(search,context,images)
    assert [r['identity_id'] for r in result['possible_candidates']]==[str(a.identity_id),str(b.identity_id)]
    assert result['exact_top_score_tie'] and result['top_score_gap']==0
    assert result['verified'] is False and result['threshold'] is None

@pytest.mark.parametrize('top_k',[0,6,True,1.5,'3'])
def test_top_k_bound_before_inference(search,context,images,top_k):
    with pytest.raises(CandidateError):invoke(search,context,images,top_k=top_k)
    assert context.bundle.detector.calls==0

@pytest.mark.parametrize('field',['adult_consent','research_opt_in','search_opt_in'])
def test_all_probe_opt_ins_before_inference(search,context,images,field):
    flags=dict(adult_consent=True,research_opt_in=True,search_opt_in=True);flags[field]=False
    with pytest.raises(CandidateError):search.search(images[0],context.bundle,**flags)
    assert context.bundle.detector.calls==0

@pytest.mark.parametrize('status',['pending','revoked','expired','deletion_pending'])
def test_inactive_states_never_enter_ranking(gallery,search,context,images,status):
    add_reference(gallery,status=status)
    assert invoke(search,context,images)['candidate_count']==0

@pytest.mark.parametrize('change',[{'model_policy_fingerprint':'wrong'}, {'purpose_version':'wrong'},
    {'expires_at':datetime.now(timezone.utc)-timedelta(seconds=1)}])
def test_wrong_fingerprint_purpose_and_expiry_excluded(gallery,search,context,images,change):
    row=add_reference(gallery);gallery.repository.rows[row.identity_id]=replace(row,**change)
    assert invoke(search,context,images)['candidate_count']==0

def test_corrupt_reference_excluded_but_wholly_broken_gallery_refuses(gallery,search,context,images):
    bad=add_reference(gallery,name='Fictional bad');good=add_reference(gallery,name='Fictional good')
    gallery.files.final(bad.template_locator).write_bytes(b'CORRUPT_PRIVATE_REFERENCE')
    result=invoke(search,context,images)
    assert [r['identity_id'] for r in result['possible_candidates']]==[str(good.identity_id)]
    assert result['excluded_reference_count']==1
    gallery.repository.rows.pop(good.identity_id)
    with pytest.raises(CandidateError) as error:invoke(search,context,images)
    assert error.value.code=='CANDIDATE_GALLERY_UNAVAILABLE'

@pytest.mark.parametrize('mode',['missing','wrong'])
def test_missing_wrong_keys_fail_safely(gallery,search,context,images,mode):
    add_reference(gallery)
    gallery.files.key=None if mode=='missing' else Fernet.generate_key()
    with pytest.raises(CandidateError):invoke(search,context,images)

def test_metadata_and_cipher_fingerprints_must_both_match(gallery,search,context,images):
    row=add_reference(gallery,model_id='wrong')
    gallery.repository.rows[row.identity_id]=replace(row,model_policy_fingerprint=context.bundle.model_id)
    with pytest.raises(CandidateError):invoke(search,context,images)

@pytest.mark.parametrize('dimension',[511,513])
def test_reference_dimension_mismatch_never_ranked(gallery,search,context,images,dimension):
    add_reference(gallery,np.ones(dimension))
    with pytest.raises(CandidateError):invoke(search,context,images)

def test_ciphertext_size_bound(gallery,search,context,images):
    row=add_reference(gallery)
    gallery.files.final(row.template_locator).write_bytes(b'PRIVATE_TOO_LARGE'*5000)
    with pytest.raises(CandidateError):invoke(search,context,images)

def test_gallery_limit_no_silent_truncation(gallery,search,context,images,monkeypatch):
    row=add_reference(gallery)
    monkeypatch.setattr(search.catalog,'snapshot',lambda:[row]*101)
    with pytest.raises(CandidateError) as e:invoke(search,context,images)
    assert e.value.code=='CANDIDATE_GALLERY_TOO_LARGE'

def test_newest_revoked_version_cannot_fall_back_to_old_active(gallery,search,context,images,monkeypatch):
    row=add_reference(gallery)
    newer=replace(row,enrolled_at=row.enrolled_at+timedelta(seconds=1),status='revoked')
    monkeypatch.setattr(search.catalog,'snapshot',lambda:[row,newer])
    assert invoke(search,context,images)['candidate_count']==0

def test_shared_locator_not_assigned_to_multiple_names(gallery,search,context,images,monkeypatch):
    a=add_reference(gallery,name='Fictional A');b=add_reference(gallery,name='Fictional B')
    monkeypatch.setattr(search.catalog,'snapshot',lambda:[a,replace(b,template_locator=a.template_locator)])
    assert invoke(search,context,images)['candidate_count']==0

def test_revalidation_discards_revoked_or_replaced_rows(gallery,search,context,images,monkeypatch):
    add_reference(gallery)
    monkeypatch.setattr(search.catalog,'revalidate',lambda rows:[])
    assert invoke(search,context,images)['candidate_count']==0

def test_expiry_rechecked_after_reference_load(gallery,search,context,images):
    row=add_reference(gallery)
    times=iter([row.enrolled_at+timedelta(seconds=1),row.expires_at+timedelta(seconds=1)])
    search.clock=lambda:next(times)
    assert invoke(search,context,images)['candidate_count']==0

def test_database_outage_not_reported_as_empty_gallery(search,context,images,monkeypatch):
    def fail():raise RuntimeError('PRIVATE_DATABASE_PASSWORD')
    monkeypatch.setattr(search.catalog,'snapshot',fail)
    with pytest.raises(CandidateError) as e:invoke(search,context,images)
    assert 'PRIVATE_DATABASE_PASSWORD' not in str(e.value)

def test_gallery_lock_blocks_parallel_mutation(gallery,search,context,images):
    with gallery.files.locked():
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(invoke,search,context,images)
            with pytest.raises(EnrollmentError):future.result()

@pytest.mark.parametrize('mode',['multiple','clipped','exception'])
def test_quality_and_bystander_refusals_before_gallery(search,context,images,mode):
    context.bundle.detector.mode=mode
    with pytest.raises(APIError):invoke(search,context,images)
    assert search.catalog.snapshots==0

def test_no_face_is_unknown_with_no_gallery_access(search,context,images):
    context.bundle.detector.mode='none'
    result=invoke(search,context,images)
    assert result['face_detected'] is False and result['candidate_count']==0 and search.catalog.snapshots==0

@pytest.mark.parametrize('model',['detector','recognizer'])
def test_cpu_model_execution_refused(search,context,images,model):
    getattr(context.bundle,model).session.providers=['CPUExecutionProvider']
    with pytest.raises((APIError,VerificationError)):invoke(search,context,images)
    assert search.catalog.snapshots==0

def test_close_scores_report_only_gap_without_acceptance(gallery,search,context,images):
    add_reference(gallery,name='Fictional A')
    vector=np.ones(512);vector[0]+=0.1
    add_reference(gallery,vector,name='Fictional B')
    result=invoke(search,context,images)
    assert 0<result['top_score_gap']<0.001 and not result['exact_top_score_tie']
    assert result['candidate_count']==2 and result['verified'] is False and result['threshold'] is None

def test_metadata_cannot_extend_consent_beyond_thirty_days(gallery,search,context,images):
    row=add_reference(gallery)
    gallery.repository.rows[row.identity_id]=replace(row,expires_at=row.consent_recorded_at+timedelta(days=31))
    assert invoke(search,context,images)['candidate_count']==0
