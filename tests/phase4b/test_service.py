from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID,uuid4
import json
import pytest
from cryptography.fernet import Fernet
from faceattend_api.errors import APIError
from faceattend_enrollment.files import GalleryFiles
from faceattend_enrollment.service import EnrollmentService
from faceattend_enrollment.errors import EnrollmentError
from .conftest import enroll_core

def test_encryption_restart_roundtrip_and_delete(gallery,context,images):
    result=enroll_core(gallery,context.bundle,images)
    identity=UUID(result['identity_id']);row=gallery.repository.get(identity)
    ciphertext=gallery.files.final(row.template_locator).read_bytes()
    assert b'embedding' not in ciphertext and b'Fictional Adult' not in ciphertext
    assert row.template_locator not in json.dumps(result)
    restarted=EnrollmentService(gallery.repository,GalleryFiles(gallery.root,key=gallery.key_file.read_bytes()))
    assert restarted.inspect(identity,context.bundle.model_id)['status']=='active'
    assert result['verified'] is False and result['threshold'] is None
    assert restarted.delete(identity)=={'status':'deleted'}
    assert not gallery.repository.rows and not list(gallery.files.templates.iterdir())

@pytest.mark.parametrize('case',['adult','research','persistent'])
def test_all_consents_required(gallery,context,images,case):
    values={'adult_consent':True,'research_opt_in':True,'enrollment_consent':True}
    values[{'adult':'adult_consent','research':'research_opt_in','persistent':'enrollment_consent'}[case]]=False
    with pytest.raises(EnrollmentError) as e: gallery.service.enroll('Fictional Adult',images,context.bundle,**values)
    assert e.value.code=='ENROLLMENT_CONSENT_REQUIRED' and not gallery.repository.rows

@pytest.mark.parametrize('name',['',' bad','Bad/Path','Bad\nName','A'*65,'Name; DROP TABLE x'])
def test_invalid_labels_fail_before_staging(gallery,context,images,name):
    with pytest.raises(EnrollmentError): gallery.service.enroll(name,images,context.bundle,adult_consent=True,research_opt_in=True,enrollment_consent=True)
    assert not list(gallery.files.staging.iterdir())

@pytest.mark.parametrize('days',[0,31,True,'30'])
def test_invalid_expiry(gallery,context,images,days):
    with pytest.raises(EnrollmentError): enroll_core(gallery,context.bundle,images,retention_days=days)

@pytest.mark.parametrize('mode',['none','multiple','clipped'])
def test_original_quality_refusals_remove_stage(gallery,context,images,mode):
    context.bundle.detector.mode=mode
    with pytest.raises(APIError): enroll_core(gallery,context.bundle,images)
    assert not gallery.repository.rows and not list(gallery.files.staging.iterdir())

def test_duplicate_image_rejected(gallery,context,images):
    with pytest.raises(APIError) as e: enroll_core(gallery,context.bundle,[images[0]]*3)
    assert e.value.code=='DUPLICATE_IMAGE' and not list(gallery.files.staging.iterdir())

def test_duplicate_name_and_explicit_reenroll(gallery,context,images):
    result=enroll_core(gallery,context.bundle,images);identity=UUID(result['identity_id'])
    old=gallery.repository.get(identity).template_locator
    with pytest.raises(EnrollmentError) as e: enroll_core(gallery,context.bundle,images)
    assert e.value.code=='DUPLICATE_DISPLAY_NAME'
    new=enroll_core(gallery,context.bundle,images,identity_id=identity)
    assert new['identity_id']==str(identity) and gallery.repository.get(identity).template_locator!=old
    assert not gallery.files.final(old).exists() and len(list(gallery.files.templates.iterdir()))==1

def test_missing_wrong_key_fingerprint_and_revocation(gallery,context,images):
    result=enroll_core(gallery,context.bundle,images);identity=UUID(result['identity_id'])
    for key in (None,Fernet.generate_key()):
        bad=EnrollmentService(gallery.repository,GalleryFiles(gallery.root,key=key))
        with pytest.raises(EnrollmentError): bad.inspect(identity,context.bundle.model_id)
    with pytest.raises(EnrollmentError): gallery.service.inspect(identity,'different_policy')
    assert gallery.service.revoke(identity)['status']=='revoked'
    with pytest.raises(EnrollmentError): gallery.service.inspect(identity,context.bundle.model_id)
    assert not list(gallery.files.templates.iterdir())
    assert enroll_core(gallery,context.bundle,images,identity_id=identity)['status']=='active'

def test_lost_key_does_not_block_normal_deletion(gallery,context,images):
    result=enroll_core(gallery,context.bundle,images)
    lost=EnrollmentService(gallery.repository,GalleryFiles(gallery.root,key=None))
    assert lost.delete(UUID(result['identity_id']))=={'status':'deleted'}
    assert not gallery.repository.rows

def test_expiry_read_refusal_and_operator_cleanup(gallery,context,images):
    result=enroll_core(gallery,context.bundle,images);identity=UUID(result['identity_id'])
    entry=gallery.repository.get(identity)
    gallery.service.clock=lambda:entry.expires_at
    with pytest.raises(EnrollmentError): gallery.service.inspect(identity,context.bundle.model_id)
    assert gallery.service.cleanup_expired()=={'deleted_enrollments':1}
    assert not gallery.repository.rows

def test_corrupt_template_refused_without_vector_exposure(gallery,context,images):
    result=enroll_core(gallery,context.bundle,images);identity=UUID(result['identity_id'])
    gallery.files.final(gallery.repository.get(identity).template_locator).write_bytes(b'corrupt')
    with pytest.raises(EnrollmentError) as e: gallery.service.inspect(identity,context.bundle.model_id)
    assert e.value.code=='ENROLLMENT_REFERENCE_UNAVAILABLE'

@pytest.mark.parametrize('point',['reserve_unknown_commit','promote','activate'])
def test_crash_and_ambiguous_commit_recovery(gallery,context,images,monkeypatch,point):
    if point=='reserve_unknown_commit':
        original=gallery.repository.reserve
        def broken(*args,**kw): original(*args,**kw);raise EnrollmentError('ENROLLMENT_UNAVAILABLE')
        monkeypatch.setattr(gallery.repository,'reserve',broken)
    else:
        target=gallery.files if point=='promote' else gallery.repository
        original=getattr(target,point)
        monkeypatch.setattr(target,point,lambda *a,**k: (_ for _ in ()).throw(EnrollmentError('ENROLLMENT_CLEANUP_PENDING')))
    with pytest.raises(EnrollmentError): enroll_core(gallery,context.bundle,images)
    assert len(gallery.repository.rows)==1 and next(iter(gallery.repository.rows.values())).status=='pending'
    monkeypatch.undo()
    assert gallery.service.recover()['completed_operations']==1
    assert next(iter(gallery.repository.rows.values())).status=='active' and not list(gallery.files.staging.iterdir())

def test_database_failure_orphan_stages_wait_for_age(gallery,context,images,monkeypatch):
    monkeypatch.setattr(gallery.repository,'reserve',lambda *a,**k: (_ for _ in ()).throw(EnrollmentError('ENROLLMENT_UNAVAILABLE')))
    with pytest.raises(EnrollmentError): enroll_core(gallery,context.bundle,images)
    assert not gallery.repository.rows and len(list(gallery.files.staging.iterdir()))==1
    assert gallery.service.recover()['deleted_old_orphan_stages']==0
    now=gallery.service.clock();gallery.service.clock=lambda:now+timedelta(hours=25)
    assert gallery.service.recover()['deleted_old_orphan_stages']==1

def test_deletion_failure_remains_blocked_and_recovers(gallery,context,images,monkeypatch):
    result=enroll_core(gallery,context.bundle,images);identity=UUID(result['identity_id'])
    original=gallery.files.delete
    monkeypatch.setattr(gallery.files,'delete',lambda *a: (_ for _ in ()).throw(EnrollmentError('ENROLLMENT_CLEANUP_PENDING')))
    with pytest.raises(EnrollmentError): gallery.service.delete(identity)
    assert gallery.repository.get(identity).status=='deletion_pending'
    with pytest.raises(EnrollmentError): gallery.service.inspect(identity,context.bundle.model_id)
    monkeypatch.setattr(gallery.files,'delete',original)
    assert gallery.service.recover()['completed_operations']==1 and not gallery.repository.rows

def test_process_lock_rejects_parallel_enrollment(gallery,context,images):
    with gallery.files.locked():
        with pytest.raises(EnrollmentError) as e: enroll_core(gallery,context.bundle,images)
        assert e.value.code=='ENROLLMENT_BUSY'

def test_unrelated_file_never_removed_by_cleanup(gallery,context,images):
    sentinel=gallery.root/'unrelated.keep';sentinel.write_text('unrelated')
    result=enroll_core(gallery,context.bundle,images);gallery.service.delete(UUID(result['identity_id']))
    assert sentinel.read_text()=='unrelated'
