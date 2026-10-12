from pathlib import Path
import json
import pytest
from faceattend_enrollment.operator import setup_secrets
from faceattend_enrollment.auth import OperatorAuth
from faceattend_enrollment.models import Identity,EnrollmentBase
from faceattend_enrollment.security import read_secret
from faceattend_enrollment.errors import EnrollmentError
from .conftest import PASSWORD

def test_schema_has_only_minimal_registration_metadata():
    table=Identity.__table__
    assert set(table.columns.keys())=={'identity_id','display_name','template_locator','consent_recorded_at','purpose_version','enrolled_at','expires_at','status','model_policy_fingerprint'}
    assert set(EnrollmentBase.metadata.tables)=={'demo.demo_enrolled_identities'}
    assert all(not c.nullable for c in table.columns)

def test_operator_supplied_setup_separates_keys_and_never_overwrites(private_root):
    gallery=private_root/'gallery';key=private_root/'secrets/key.key';auth=private_root/'secrets/operator.secret'
    setup_secrets(gallery,key,auth,PASSWORD)
    assert key.is_file() and auth.is_file() and not list(gallery.iterdir())
    assert PASSWORD not in auth.read_text()
    assert OperatorAuth.from_file(auth).login(PASSWORD)
    with pytest.raises(EnrollmentError):setup_secrets(gallery,key,auth,'Another fictional password')
    assert not any(p.suffix.lower() in {'.jpg','.jpeg','.png'} for p in private_root.rglob('*'))

def test_secret_size_bound_and_private_storage(private_root):
    oversized=private_root/'oversized.secret';oversized.write_bytes(b'PRIVATE_SECRET'*500)
    with pytest.raises(EnrollmentError) as e: read_secret(oversized)
    assert 'PRIVATE_SECRET' not in str(e.value)
