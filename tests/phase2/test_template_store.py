import json
import numpy as np
import pytest
from cryptography.fernet import Fernet
from face_verification import TemplateStore, Status, VerificationError, delete_enrollment

def test_roundtrip_encrypted(enrolled):
    value = enrolled.load('volunteer_001')
    np.testing.assert_allclose(value['embedding'], [1,0,0])
    content = next(enrolled.directory.glob('*.fernet')).read_bytes()
    assert b'volunteer_001' not in content and b'embedding' not in content
    with pytest.raises((ValueError,UnicodeDecodeError)):
        json.loads(content)

def test_corruption(enrolled):
    next(enrolled.directory.glob('*.fernet')).write_bytes(b'broken')
    with pytest.raises(VerificationError) as exc:
        enrolled.load('volunteer_001')
    assert exc.value.status == Status.CORRUPT_TEMPLATE

def test_wrong_key(enrolled):
    other = TemplateStore(enrolled.directory, key=Fernet.generate_key())
    with pytest.raises(VerificationError) as exc:
        other.load('volunteer_001')
    assert exc.value.status == Status.CORRUPT_TEMPLATE

def test_missing_key(tmp_path,monkeypatch):
    monkeypatch.delenv('FACEATTEND_TEMPLATE_KEY',raising=False)
    with pytest.raises(VerificationError) as exc:
        TemplateStore(tmp_path)
    assert exc.value.status == Status.KEY_UNAVAILABLE

def test_delete(enrolled):
    assert delete_enrollment('volunteer_001',enrolled).deleted
    assert delete_enrollment('volunteer_001',enrolled).status == Status.UNKNOWN_ID

def test_retention_expires(store):
    from datetime import datetime, timezone, timedelta
    recorded = (datetime.now(timezone.utc)-timedelta(days=31)).isoformat()
    store.save('volunteer_001',[1,0,0],model_id='test',sample_hashes=['a','b','c'],
        consent={'adult':True,'identity_verification':True,'recorded_at':recorded})
    with pytest.raises(VerificationError) as exc:
        store.load('volunteer_001')
    assert exc.value.status == Status.TEMPLATE_EXPIRED
    assert delete_enrollment('volunteer_001',store).deleted

@pytest.mark.parametrize('user', ['../bad','Jane Doe','student@school.org','0001','volunteer_../../x'])
def test_invalid_id(store,user):
    with pytest.raises(VerificationError):
        store.load(user)
