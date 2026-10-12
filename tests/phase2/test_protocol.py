import json
from pathlib import Path
import pytest
from face_verification.protocol import load_protocol
from face_verification import Status, VerificationError

def manifest(tmp_path):
    for i in range(4):
        (tmp_path/f'{i}.jpg').write_bytes(b'synthetic placeholder for path validation only')
    value = {'schema_version':1,'authorized_noncommercial_research':True,'participants':[{'user_id':'volunteer_001',
        'adult_consent':True,'identity_consent':True,'split':'development',
        'enrollment':[{'path':f'{i}.jpg','session':'A'} for i in range(3)],
        'probes':[{'path':'3.jpg','session':'B'}]}]}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(value))
    return value,path

def test_valid_private_manifest(tmp_path):
    value,path = manifest(tmp_path)
    result = load_protocol(path,consent_confirmed=True)
    assert result['participants'][0]['enrollment'][0]['_resolved_path'].is_absolute()

def test_no_consent_before_any_file_access(tmp_path):
    with pytest.raises(VerificationError) as exc:
        load_protocol(tmp_path/'absent.json')
    assert exc.value.status == Status.CONSENT_REQUIRED

@pytest.mark.parametrize('change',['duplicate_path','duplicate_subject','probe_session','minor','purpose'])
def test_invalid_manifest(tmp_path,change):
    value,path = manifest(tmp_path)
    person = value['participants'][0]
    if change == 'duplicate_path':
        person['probes'][0]['path'] = person['enrollment'][0]['path']
    if change == 'duplicate_subject':
        value['participants'].append(dict(person,split='test'))
    if change == 'probe_session':
        person['probes'][0]['session'] = 'A'
    if change == 'minor':
        person['adult_consent'] = False
    if change == 'purpose':
        value['authorized_noncommercial_research'] = False
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError,VerificationError)):
        load_protocol(path,consent_confirmed=True)
