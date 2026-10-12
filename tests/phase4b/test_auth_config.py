from pathlib import Path
from dataclasses import replace
import json
import pytest
from starlette.requests import Request
from faceattend_enrollment.auth import OperatorAuth,password_record,COOKIE
from faceattend_enrollment.config import EnrollmentSettings
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_enrollment.errors import EnrollmentError
from faceattend_enrollment.security import protected_path,read_secret
from .conftest import PASSWORD

def request(token='',csrf=''):
    return Request({'type':'http','headers':[(b'cookie',f'{COOKIE}={token}'.encode()),(b'x-faceattend-csrf',csrf.encode())]})

def test_default_disabled_no_engine_or_secret_reads(monkeypatch):
    def forbidden(*args): raise AssertionError('disabled must not load secrets/connect')
    monkeypatch.setattr('faceattend_enrollment.dependencies.make_engine',forbidden)
    monkeypatch.setattr('faceattend_enrollment.dependencies.read_secret',forbidden)
    deps=EnrollmentDependencies.from_env();deps.start()
    assert deps.engine is None and deps.service is None and deps.error=='ENROLLMENT_DISABLED'

@pytest.mark.parametrize('flag',['true','True','1','yes'])
def test_invalid_or_incomplete_enabled_config_fail_closed(monkeypatch,flag):
    monkeypatch.setenv('FACEATTEND_ENROLLMENT_ENABLED',flag)
    deps=EnrollmentDependencies.from_env();deps.start()
    assert deps.service is None and deps.error=='ENROLLMENT_UNAVAILABLE'

@pytest.mark.parametrize('path',['relative.key','C:/Users/User/OneDrive/private.key',r'\\remote\share\key'])
def test_unsafe_paths_refused_without_echo(path):
    with pytest.raises(EnrollmentError) as e: protected_path(path)
    assert path not in str(e.value)

@pytest.mark.parametrize('days',[0,31,True,1.5])
def test_retention_ceiling(days):
    with pytest.raises(EnrollmentError): EnrollmentSettings(retention_days=days)

def test_key_cannot_be_next_to_templates(private_root):
    gallery=private_root/'gallery';gallery.mkdir()
    with pytest.raises(EnrollmentError):
        EnrollmentSettings(enabled=True,database_url='postgresql+psycopg://test:PRIVATE_SECRET@127.0.0.1/test',
            gallery_dir=gallery,key_file=gallery/'key',auth_file=private_root/'auth')

def test_missing_secret_failure_sanitized(private_root):
    with pytest.raises(EnrollmentError) as e: read_secret(private_root/'PRIVATE_SECRET_MISSING')
    assert 'PRIVATE_SECRET' not in str(e.value)

def test_password_record_never_plaintext():
    value=password_record(PASSWORD)
    assert PASSWORD not in json.dumps(value) and set(value)=={'version','salt','digest'}

def test_session_csrf_expiry_replacement_logout():
    now=[0.0];auth=OperatorAuth(password_record(PASSWORD),clock=lambda:now[0])
    token,csrf=auth.login(PASSWORD);auth.authorize(request(token,csrf),True)
    with pytest.raises(EnrollmentError) as e: auth.authorize(request(token,'bad'),True)
    assert e.value.code=='CSRF_REQUIRED'
    new,other=auth.login(PASSWORD)
    with pytest.raises(EnrollmentError): auth.authorize(request(token,csrf))
    auth.authorize(request(new,other))
    now[0]=901
    with pytest.raises(EnrollmentError): auth.authorize(request(new,other))
    now[0]=1000;token,csrf=auth.login(PASSWORD);auth.logout()
    with pytest.raises(EnrollmentError): auth.authorize(request(token,csrf))

def test_login_rate_limit_bounds_work():
    auth=OperatorAuth(password_record(PASSWORD))
    for _ in range(5):
        with pytest.raises(EnrollmentError) as e: auth.login('wrong')
        assert e.value.code=='OPERATOR_AUTH_REQUIRED'
    with pytest.raises(EnrollmentError) as e: auth.login(PASSWORD)
    assert e.value.code=='RATE_LIMITED'

def test_mutation_rate_limit():
    auth=OperatorAuth(password_record(PASSWORD));token,csrf=auth.login(PASSWORD)
    for _ in range(20): auth.authorize(request(token,csrf),True)
    with pytest.raises(EnrollmentError) as e: auth.authorize(request(token,csrf),True)
    assert e.value.code=='RATE_LIMITED'

def test_restart_clears_sessions():
    record=password_record(PASSWORD);auth=OperatorAuth(record);token,csrf=auth.login(PASSWORD)
    with pytest.raises(EnrollmentError): OperatorAuth(record).authorize(request(token,csrf))
