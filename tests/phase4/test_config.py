import logging
import pytest
from faceattend_db.config import DBSettings, local_url
from faceattend_db.instrumentation import Telemetry
from faceattend_db.session import make_engine

URL='postgresql+psycopg://runtime:SECRET@127.0.0.1:55432/test'

def test_default_off_no_engine_or_connection(monkeypatch):
    monkeypatch.delenv('FACEATTEND_DB_TELEMETRY_ENABLED',raising=False)
    monkeypatch.setenv('FACEATTEND_DATABASE_URL','SECRET malformed ignored')
    def forbidden(*args): raise AssertionError('disabled must never create engine')
    monkeypatch.setattr('faceattend_db.instrumentation.make_engine',forbidden)
    service=Telemetry.from_env()
    service.start(); service.record('face_detect','completed',10)
    assert service.health()=='disabled' and service.engine is None
    service.close()

@pytest.mark.parametrize('url',[
    'postgresql://u:SECRET@127.0.0.1/test', 'postgresql+psycopg://u:SECRET@remote.example/test',
    'postgresql+psycopg://u:SECRET@localhost/test', 'sqlite:///private',
    'postgresql+psycopg://u:SECRET@127.0.0.1:80/test',
    'postgresql+psycopg://u:SECRET@127.0.0.1/test?host=remote',
    'postgresql+psycopg://u@127.0.0.1/test', 'SECRET',None])
def test_bad_urls_sanitized(url):
    with pytest.raises(ValueError) as e: local_url(url)
    assert 'SECRET' not in str(e.value) and 'remote' not in str(e.value)

@pytest.mark.parametrize('flag',['True','1','yes',''])
def test_invalid_flag_unavailable_not_enabled(monkeypatch,flag):
    monkeypatch.setenv('FACEATTEND_DB_TELEMETRY_ENABLED',flag)
    service=Telemetry.from_env()
    service.start()
    assert service.health()=='unavailable' and service.engine is None

@pytest.mark.parametrize('days',[0,366,True,1.1,'30'])
def test_retention_validation(days):
    with pytest.raises(ValueError): DBSettings(retention_days=days)

def test_url_repr_and_timeouts_no_secrets():
    settings=DBSettings(enabled=True,url=URL)
    assert 'SECRET' not in repr(settings)
    engine=make_engine(settings.url)
    assert engine.hide_parameters and not engine.echo and engine.pool.size()==2 and engine.pool.timeout()==1
    engine.dispose()

def test_database_exception_log_sanitized(monkeypatch,caplog):
    def fail(*args): raise RuntimeError('SECRET private portrait vector')
    monkeypatch.setattr('faceattend_db.instrumentation.make_engine',fail)
    service=Telemetry(DBSettings(enabled=True,url=URL))
    service.start(); service.start()
    assert service.health()=='unavailable'
    assert 'SECRET' not in caplog.text and caplog.text.count('model processing continues')==1
