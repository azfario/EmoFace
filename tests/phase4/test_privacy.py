import json
import pytest
from faceattend_db.config import DBSettings
from faceattend_db.instrumentation import Telemetry
from faceattend_db.models import Base
from faceattend_db.session import make_engine
from tests.phase3.conftest import CONSENT, upload

def test_configured_credentials_not_in_responses_openapi_or_logs(context,monkeypatch,caplog):
    url='postgresql+psycopg://PRIVATE_ACCOUNT:NEVER_DISCLOSE@127.0.0.1:55432/private_db'
    service=Telemetry(DBSettings(enabled=True,url=url))
    monkeypatch.setattr('faceattend_db.instrumentation.make_engine',lambda _: (_ for _ in ()).throw(RuntimeError(url)))
    service.start(); context.app.state.telemetry=service
    output=''.join(context.client.get(path).text for path in ['/health','/health/db','/api/v1/models/info','/openapi.json','/docs'])
    response=context.client.post('/api/v1/face/detect',data=CONSENT,files={'image':upload()})
    assert response.status_code==200
    for forbidden in ('NEVER_DISCLOSE','PRIVATE_ACCOUNT','private_db',url):
        assert forbidden not in output+response.text+caplog.text+repr(service.settings)

def test_no_biometric_or_person_table_metadata():
    assert set(Base.metadata.tables)=={'research_daily_operation_stats'}
    allowed={'day_utc','operation','outcome','request_count','total_processing_ms'}
    assert set(Base.metadata.tables['research_daily_operation_stats'].columns.keys())==allowed

def test_shutdown_disposes_engine_without_creating_schema(monkeypatch):
    from unittest.mock import MagicMock
    engine=MagicMock()
    monkeypatch.setattr('faceattend_db.instrumentation.make_engine',lambda _:engine)
    monkeypatch.setattr('faceattend_db.instrumentation.runtime_ready',lambda _:True)
    service=Telemetry(DBSettings(enabled=True,url='postgresql+psycopg://u:secret@127.0.0.1:55432/test'))
    service.start(); assert service.status=='ready'
    service.close(); engine.dispose.assert_called_once()
    assert service.engine is None and service.repository is None

def test_utc_bucket_date_and_processing_range(monkeypatch):
    from unittest.mock import MagicMock
    from faceattend_db.repository import Repository
    from datetime import datetime, timezone
    f=MagicMock()
    Repository(f).increment_daily('face_detect','completed',0)
    stmt=f.return_value.__enter__.return_value.execute.call_args.args[0]
    assert stmt.compile().params['day_utc']==datetime.now(timezone.utc).date()
