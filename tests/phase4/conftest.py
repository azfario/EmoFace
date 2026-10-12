from tests.phase3.conftest import context
import pytest

@pytest.fixture(autouse=True)
def isolated_operator_configuration(monkeypatch):
    # Direct Phase 4 pytest invocation also must not inherit production writes.
    monkeypatch.setenv('FACEATTEND_DB_TELEMETRY_ENABLED','false')
    monkeypatch.setenv('FACEATTEND_DB_RETENTION_DAYS','30')
    monkeypatch.delenv('FACEATTEND_DATABASE_URL',raising=False)
