"""First-use/reuse/fail-closed launcher checks, without Docker or private inputs."""
import importlib.util,io,json,os,shutil,subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
PASSWORD='Fictional launcher passphrase 你好'

@pytest.fixture
def bootstrap(tmp_path):
    spec=importlib.util.spec_from_file_location('launcher_bootstrap',ROOT/'docker/bootstrap_secrets.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    directory=tmp_path/'secrets';directory.mkdir();module.DIRECTORY=directory
    return module

def create(module,password=PASSWORD):
    module.create(io.BytesIO(json.dumps({'password':password}).encode('utf-8')))

def test_new_secrets_unicode_auth_and_readonly_validation(bootstrap):
    create(bootstrap);bootstrap.validate()
    before={p.name:p.read_bytes() for p in bootstrap.DIRECTORY.iterdir()}
    assert set(before)==set(bootstrap.NAMES)
    assert all(PASSWORD.encode('utf-8') not in raw for raw in before.values())
    auth=bootstrap.OperatorAuth(json.loads(before['operator_auth']))
    auth.login(PASSWORD)
    bootstrap.validate()
    assert before=={p.name:p.read_bytes() for p in bootstrap.DIRECTORY.iterdir()}
    if os.name!='nt':assert all(p.stat().st_mode & 0o777==0o644 for p in bootstrap.DIRECTORY.iterdir())

def test_database_initialization_refuses_unreadable_secret_files():
    script=(ROOT/'docker/init_database.sh').read_text()
    assert '[ ! -r "$secret_file" ]' in script and '[ ! -s "$secret_file" ]' in script
    assert script.index('[ ! -r "$secret_file" ]') < script.index('CREATE ROLE')

@pytest.mark.parametrize('password',['short','x'*257])
def test_invalid_password_leaves_no_files(bootstrap,password):
    with pytest.raises(Exception):create(bootstrap,password)
    assert not list(bootstrap.DIRECTORY.iterdir())

def test_existing_or_partial_secrets_never_regenerated(bootstrap):
    create(bootstrap)
    before={p.name:p.read_bytes() for p in bootstrap.DIRECTORY.iterdir()}
    with pytest.raises(Exception):create(bootstrap)
    assert before=={p.name:p.read_bytes() for p in bootstrap.DIRECTORY.iterdir()}
    (bootstrap.DIRECTORY/'db_admin_password').unlink()
    with pytest.raises(Exception):bootstrap.validate()
    with pytest.raises(Exception):create(bootstrap)
    assert all(p.read_bytes()==before[p.name] for p in bootstrap.DIRECTORY.iterdir())

@pytest.mark.parametrize('name', ['operator_auth','gallery_key','db_runtime_password'])
def test_bad_record_key_or_database_secret_refused(bootstrap,name):
    create(bootstrap);(bootstrap.DIRECTORY/name).write_bytes(b'invalid')
    with pytest.raises(Exception):bootstrap.validate()

def test_oversized_stdin_or_unexpected_fields_refused(bootstrap):
    for value in (b'x'*8193,json.dumps({'password':PASSWORD,'extra':True}).encode()):
        with pytest.raises(Exception):bootstrap.create(io.BytesIO(value))
    assert not list(bootstrap.DIRECTORY.iterdir())

def test_cli_errors_never_echo_input(bootstrap,monkeypatch,capsys):
    monkeypatch.setattr('sys.argv',['bootstrap','create'])
    monkeypatch.setattr('sys.stdin',type('Input',(),{'buffer':io.BytesIO(b'PRIVATE_TEST_SENTINEL')})())
    assert bootstrap.main()==1
    output=capsys.readouterr();assert 'PRIVATE_TEST_SENTINEL' not in output.out+output.err

@pytest.mark.skipif(os.name!='nt',reason='Windows launcher requires PowerShell and NTFS.')
@pytest.mark.parametrize('scenario', ['fresh','reuse','retained','incomplete','remote','occupied','gpu_failure','startup_failure','running','synced','missing_model','partial','stop','acl_good','acl_bad'])
def test_windows_orchestration_fail_closed_and_reuse(tmp_path,scenario):
    powershell=shutil.which('powershell.exe')
    if not powershell:pytest.skip('Windows PowerShell unavailable.')
    models=tmp_path/'models'
    for relative in ('faceattend_best.pt','external/insightface/models/buffalo_l/det_10g.onnx','external/insightface/models/buffalo_l/w600k_r50.onnx'):
        p=models/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fictional model marker')
    if scenario=='missing_model':(models/'faceattend_best.pt').unlink()
    command=[powershell,'-NoProfile','-NonInteractive','-File',str(ROOT/'tests/phase6/launcher_harness.ps1'),'-Library',str(ROOT/'docker/Launcher.ps1'),'-FixtureRoot',str(tmp_path),'-Scenario',scenario]
    result=subprocess.run(command,capture_output=True,timeout=30)
    assert result.returncode==0, 'Fictional launcher harness failed; diagnostics withheld.'
    raw=result.stdout.decode('utf-8-sig',errors='replace')
    # Windows PowerShell console encoding can be legacy; the final JSON is ASCII.
    value=json.loads(raw.strip().splitlines()[-1])
    assert value['failed']==(scenario not in ('fresh','reuse','running','stop','acl_good')), value['failure']
    assert value['destructive_calls']==(1 if scenario=='stop' else 0) and not value['plaintext_in_argv']
    assert value['secret_runs_restricted']
    if scenario=='fresh':
        assert value['create_calls']==1 and value['prompts']==0 and value['payload_valid']
        assert value['validate_calls']==1 and value['startup_calls']==1
    if scenario=='reuse':assert value['create_calls']==0 and value['prompts']==0 and value['validate_calls']==1
    if scenario in ('retained','incomplete','remote','occupied','gpu_failure','running','synced','missing_model','partial','stop'):
        assert value['create_calls']==0 and value['directory_creations']==0 and value['startup_calls']==0


def test_fresh_password_free_setup_keeps_legacy_layout(bootstrap):
    bootstrap.create(io.BytesIO(b'{}'));bootstrap.validate()
    assert set(p.name for p in bootstrap.DIRECTORY.iterdir())==set(bootstrap.NAMES)
