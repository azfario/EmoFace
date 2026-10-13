"""Static container trust-boundary checks; live GPU/Compose evidence is separate."""
from pathlib import Path
import yaml
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def stack():return yaml.safe_load((ROOT/'compose.yaml').read_text(encoding='utf-8'))

def test_only_frontend_publishes_loopback_and_namespace_is_guarded(stack):
    assert stack['services']['frontend']['ports']==['127.0.0.1:18080:8080']
    assert stack['networks']['local_only']['enable_ipv6'] is False
    assert stack['services']['network_guard']['cap_add']==['NET_ADMIN']
    assert stack['services']['network_guard']['network_mode']=='service:frontend'
    assert stack['services']['api']['depends_on']['network_guard']['condition']=='service_completed_successfully'
    for name in ('api','db','migrate'):
        assert 'ports' not in stack['services'][name]
        assert stack['services'][name]['network_mode']=='service:frontend'

def test_only_model_worker_gets_gpu_and_no_native_database_mounts(stack):
    for name,service in stack['services'].items():
        if name=='api':assert service['deploy']['resources']['reservations']['devices']==[{'driver':'nvidia','count':1,'capabilities':['gpu']}]
        else:assert 'deploy' not in service
    assert not any('55432' in str(service) or 'postgresql/data' in str(service) for service in stack['services'].values())
    mount=next(v for v in stack['services']['api']['volumes'] if isinstance(v,dict))
    assert mount['read_only'] is True
    assert mount['bind']['create_host_path'] is False

def test_secrets_scoped_and_no_literal_credentials_or_auto_restart(stack):
    assert stack['services']['migrate']['secrets']==['db_owner_password']
    assert set(stack['services']['api']['secrets'])=={'db_runtime_password','gallery_key','operator_auth'}
    assert 'gallery_key' not in stack['services']['db']['secrets']
    for service in stack['services'].values():
        assert service['restart']=='no'
        assert not any(k.endswith('_PASSWORD') for k in service.get('environment',{}))
    assert stack['services']['api']['environment']['FACEATTEND_DB_TELEMETRY_ENABLED']=='false'

def test_proxy_preserves_host_origin_and_discards_untrusted_forwarded_headers():
    config=(ROOT/'docker/nginx.conf').read_text()
    assert 'proxy_pass http://127.0.0.1:8000' in config
    assert 'proxy_set_header Host $http_host;' in config
    assert 'proxy_set_header Origin $http_origin;' in config
    for header in ('X-Forwarded-For','X-Forwarded-Host','X-Forwarded-Proto','Forwarded'):
        assert f'proxy_set_header {header} "";' in config
    assert 'access_log off;' in config and 'error_log /dev/null;' in config
    assert 'default 0;' in config and 'return 403;' in config

def test_native_local_admission_and_cuda_policy_not_replaced():
    runtime=(ROOT/'docker/runtime.py').read_text()
    assert "host='127.0.0.1'" in runtime and 'proxy_headers=False' in runtime
    assert "if not torch.cuda.is_available():raise" in runtime
    assert "'faceattend_runtime'" in runtime and "'faceattend_migrator'" in runtime

def test_builds_exclude_private_assets_and_use_digest_pins():
    ignore=(ROOT/'.dockerignore').read_text()
    for path in ('models/','reports/','docs/','data/','outputs/','.venv/','.cache/','.git/','frontend/node_modules/'):
        assert path in ignore.splitlines()
    for name in ('Dockerfile.api','Dockerfile.frontend'):
        dockerfile=(ROOT/'docker'/name).read_text()
        assert all('@sha256:' in line for line in dockerfile.splitlines() if line.startswith('FROM '))
        assert 'COPY models' not in dockerfile and 'COPY . ' not in dockerfile
        assert 'USER ' in dockerfile and 'ARG ' not in dockerfile

def test_outbound_guard_is_namespace_scoped_and_startup_checks_its_marker():
    guard=(ROOT/'docker/network_guard.sh').read_text()
    assert 'iptables -P OUTPUT DROP' in guard and 'ip6tables -P OUTPUT DROP' in guard
    assert '-o lo -j ACCEPT' in guard and 'ESTABLISHED,RELATED' in guard
    assert '-d 127.0.0.11 -j DROP' in guard
    assert '/proc/self/ns/net' in guard
    runtime=(ROOT/'docker/runtime.py').read_text()
    assert "Path('/netstate/namespace').read_text().strip()!=os.readlink('/proc/self/ns/net')" in runtime

def test_linux_shell_scripts_keep_linux_line_endings():
    attributes=(ROOT/'.gitattributes').read_text().splitlines()
    assert '*.sh text eol=lf' in attributes
    for path in (ROOT/'docker').glob('*.sh'):
        content=path.read_bytes()
        assert content.startswith(b'#!/bin/sh\n'), path.name
        assert b'\r' not in content, path.name

def test_static_frontend_security_matches_native_policy():
    from faceattend_ui.integration import CSP
    config=(ROOT/'docker/nginx.conf').read_text()
    assert f'add_header Content-Security-Policy "{CSP}" always;' in config
    assert 'add_header X-Frame-Options "DENY" always;' in config

def test_keys_use_separate_memory_mount_and_full_linux_inventory(stack):
    runtime=(ROOT/'docker/runtime.py').read_text()
    assert "Path('/run/faceattend-private')" in runtime
    assert any(v.startswith('/run/faceattend-private:') for v in stack['services']['api']['tmpfs'])
    lock=(ROOT/'docker/requirements-linux-lock.txt').read_text(encoding='utf-8-sig')
    assert 'torch==2.13.0+cu126' in lock and 'nvidia-cudnn-cu12==9.10.2.21' in lock
    assert 'onnxruntime==' not in lock and 'onnxruntime-gpu==1.20.2' in lock
