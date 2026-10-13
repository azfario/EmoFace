"""Clone-time model hydration and immutable runtime files, using fictional bytes."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
PAIRS={'model-assets/faceattend_best.pt':'faceattend_best.pt',
       'model-assets/det_10g.onnx':'external/insightface/models/buffalo_l/det_10g.onnx',
       'model-assets/w600k_r50.onnx':'external/insightface/models/buffalo_l/w600k_r50.onnx'}

def test_only_reviewed_assets_are_lfs_and_docker_excludes_them():
    attributes=(ROOT/'.gitattributes').read_text().splitlines()
    assert attributes==[p+' filter=lfs diff=lfs merge=lfs -text' for p in PAIRS]
    ignore=(ROOT/'.gitignore').read_text().splitlines()
    assert all('!/'+p in ignore for p in PAIRS)
    assert 'models/' in ignore and '*.pt' in ignore and '*.onnx' in ignore
    assert 'model-assets/' in (ROOT/'.dockerignore').read_text().splitlines()
    for name in ('Dockerfile.api','Dockerfile.frontend'):
        assert 'COPY model-assets' not in (ROOT/'docker'/name).read_text()
    manifest=json.loads((ROOT/'configs/model-assets.json').read_text())
    assert manifest['version']==1 and len(manifest['files'])==3
    assert {item['asset']:item['runtime'] for item in manifest['files']}==PAIRS
    assert all(len(item['sha256'])==64 and 0<item['bytes']<=268435456 for item in manifest['files'])

@pytest.mark.skipif(os.name!='nt',reason='Windows model bootstrap requires PowerShell.')
@pytest.mark.parametrize('scenario',['fresh','existing','partial','pointer','cloud','junction','hash_bad','bad_path','bad_size','empty_runtime','download_failed'])
def test_hydration_checks_hashes_and_never_replaces_models(tmp_path,scenario):
    shell=shutil.which('powershell.exe')
    if not shell:pytest.skip('Windows PowerShell unavailable.')
    rows=[]
    for asset,runtime in PAIRS.items():
        raw=('fixture-'+runtime).encode()
        p=tmp_path/asset;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        rows.append({'asset':asset,'runtime':runtime,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    configs=tmp_path/'configs';configs.mkdir()
    if scenario in ('existing','partial','empty_runtime'):
        for i,runtime in enumerate(PAIRS.values()):
            if scenario in ('partial','empty_runtime') and i>0:continue
            p=tmp_path/'models'/runtime;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'' if scenario=='empty_runtime' else b'preserved-custom-model')
    if scenario in ('pointer','download_failed'):
        (tmp_path/next(iter(PAIRS))).write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+'0'*64+'\nsize 123\n')
    if scenario=='hash_bad':(tmp_path/next(iter(PAIRS))).write_bytes(b'wrong-model')
    if scenario=='bad_path':rows[0]['runtime']='../outside-fixture.pt'
    if scenario=='bad_size':rows[0]['bytes']=True
    (configs/'model-assets.json').write_text(json.dumps({'version':1,'files':rows}))
    result=subprocess.run([shell,'-NoProfile','-NonInteractive','-File',str(ROOT/'tests/phase6/model_assets_harness.ps1'),
        '-Library',str(ROOT/'docker/ModelAssets.ps1'),'-FixtureRoot',str(tmp_path),'-Scenario',scenario],capture_output=True,timeout=30)
    assert result.returncode==0,'Fictional model harness failed; raw diagnostics withheld.'
    value=json.loads(result.stdout.decode('utf-8-sig',errors='replace').strip().splitlines()[-1])
    bad=scenario in ('hash_bad','bad_path','bad_size','empty_runtime','download_failed','junction')
    assert value['failed']==bad and value['existing_preserved'],value['failure']
    assert value['downloads']==(1 if scenario in ('pointer','download_failed') else 0)
    assert value['runtime_files']==(1 if scenario=='empty_runtime' else 0 if bad else 3)
    assert not (tmp_path/'outside-fixture.pt').exists()
