"""Record versions and reproducible additive installation without changing Phase 1."""
import importlib.metadata as meta
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/phase2'
REPORTS.mkdir(parents=True,exist_ok=True)
baseline = json.loads((REPORTS/'phase1_integrity.json').read_text())
packages = {d.metadata['Name'].lower():d.version for d in meta.distributions()}
added = {name:version for name,version in packages.items() if name not in baseline['packages']}
requirements = '# Additions to the preserved Phase 1 .venv. InsightFace installed separately with --no-deps.\n'
requirements += '\n'.join(f'{name}=={version}' for name,version in sorted(added.items()) if name != 'insightface')+'\n'
(ROOT/'requirements-phase2.txt').write_text(requirements)
(ROOT/'requirements-phase2-insightface.txt').write_text('# Install ONLY with --no-deps to avoid installing the conflicting CPU runtime.\ninsightface==1.0.1\n')
(ROOT/'requirements-phase2-lock.txt').write_text('# Observed complete environment; version inventory, not a standalone installer.\n'+'\n'.join(f'{name}=={version}' for name,version in sorted(packages.items()))+'\n')
(ROOT/'requirements-phase2-constraints.txt').write_text('# Preserve existing Phase 1 versions when adding packages.\n'+'\n'.join(f'{name}=={version}' for name,version in sorted(baseline['packages'].items()) if name != 'faceattend')+'\n')
import torch
import onnxruntime as ort
smi = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader'],capture_output=True,text=True)
check = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
report = {'python':sys.version.split()[0], 'interpreter':str(Path(sys.executable).relative_to(ROOT)),
    'torch':torch.__version__, 'torchvision':packages['torchvision'], 'cuda':torch.version.cuda,
    'cudnn':torch.backends.cudnn.version(), 'onnxruntime':ort.__version__, 'gpu':smi.stdout.strip(),
    'added_packages':added,'existing_packages_changed':{p:[v,packages.get(p)] for p,v in baseline['packages'].items() if packages.get(p)!=v},
    'cpu_onnxruntime_distribution_installed':'onnxruntime' in packages,
    'pip_check':{'exit_code':check.returncode,'output':check.stdout.strip(),
        'explanation':'Expected InsightFace dependency-metadata warning: official GPU setup replaces CPU onnxruntime with onnxruntime-gpu, which supplies the same Python module. No additional conflicts allowed.'},
    'vscode_kernel':'faceattend-gpu','vscode_gui_selection':'MANUAL_NOT_OBSERVED'}
(REPORTS/'environment.json').write_text(json.dumps(report,indent=2))
unexpected = [line for line in check.stdout.splitlines() if line != 'insightface 1.0.1 requires onnxruntime, which is not installed.']
if report['existing_packages_changed'] or report['cpu_onnxruntime_distribution_installed'] or unexpected:
    raise SystemExit('Unexpected environment change or conflict; stop before inference.')
print(json.dumps(report,indent=2))
