"""Run source-checkout checks without original-laptop reports or private data."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend',action='store_true',help='Also run npm tests and a production build; install frontend packages first.')
    args=parser.parse_args()
    # Historical receipts are not prerequisites for development on a new PC.
    # Only create empty output directories; do not read original credentials/data.
    for phase in ('phase2','phase3','phase4','phase4b','phase4c','phase4d','phase5'):
        (ROOT/'reports'/phase).mkdir(parents=True,exist_ok=True)
    temp=ROOT/'.cache/source_checks/tmp';temp.mkdir(parents=True,exist_ok=True)
    env={name:value for name,value in os.environ.items() if not name.startswith('FACEATTEND_')}
    env.update(TEMP=str(temp),TMP=str(temp),MPLCONFIGDIR=str(ROOT/'.cache/source_checks/matplotlib'))
    # Private-image, database and saved-model checks remain explicit separate opt-ins.
    # Existing synthetic CUDA tests may run when CUDA is available.
    code=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','--basetemp',str(ROOT/'.cache/source_checks/pytest')],cwd=ROOT,env=env).returncode
    if code:return code
    if args.frontend:
        npm=shutil.which('npm.cmd' if os.name=='nt' else 'npm')
        if not npm:
            print('Node/npm unavailable; install the documented runtime before frontend checks.',file=sys.stderr);return 1
        for command in ([npm,'--prefix','frontend','test'],[npm,'--prefix','frontend','run','build']):
            code=subprocess.run(command,cwd=ROOT,env=env).returncode
            if code:return code
    return 0

if __name__=='__main__':sys.exit(main())
