"""Load scoped local owner/runtime secrets for one child command; never print them."""
import argparse, json, os, subprocess, sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope',choices=['runtime','owner','test'])
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.command:
        parser.error('Supply a command to run.')
    try:
        secrets=json.loads((Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase4/postgresql/credentials.json').read_text())
        env=os.environ.copy()
        for name in ('FACEATTEND_DATABASE_URL','FACEATTEND_MIGRATION_DATABASE_URL','FACEATTEND_TEST_OWNER_URL','FACEATTEND_TEST_RUNTIME_URL'):
            env.pop(name,None)
        if args.scope=='runtime':
            env['FACEATTEND_DATABASE_URL']=secrets['runtime']
        elif args.scope=='owner':
            env['FACEATTEND_MIGRATION_DATABASE_URL']=secrets['owner']
        else:
            env.update(FACEATTEND_TEST_OWNER_URL=secrets['test_owner'],FACEATTEND_TEST_RUNTIME_URL=secrets['test_runtime'])
        return subprocess.run(args.command,env=env).returncode
    except Exception:
        print('Local database credentials or child command are unavailable.',file=sys.stderr)
        return 1

if __name__=='__main__': sys.exit(main())
