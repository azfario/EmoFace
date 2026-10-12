"""Scoped credentials from the established external source; no secret output."""
from pathlib import Path
import argparse,json,os,subprocess,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope',choices=['runtime','owner','test'])
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.command: parser.error('Supply one child command.')
    try:
        private=json.loads((Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase4/postgresql/credentials.json').read_text())
        env=os.environ.copy()
        for name in ('FACEATTEND_DATABASE_URL','FACEATTEND_ENROLLMENT_DATABASE_URL','FACEATTEND_MIGRATION_DATABASE_URL',
                     'FACEATTEND_TEST_OWNER_URL','FACEATTEND_TEST_RUNTIME_URL'):
            env.pop(name,None)
        if args.scope=='runtime': env['FACEATTEND_ENROLLMENT_DATABASE_URL']=private['runtime']
        elif args.scope=='owner': env['FACEATTEND_MIGRATION_DATABASE_URL']=private['owner']
        else: env.update(FACEATTEND_TEST_OWNER_URL=private['test_owner'],FACEATTEND_TEST_RUNTIME_URL=private['test_runtime'])
        return subprocess.run(args.command,env=env).returncode
    except Exception:
        print('Local enrollment credentials or child command unavailable.',file=sys.stderr);return 1

if __name__=='__main__': sys.exit(main())
