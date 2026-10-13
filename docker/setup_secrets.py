"""Explicit host setup; never generates retained credentials during app startup."""
from pathlib import Path
import argparse,getpass,json,os,secrets,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cryptography.fernet import Fernet
from faceattend_enrollment.auth import password_record
from faceattend_enrollment.security import create_private_directory,private_permissions,protected_path

QA_PASSWORD='Fictional Phase 6 operator password'

def setup(directory,password):
    path=protected_path(directory)
    if path.exists():raise ValueError('Refusing existing secret directory.')
    folder=create_private_directory(path)
    values={name:secrets.token_urlsafe(32).encode('ascii') for name in ('db_admin_password','db_owner_password','db_runtime_password')}
    values['gallery_key']=Fernet.generate_key()
    values['operator_auth']=json.dumps(password_record(password)).encode('utf-8')
    os.umask(0o077)
    for name,value in values.items():
        with os.fdopen(os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as stream:stream.write(value)
        private_permissions(folder/name)
    return folder

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--qa',action='store_true');args=parser.parse_args()
    try:
        directory=Path(args.directory)
        if args.qa:
            allowed=Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase6/private'
            if not directory.name.startswith('qa_') or not directory.resolve().is_relative_to(allowed.resolve()):raise ValueError
            password=QA_PASSWORD
        else:
            password=getpass.getpass('Choose your local operator password (12–256 characters): ')
            if password!=getpass.getpass('Repeat operator password: '):raise ValueError
        setup(directory,password)
        print('Protected Compose secrets created; no database or enrollment was created.')
    except Exception:
        print('Secret setup unavailable; verify a new protected directory and matching password.',file=sys.stderr);sys.exit(1)
