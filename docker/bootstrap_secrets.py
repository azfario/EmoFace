"""One-shot container setup on a host-ACL-protected bind mount.

The Windows launcher validates the host ACL before mounting /setup. Passwords
arrive on stdin, never argv/environment, and no secret values are printed.
"""
import argparse,json,os,re,stat,sys
from pathlib import Path
from cryptography.fernet import Fernet
from faceattend_enrollment.auth import OperatorAuth,password_record

DIRECTORY=Path('/setup')
NAMES=('db_admin_password','db_owner_password','db_runtime_password','gallery_key','operator_auth')

def validate():
    if DIRECTORY.is_symlink() or not DIRECTORY.is_dir():raise ValueError
    if {p.name for p in DIRECTORY.iterdir()}!=set(NAMES):raise ValueError
    values={}
    for name in NAMES:
        path=DIRECTORY/name
        mode=path.lstat()
        if not stat.S_ISREG(mode.st_mode) or not 1<=mode.st_size<=4096:raise ValueError
        values[name]=path.read_bytes().strip()
    for name in NAMES[:3]:
        if not re.fullmatch(rb'[A-Za-z0-9_-]{20,128}',values[name]):raise ValueError
    Fernet(values['gallery_key'])
    OperatorAuth(json.loads(values['operator_auth']))
    values.clear()

def create(stream):
    if DIRECTORY.is_symlink() or not DIRECTORY.is_dir() or any(DIRECTORY.iterdir()):raise ValueError
    raw=stream.read(8193)
    if len(raw)>8192:raise ValueError
    request=json.loads(raw)
    if not isinstance(request,dict) or set(request) not in (set(),{'password'}):raise ValueError
    import secrets
    # Keep the legacy five-file layout for retained installations. In the
    # default login-free mode this random password record is dormant.
    record=password_record(request.get('password',secrets.token_urlsafe(32)))
    request.clear();raw=b''
    values={name:secrets.token_urlsafe(32).encode('ascii') for name in NAMES[:3]}
    values['gallery_key']=Fernet.generate_key()
    values['operator_auth']=json.dumps(record).encode('utf-8')
    record.clear()
    os.umask(0o077)
    try:
        for name in NAMES:
            # Windows host confidentiality is enforced by the launcher's NTFS
            # ACL. Explicit Compose secret mounts need to be readable by both
            # API UID 10001 and PostgreSQL UID 999, as native Windows files were.
            descriptor=os.open(DIRECTORY/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o644)
            with os.fdopen(descriptor,'wb') as output:output.write(values[name])
            os.chmod(DIRECTORY/name,0o644)
        validate()
    finally:values.clear()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('create','validate'));args=parser.parse_args()
    try:
        if args.action=='create':create(sys.stdin.buffer)
        else:validate()
        print('Protected secret setup ready; values withheld.')
    except Exception:
        print('Secret setup failed; existing files were not overwritten. Check the protected directory.',file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':sys.exit(main())
