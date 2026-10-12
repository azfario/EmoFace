"""Explicit operator secrets, metadata maintenance and local HTTP client."""
from pathlib import Path
import argparse,getpass,json,os,sys
from cryptography.fernet import Fernet
import httpx
from .auth import password_record
from .security import protected_path,create_private_directory
from .config import EnrollmentSettings
from .dependencies import EnrollmentDependencies
from .errors import EnrollmentError

def setup_secrets(gallery_dir,key_file,auth_file,password):
    gallery=protected_path(gallery_dir);key=protected_path(key_file);auth=protected_path(auth_file)
    if key==auth or key.is_relative_to(gallery) or auth.is_relative_to(gallery):
        raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
    if key.exists() or auth.exists(): raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
    hashed=password_record(password)
    create_private_directory(gallery)
    create_private_directory(key.parent);create_private_directory(auth.parent)
    written=[]
    try:
        for path,value in [(key,Fernet.generate_key()),(auth,json.dumps(hashed).encode())]:
            with path.open('xb') as stream:
                stream.write(value);stream.flush();os.fsync(stream.fileno())
            written.append(path)
            if os.name!='nt': path.chmod(0o600)
    except Exception:
        for path in written: path.unlink(missing_ok=True)
        raise EnrollmentError('ENROLLMENT_STORAGE_ERROR') from None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    setup=commands.add_parser('setup')
    setup.add_argument('--gallery-dir',required=True);setup.add_argument('--key-file',required=True);setup.add_argument('--auth-file',required=True)
    for command in ('list','enroll','reenroll','inspect','revoke','delete'):
        p=commands.add_parser(command);p.add_argument('--port',type=int,default=8000)
        p.add_argument('--private-output',action='store_true',help='Print authorized personal registration metadata; do not redirect/share.')
        if command in ('enroll','reenroll'):
            p.add_argument('--images',nargs='+',required=True)
            p.add_argument('--confirm-adult-consent',action='store_true')
            p.add_argument('--confirm-persistent-enrollment',action='store_true')
    for command in ('retention','recover'):
        p=commands.add_parser(command);p.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    deps=None
    try:
        if args.command=='setup':
            password=getpass.getpass('Choose a local operator password (at least 12 characters): ')
            if password!=getpass.getpass('Repeat operator password: '): raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
            setup_secrets(args.gallery_dir,args.key_file,args.auth_file,password)
            print(json.dumps({'status':'operator_secrets_created','enrollments_created':0}));return 0
        if args.command in ('retention','recover'):
            deps=EnrollmentDependencies(EnrollmentSettings.from_env());deps.start()
            auth=deps.require_auth()
            token,csrf=auth.login(getpass.getpass('Operator password: '))
            service=deps.require_service()
            if not args.apply:
                print(json.dumps({'status':'preview_only','expired_enrollments':len(service.repository.expired(service.clock()))}));return 0
            result=service.cleanup_expired() if args.command=='retention' else service.recover()
            print(json.dumps(result));return 0
        if not 1024<=args.port<=65535: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
        body=None;loaded=[]
        if args.command in ('enroll','reenroll'):
            if not args.confirm_adult_consent or not args.confirm_persistent_enrollment or not 3<=len(args.images)<=5:
                raise EnrollmentError('ENROLLMENT_CONSENT_REQUIRED')
            name=input('Chosen pseudonym/display name (private): ')
            for value in args.images:
                path=protected_path(value)
                if not path.is_file() or not 0<path.stat().st_size<=8*1024*1024: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
                data=path.read_bytes()
                mime='image/png' if data[:8]==b'\x89PNG\r\n\x1a\n' else 'image/jpeg'
                loaded.append(('enrollment_images',('operator-input',data,mime)))
            body={'display_name':name,'adult_consent':'true','research_opt_in':'true','enrollment_consent':'true','retention_days':'30'}
        identity=input('Registration UUID (private): ') if args.command in ('reenroll','inspect','revoke','delete') else None
        if identity:
            from .repository import valid_id
            identity=str(valid_id(identity))
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=120,trust_env=False) as client:
            result=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':getpass.getpass('Operator password: ')})
            if result.status_code!=200: raise EnrollmentError('OPERATOR_AUTH_REQUIRED')
            headers={'x-faceattend-csrf':result.json()['csrf_token']}
            try:
                base='/api/v1/demo/enrollments'
                if args.command=='list': result=client.get(base)
                elif args.command in ('enroll','reenroll'):
                    target=base if args.command=='enroll' else base+'/'+identity+'/reenroll'
                    result=client.post(target,headers=headers,data=body,files=loaded)
                elif args.command=='delete': result=client.delete(base+'/'+identity,headers=headers)
                else:
                    suffix='inspect-reference' if args.command=='inspect' else 'revoke'
                    result=client.post(base+'/'+identity+'/'+suffix,headers=headers)
                if result.status_code>=400:
                    print(json.dumps({'status':result.json().get('status','REQUEST_FAILED')}));return 1
                value=result.json()
                print(json.dumps(value if args.private_output else {'status':value.get('status','completed'),
                    'enrollment_count':len(value.get('enrollments',[])) if args.command=='list' else None}))
                return 0
            finally:
                loaded.clear()
                if body is not None: body.clear()
                client.delete('/api/v1/demo/session',headers=headers)
    except Exception:
        print('Local enrollment command failed; no secret or personal input was logged.',file=sys.stderr);return 1
    finally:
        if deps is not None: deps.close()
