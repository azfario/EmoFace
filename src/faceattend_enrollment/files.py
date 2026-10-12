"""Scoped ciphertext only, reusable core store and crash-safe process lock."""
from contextlib import contextmanager
from pathlib import Path
import json,os,re
from uuid import UUID
from cryptography.fernet import Fernet
from face_verification.template_store import TemplateStore
from face_verification.enrollment import delete_enrollment
from face_verification.schemas import Status
from .repository import valid_id,valid_locator
from .security import protected_path,private_permissions
from .errors import EnrollmentError

class DeletionStore:
    # Reuse the core validated locator/path/delete interfaces without decrypting.
    _path=TemplateStore._path
    delete=TemplateStore.delete
    def __init__(self,directory): self.directory=directory

class GalleryFiles:
    def __init__(self,root,key=None):
        self.root=private_permissions(root)
        if not self.root.is_dir(): raise EnrollmentError('ENROLLMENT_STORAGE_ERROR')
        self.templates=self.child('templates');self.staging=self.child('staging')
        self.templates.mkdir(exist_ok=True);self.staging.mkdir(exist_ok=True)
        try: self.cipher=Fernet(key) if key else None
        except Exception: raise EnrollmentError('ENROLLMENT_KEY_UNAVAILABLE') from None
        self.key=key

    def child(self,*parts):
        p=protected_path(self.root.joinpath(*parts))
        if not p.is_relative_to(self.root): raise EnrollmentError('ENROLLMENT_STORAGE_ERROR')
        return p

    @staticmethod
    def user(locator): return 'volunteer_'+valid_locator(locator)

    def store(self,directory=None):
        if self.key is None: raise EnrollmentError('ENROLLMENT_KEY_UNAVAILABLE')
        return TemplateStore(directory or self.templates,key=self.key,retention_days=30)

    def final(self,locator):
        path=DeletionStore(self.templates)._path(self.user(locator))
        return self.child('templates',path.name)

    @contextmanager
    def locked(self):
        path=self.child('mutation.lock')
        stream=None
        acquired=False
        try:
            stream=path.open('a+b')
            if path.stat().st_size==0:
                stream.write(b'0');stream.flush()
            stream.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                acquired=True
            except OSError: raise EnrollmentError('ENROLLMENT_BUSY') from None
            yield
        except EnrollmentError: raise
        except Exception:
            if acquired: raise  # Preserve core QC/model and service exceptions.
            raise EnrollmentError('ENROLLMENT_STORAGE_ERROR') from None
        finally:
            if stream is not None:
                try:
                    if acquired:
                        stream.seek(0)
                        if os.name=='nt':
                            import msvcrt
                            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                        else:
                            import fcntl
                            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
                finally: stream.close()

    def stage(self,identity_id,locator,intent):
        if self.cipher is None: raise EnrollmentError('ENROLLMENT_KEY_UNAVAILABLE')
        if len(list(self.staging.iterdir()))>=100: raise EnrollmentError('GALLERY_FULL')
        identity_id=valid_id(identity_id)
        folder=self.child('staging',identity_id.hex)
        try:
            folder.mkdir()
            encrypted=self.cipher.encrypt(json.dumps(intent,allow_nan=False).encode())
            with (folder/'intent.fernet').open('xb') as f:
                f.write(encrypted);f.flush();os.fsync(f.fileno())
            store_dir=self.child('staging',identity_id.hex,'store');store_dir.mkdir()
            return self.store(store_dir)
        except Exception:
            # No DB reservation exists yet; clear only this known staged write.
            self.clean_stage(identity_id)
            raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING') from None

    def promote(self,identity_id,locator):
        folder=self.child('staging',valid_id(identity_id).hex,'store')
        source=protected_path(DeletionStore(folder)._path(self.user(locator)))
        target=self.final(locator)
        try:
            if target.exists():
                if source.exists(): raise ValueError
                return  # Recovery of a rename completed before a crash.
            source.replace(target)
        except Exception: raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING') from None

    def delete(self,locator):
        # Lost keys cannot block authorized deletion of scoped ciphertext.
        self.final(locator)
        result=delete_enrollment(self.user(locator),DeletionStore(self.templates))
        if result.status not in {Status.OK,Status.UNKNOWN_ID}:
            raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING')

    def clean_stage(self,identity_id):
        folder=self.child('staging',valid_id(identity_id).hex)
        if not folder.exists(): return
        try:
            allowed={'intent.fernet','store'}
            if any(p.name not in allowed for p in folder.iterdir()): raise ValueError
            store=self.child('staging',identity_id.hex,'store')
            if store.exists():
                for p in store.iterdir():
                    if not re.fullmatch(r'[0-9a-f]{64}\.fernet',p.name) or not p.is_file(): raise ValueError
                    self.child('staging',identity_id.hex,'store',p.name).unlink()
                store.rmdir()
            intent=self.child('staging',identity_id.hex,'intent.fernet')
            intent.unlink(missing_ok=True);folder.rmdir()
        except Exception: raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING') from None

    def intents(self,identity_id=None):
        folders=[self.child('staging',valid_id(identity_id).hex)] if identity_id else list(self.staging.iterdir())
        folders=[p for p in folders if p.exists()]
        if not folders: return []
        if self.cipher is None: raise EnrollmentError('ENROLLMENT_KEY_UNAVAILABLE')
        result=[]
        try:
            if len(folders)>100: raise ValueError
            for folder in folders:
                identity_id=valid_id(UUID(hex=folder.name))
                path=self.child('staging',identity_id.hex,'intent.fernet')
                if path.stat().st_size>16384: raise ValueError
                intent=json.loads(self.cipher.decrypt(path.read_bytes()))
                if set(intent)!={'identity_id','locator','old_locator','created_at','model_id'} or intent['identity_id']!=str(identity_id): raise ValueError
                valid_locator(intent['locator'])
                if intent['old_locator'] is not None: valid_locator(intent['old_locator'])
                result.append(intent)
            return result
        except Exception: raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING') from None
