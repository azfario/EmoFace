"""Reject synced/network/reparse paths and insecure secrets; no path logging."""
import os, stat, subprocess
from pathlib import Path
from .errors import EnrollmentError

ROOT=Path(__file__).resolve().parents[2]

def protected_path(value):
    try:
        p=Path(value)
        if not p.is_absolute() or str(p).startswith('\\\\'):
            raise ValueError
        resolved=p.resolve()
        if resolved.is_relative_to(ROOT) or any('onedrive' in part.casefold() for part in resolved.parts):
            raise ValueError
        for ancestor in (p,*p.parents):
            if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor.stat(),'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0)):
                raise ValueError
        return resolved
    except Exception:
        raise EnrollmentError('ENROLLMENT_STORAGE_ERROR') from None

def private_permissions(path):
    path=protected_path(path)
    try:
        if os.name!='nt':
            if path.stat().st_mode & 0o077: raise ValueError
        else:
            literal=str(path).replace("'","''")
            script=("$u=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
                "$a=(Get-Acl -LiteralPath '"+literal+"').Access; "
                "$bad=@($a | Where-Object {$_.AccessControlType -eq 'Allow' -and "
                "$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -notin "
                "@($u,'S-1-5-18','S-1-5-32-544')}); if($bad.Count){exit 1}")
            result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',script],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode: raise ValueError
    except Exception:
        raise EnrollmentError('ENROLLMENT_STORAGE_ERROR') from None
    return path

def read_secret(path,maximum=4096):
    try:
        p=private_permissions(path)
        if not p.is_file() or p.stat().st_size>maximum: raise ValueError
        return p.read_bytes()
    except Exception:
        raise EnrollmentError('ENROLLMENT_KEY_UNAVAILABLE') from None

def create_private_directory(value):
    path=protected_path(value)
    if path.exists(): return private_permissions(path)
    try:
        path.mkdir(parents=True)
        if os.name!='nt': path.chmod(0o700)
        else:
            literal=str(path).replace("'","''")
            script="$u=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; & icacls '"+literal+"' /inheritance:r /grant:r ('*'+$u+':(OI)(CI)F') | Out-Null; exit $LASTEXITCODE"
            result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',script],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode: raise ValueError
        return private_permissions(path)
    except Exception: raise EnrollmentError('ENROLLMENT_STORAGE_ERROR') from None
