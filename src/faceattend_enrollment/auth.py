"""One local operator session, bounded rates and constant-time credentials."""
import base64,hashlib,hmac,json,secrets,threading,time
from .security import read_secret
from .errors import EnrollmentError

COOKIE='faceattend_operator'

def password_record(password):
    if not isinstance(password,str) or not 12<=len(password)<=256:
        raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
    salt=secrets.token_bytes(16)
    digest=hashlib.scrypt(password.encode(),salt=salt,n=2**14,r=8,p=1,dklen=32)
    return {'version':1,'salt':base64.b64encode(salt).decode(),'digest':base64.b64encode(digest).decode()}

class OperatorAuth:
    login_required=True
    def __init__(self,record,clock=time.monotonic):
        try:
            if set(record)!={'version','salt','digest'} or record['version']!=1: raise ValueError
            self.salt=base64.b64decode(record['salt'],validate=True)
            self.digest=base64.b64decode(record['digest'],validate=True)
            if len(self.salt)!=16 or len(self.digest)!=32: raise ValueError
        except Exception:
            raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None
        self.clock=clock;self.lock=threading.Lock();self.session=None
        self.rates={'login':[],'mutation':[],'live':[]}

    @classmethod
    def from_file(cls,path):
        try: return cls(json.loads(read_secret(path)))
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def _rate(self,kind):
        now=self.clock(); events=self.rates[kind]
        events[:]=[t for t in events if now-t<60]
        limit={'login':5,'mutation':20,'live':1200}[kind]
        if len(events)>=limit or kind=='live' and sum(now-t<1 for t in events)>=30: raise EnrollmentError('RATE_LIMITED')
        events.append(now)

    def login(self,password):
        with self.lock: self._rate('login')
        if not isinstance(password,str) or not 1<=len(password)<=256:
            raise EnrollmentError('OPERATOR_AUTH_REQUIRED')
        derived=hashlib.scrypt(password.encode(),salt=self.salt,n=2**14,r=8,p=1,dklen=32)
        if not hmac.compare_digest(derived,self.digest): raise EnrollmentError('OPERATOR_AUTH_REQUIRED')
        token=secrets.token_urlsafe(32); csrf=secrets.token_urlsafe(32)
        with self.lock:
            # A new successful login replaces any previous local session.
            self.session=(hashlib.sha256(token.encode()).digest(),csrf,self.clock()+900)
        return token,csrf

    def authorize(self,request,mutation=False,*,live=False):
        try:
            if sum(k==b'cookie' for k,v in request.scope.get('headers',[]))>1: raise ValueError
            token=request.cookies.get(COOKIE,'')
            if not 20<=len(token)<=128: raise ValueError
            with self.lock:
                session=self.session
                if session is None or self.clock()>=session[2] or not hmac.compare_digest(hashlib.sha256(token.encode()).digest(),session[0]):
                    raise ValueError
                if mutation:
                    if not hmac.compare_digest(request.headers.get('x-faceattend-csrf',''),session[1]):
                        raise EnrollmentError('CSRF_REQUIRED')
                    self._rate('live' if live else 'mutation')
        except EnrollmentError: raise
        except Exception: raise EnrollmentError('OPERATOR_AUTH_REQUIRED') from None

    def logout(self,request=None):
        with self.lock: self.session=None

class LocalAccess(OperatorAuth):
    """Password-free local access with transient CSRF sessions and bounded rates.

    This is request protection, not identity authentication. Every local user
    can open a session. Independent tabs do not invalidate each other's tokens.
    """
    login_required=False

    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.lock=threading.Lock();self.sessions={}
        self.rates={'login':[],'mutation':[],'live':[]}

    def open_session(self,request=None):
        with self.lock:
            self._rate('login')
            self.sessions={key:value for key,value in self.sessions.items() if self.clock()<value[1]}
            if request is not None:
                token=request.cookies.get(COOKIE,'')
                key=hashlib.sha256(token.encode()).digest()
                existing=self.sessions.get(key)
                if existing is not None:
                    # Tabs share cookies. Reuse the matching CSRF token rather
                    # than making one tab invalidate another during bootstrap.
                    self.sessions[key]=(existing[0],self.clock()+900)
                    return token,existing[0]
            if len(self.sessions)>=32:raise EnrollmentError('RATE_LIMITED')
            token=secrets.token_urlsafe(32);csrf=secrets.token_urlsafe(32)
            self.sessions[hashlib.sha256(token.encode()).digest()]=(csrf,self.clock()+900)
        return token,csrf

    def authorize(self,request,mutation=False,*,live=False):
        try:
            if sum(k==b'cookie' for k,v in request.scope.get('headers',[]))>1:raise ValueError
            token=request.cookies.get(COOKIE,'')
            if not 20<=len(token)<=128:raise ValueError
            with self.lock:
                session=self.sessions.get(hashlib.sha256(token.encode()).digest())
                if session is None or self.clock()>=session[1]:raise ValueError
                if mutation:
                    if not hmac.compare_digest(request.headers.get('x-faceattend-csrf',''),session[0]):
                        raise EnrollmentError('CSRF_REQUIRED')
                    self._rate('live' if live else 'mutation')
        except EnrollmentError:raise
        except Exception:raise EnrollmentError('OPERATOR_AUTH_REQUIRED') from None

    def logout(self,request=None):
        with self.lock:
            if request is None:self.sessions.clear()
            else:self.sessions.pop(hashlib.sha256(request.cookies.get(COOKIE,'').encode()).digest(),None)
