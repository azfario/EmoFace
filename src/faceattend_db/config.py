from dataclasses import dataclass, field
import os
from sqlalchemy.engine import make_url, URL

SAFE_CONFIG_ERROR = 'Local database configuration is invalid.'

def local_url(value):
    try:
        url = make_url(value)
        if (url.drivername != 'postgresql+psycopg' or url.host != '127.0.0.1'
                or not url.username or not url.password or not url.database
                or not 1024 <= (url.port or 5432) <= 65535 or url.query):
            raise ValueError
        return url
    except Exception:
        raise ValueError(SAFE_CONFIG_ERROR) from None

@dataclass(frozen=True)
class DBSettings:
    enabled: bool = False
    url: URL | None = field(default=None, repr=False)
    retention_days: int = 30

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.retention_days) is not int or not 1 <= self.retention_days <= 365:
            raise ValueError(SAFE_CONFIG_ERROR)
        if self.enabled:
            object.__setattr__(self, 'url', local_url(self.url))

    @classmethod
    def from_env(cls):
        try:
            flag = os.environ.get('FACEATTEND_DB_TELEMETRY_ENABLED', 'false')
            if flag not in {'true', 'false'}:
                raise ValueError
            # Disabled configuration ignores the URL, without parsing/connecting.
            return cls(enabled=flag == 'true',
                       url=os.environ.get('FACEATTEND_DATABASE_URL') if flag == 'true' else None,
                       retention_days=int(os.environ.get('FACEATTEND_DB_RETENTION_DAYS', '30')))
        except Exception:
            raise ValueError(SAFE_CONFIG_ERROR) from None
