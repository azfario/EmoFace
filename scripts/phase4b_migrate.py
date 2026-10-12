from pathlib import Path
import os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sqlalchemy import text
from faceattend_db.session import make_engine
from faceattend_db.config import local_url
from faceattend_enrollment.migration import upgrade

def main():
    engine=None
    try:
        url=local_url(os.environ.get('FACEATTEND_MIGRATION_DATABASE_URL',''))
        if url.database!='faceattend_research' or url.username!='faceattend_migrator': raise ValueError
        upgrade(url)
        engine=make_engine(url)
        with engine.begin() as c:
            c.execute(text('GRANT USAGE ON SCHEMA demo TO faceattend_runtime'))
            c.execute(text('GRANT SELECT, INSERT, UPDATE, DELETE ON demo.demo_enrolled_identities TO faceattend_runtime'))
        print('Enrollment metadata migration ready; original aggregate schema preserved.')
        return 0
    except Exception:
        print('Local enrollment migration unavailable.',file=sys.stderr);return 1
    finally:
        if engine is not None: engine.dispose()

if __name__=='__main__': sys.exit(main())
