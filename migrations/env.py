import os
from alembic import context
from faceattend_db.models import Base
from faceattend_db.config import local_url
from faceattend_db.session import make_engine

def run():
    if context.is_offline_mode():
        context.configure(dialect_name='postgresql', target_metadata=Base.metadata, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    try:
        # Separate owner secret: runtime URL is never used for migrations.
        url=local_url(os.environ.get('FACEATTEND_MIGRATION_DATABASE_URL',''))
        engine=make_engine(url)
        try:
            with engine.connect() as connection:
                context.configure(connection=connection,target_metadata=Base.metadata)
                with context.begin_transaction():
                    context.run_migrations()
        finally:
            engine.dispose()
    except Exception:
        raise RuntimeError('Local migration failed; check owner credentials and database readiness.') from None

run()
