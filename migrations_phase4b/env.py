import os
from alembic import context
from faceattend_db.session import make_engine
from faceattend_enrollment.models import EnrollmentBase

def run():
    if context.is_offline_mode():
        context.configure(dialect_name='postgresql',target_metadata=EnrollmentBase.metadata,
                          version_table_schema='demo',literal_binds=True)
        with context.begin_transaction(): context.run_migrations()
        return
    engine=None
    try:
        engine=make_engine(os.environ.get('FACEATTEND_MIGRATION_DATABASE_URL',''))
        with engine.connect() as connection:
            context.configure(connection=connection,target_metadata=EnrollmentBase.metadata,version_table_schema='demo')
            with context.begin_transaction(): context.run_migrations()
    except Exception:
        raise RuntimeError('Local enrollment migration failed; check owner configuration.') from None
    finally:
        if engine is not None: engine.dispose()

run()
