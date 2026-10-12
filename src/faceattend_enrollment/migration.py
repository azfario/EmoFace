"""Explicit owner-only bootstrap; no application startup DDL."""
from pathlib import Path
import os
from sqlalchemy import inspect,text
from alembic import command
from alembic.config import Config
from faceattend_db.session import make_engine

ROOT=Path(__file__).resolve().parents[2]

def upgrade(owner_url):
    engine=make_engine(owner_url)
    previous=os.environ.get('FACEATTEND_MIGRATION_DATABASE_URL')
    os.environ['FACEATTEND_MIGRATION_DATABASE_URL']=str(engine.url.render_as_string(hide_password=False))
    configuration=Config(str(ROOT/'alembic-phase4b.ini'))
    try:
        with engine.begin() as c:
            if c.execute(text('SELECT version_num FROM public.alembic_version')).scalar_one()!='0001_daily_stats':
                raise ValueError
            c.execute(text('CREATE SCHEMA IF NOT EXISTS demo'))
            c.execute(text('REVOKE ALL ON SCHEMA demo FROM PUBLIC'))
        if 'alembic_version' not in inspect(engine).get_table_names(schema='demo'):
            # Adopt the verified existing revision only in the NEW version table.
            command.stamp(configuration,'0001_daily_stats')
        command.upgrade(configuration,'head')
    except Exception:
        raise RuntimeError('Enrollment migration failed; original aggregate history is unchanged.') from None
    finally:
        engine.dispose()
        if previous is None: os.environ.pop('FACEATTEND_MIGRATION_DATABASE_URL',None)
        else: os.environ['FACEATTEND_MIGRATION_DATABASE_URL']=previous
