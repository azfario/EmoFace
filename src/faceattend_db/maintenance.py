"""Explicit local operator commands; retention uses owner credentials only."""
import argparse, json, os, sys
from datetime import date, datetime, timedelta, timezone
from .config import local_url, DBSettings
from .instrumentation import Telemetry
from .repository import Repository
from .session import make_engine, sessions

def main(mode):
    parser=argparse.ArgumentParser(description='Local aggregate database operator tool.')
    if mode=='summary':
        parser.add_argument('--start',type=date.fromisoformat,required=True)
        parser.add_argument('--end',type=date.fromisoformat,required=True)
    if mode=='retention':
        parser.add_argument('--apply',action='store_true',help='Explicitly delete aggregate dates older than the retention cutoff.')
    args=parser.parse_args()
    engine=None
    try:
        if mode=='health':
            service=Telemetry.from_env()
            try:
                service.start()
                status=service.status
                print(json.dumps({'status':status}))
                return 1 if status=='unavailable' else 0
            finally:
                service.close()
        url=local_url(os.environ.get('FACEATTEND_MIGRATION_DATABASE_URL' if mode=='retention' else 'FACEATTEND_DATABASE_URL',''))
        engine=make_engine(url)
        repository=Repository(sessions(engine))
        if mode=='summary':
            print(json.dumps(repository.get_aggregate_summary(args.start,args.end),indent=2))
        else:
            days=DBSettings.from_env().retention_days
            cutoff=datetime.now(timezone.utc).date()-timedelta(days=days)
            removed=repository.purge_old_aggregates(cutoff) if args.apply else None
            print(json.dumps({'cutoff_utc':cutoff.isoformat(),'applied':args.apply,'deleted_aggregate_rows':removed}))
        return 0
    except Exception:
        print('Local aggregate database command failed.',file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
