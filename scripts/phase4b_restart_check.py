"""Fresh-process persistence probe; reads/decrypts selected metadata only."""
from pathlib import Path
import os,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from faceattend_enrollment.dependencies import EnrollmentDependencies

def main():
    deps=EnrollmentDependencies.from_env()
    try:
        deps.start()
        result=deps.require_service().inspect(os.environ['FACEATTEND_RESTART_IDENTITY'],os.environ['FACEATTEND_RESTART_MODEL'])
        if result['status']!='active' or result['verified'] is not False:raise ValueError
        print(json.dumps({'restart_selected_reference':'passed','vectors_returned':False}));return 0
    except Exception:
        print('Selected enrollment persistence check unavailable.',file=sys.stderr);return 1
    finally:deps.close()

if __name__=='__main__':sys.exit(main())
