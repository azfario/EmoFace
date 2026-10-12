"""Explicit private functionality inputs; owner labels come only from filenames."""
from pathlib import Path
import argparse,json,os,re,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from faceattend_enrollment.security import protected_path,create_private_directory

def clean():
    base=protected_path(Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase2/private')
    removed=0
    for phase in ('phase4b','phase4c'):
        target=protected_path(base/(phase+'_test_inputs'))
        if target.parent!=base:raise ValueError
        if not target.exists():continue
        children=list(target.iterdir())
        pattern=r'group_00[123]_'+('image' if phase=='phase4b' else '(enrollment|probes)')+r'_0[1-5]\.(jpg|jpeg|png)'
        if any(p.is_symlink() or not p.is_file() or (p.name!='manifest.json' and not re.fullmatch(pattern,p.name)) for p in children):raise ValueError
        manifest=json.loads((target/'manifest.json').read_text())
        if len(manifest['groups'])!=3:raise ValueError
        fields=('images',) if phase=='phase4b' else ('enrollment','probes')
        expected={name for group in manifest['groups'] for field in fields for name in group[field]}|{'manifest.json'}
        if expected!={p.name for p in children}:raise ValueError
        for p in children:
            safe=protected_path(p)
            if safe.parent!=target:raise ValueError
        for p in children:p.unlink();removed+=p.name!='manifest.json'
        target.rmdir()
    return removed

def prepare(source):
    def groups(folder):
        values={}
        for path in folder.iterdir():
            if not path.is_file():continue
            if path.is_symlink() or path.suffix.casefold() not in {'.jpg','.jpeg','.png'} or not 0<path.stat().st_size<=8*1024*1024:raise ValueError
            label=' '.join(re.findall('[A-Za-z]+',path.stem.casefold()))
            if not label:raise ValueError
            values.setdefault(label,[]).append(path)
        return values
    enrollment=groups(source/'Enrollment');probes=groups(source/'Test')
    if len(enrollment)!=3 or set(enrollment)!=set(probes) or any(not 3<=len(v)<=5 for v in enrollment.values()) or any(not 1<=len(v)<=5 for v in probes.values()):raise ValueError
    base=protected_path(Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase2/private')
    targets={phase:protected_path(base/(phase+'_test_inputs')) for phase in ('phase4b','phase4c')}
    if any(p.exists() for p in targets.values()):raise ValueError
    for phase,target in targets.items():
        create_private_directory(target)
        manifest={'adult_consent':True,'persistent_enrollment_consent':True,'groups':[],
            'scope':'isolated functionality only; not calibration','selected_group_indices':[0,1,2]}
        if phase=='phase4c':manifest['separate_probe_sessions_user_confirmed']=True
        for n,label in enumerate(sorted(enrollment),1):
            entry={'chosen_test_pseudonym':f'Fictional Adult {n:03}'}
            for field,paths in ([('images',enrollment[label])] if phase=='phase4b' else [('enrollment',enrollment[label]),('probes',probes[label])]):
                entry[field]=[]
                for index,path in enumerate(sorted(paths),1):
                    kind='image' if phase=='phase4b' else field
                    name=f'group_{n:03}_{kind}_{index:02}{path.suffix.casefold()}'
                    # Read no more than the approved single-image bound.
                    with path.open('rb') as stream:content=stream.read(8*1024*1024+1)
                    if not 0<len(content)<=8*1024*1024:raise ValueError
                    with (target/name).open('xb') as stream:stream.write(content)
                    entry[field].append(name)
            manifest['groups'].append(entry)
        (target/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return sum(len(v) for v in enrollment.values())*2+sum(len(v) for v in probes.values())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--confirm-adult-consent',action='store_true')
    parser.add_argument('--confirm-persistent-enrollment',action='store_true')
    parser.add_argument('--confirm-separate-probe-sessions',action='store_true')
    parser.add_argument('--cleanup',action='store_true',help='Remove only known disposable input copies after tests.')
    args=parser.parse_args()
    try:
        if args.cleanup:
            print(json.dumps({'disposable_input_copies_removed':clean()}));return 0
        if not args.source_root or not all((args.confirm_adult_consent,args.confirm_persistent_enrollment,args.confirm_separate_probe_sessions)):raise ValueError
        print(json.dumps({'private_input_copies_prepared':prepare(args.source_root),'source_names_omitted':True}));return 0
    except Exception:
        print('Input preparation/cleanup stopped; private paths and owner labels withheld. Review scoped files locally.',file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
