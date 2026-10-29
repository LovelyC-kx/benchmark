"""One command for the L3.6 implementation; journals failures without hiding them."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def build_commands(output,run_directory,primary_directory=None,primary_scenes=8192):
    out,run_directory=Path(output).resolve(),Path(run_directory).resolve()
    primary=Path(primary_directory).resolve() if primary_directory else out/'primary'
    commands=[]
    def add(name,script,*args):commands.append((name,[sys.executable,str(ROOT/script),*map(str,args)]))
    add('known_controls','run_temporal_controls.py','--output',out/'known_controls.json')
    if primary_directory is None:add('primary','run_temporal_review.py','--output',primary,'--run-directory',run_directory,'--scenes',primary_scenes)
    add('extensions','run_temporal_extensions.py',primary,'--output',out/'extensions','--run-directory',run_directory)
    add('conditional_mean','run_mean_geometry.py','--output',out/'conditional_mean.json','--run-directory',run_directory)
    add('delay_certificates','certify_delay_readers.py','--primary',primary,'--extensions',out/'extensions','--output',out/'delay_certificates.json','--run-directory',run_directory)
    add('consistency','run_temporal_consistency.py',primary,'--output',out/'consistency','--run-directory',run_directory)
    add('phase_readers','run_phase_readers.py',primary,'--output',out/'phase_readers','--run-directory',run_directory)
    add('event_scale','run_event_scale_scan.py',primary,'--output',out/'event_scale','--run-directory',run_directory)
    add('model_interfaces','run_model_interfaces.py','--output',out/'model_interfaces.json')
    return commands

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--run-directory',required=True)
    p.add_argument('--primary-directory');p.add_argument('--primary-scenes',type=int,default=8192);p.add_argument('--print-plan',action='store_true');a=p.parse_args()
    commands=build_commands(a.output,a.run_directory,a.primary_directory,a.primary_scenes)
    if a.print_plan:print(json.dumps(commands,indent=2));return
    out,run=Path(a.output),Path(a.run_directory)
    if out.exists():raise FileExistsError(out)
    for name in ('best.pt','run.json'):
        if not (run/name).is_file():raise FileNotFoundError(run/name)
    if a.primary_directory:
        lock=json.loads((Path(a.primary_directory)/'freeze_contract.json').read_text())
        if hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()!=lock['checkpoint_sha256']:
            raise ValueError('Borrowed primary checkpoint hash differs')
    out.mkdir(parents=True);journal={'status':'running','commands':commands,'stages':[],
        'borrowed_primary':str(Path(a.primary_directory).resolve()) if a.primary_directory else None,
        'scope':'Analysis pipeline. Does not train SNNs or download real datasets.'}
    def save():
        temp=out/'journal.tmp';temp.write_text(json.dumps(journal,indent=2)+'\n');temp.replace(out/'journal.json')
    save()
    for name,cmd in commands:
        start=time.perf_counter()
        with (out/(name+'.log')).open('w') as stream:code=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
        journal['stages'].append({'name':name,'exit_code':code,'elapsed_seconds':time.perf_counter()-start})
        if code:
            journal['status']='failed';save();raise RuntimeError(f'{name} failed; inspect {out/(name+".log")}')
        save();print('completed',name,flush=True)
    journal['status']='complete';save()

if __name__=='__main__':main()
