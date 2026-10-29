"""Sequential auditable entry point for the L4.1 development stage."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--run-directory',default=str(ROOT/'evidence/source_smoke_run'));p.add_argument('--evaluation-scenes',type=int,default=8192);a=p.parse_args()
    out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);stages=[('controls','run_access_controls.py',['--output',out/'controls.json']),
        ('access','run_access_review.py',['--output',out/'access','--run-directory',Path(a.run_directory).resolve(),'--evaluation-scenes',a.evaluation_scenes]),
        ('interfaces','run_interface_grid.py',['--output',out/'interfaces.json'])]
    journal={'status':'running','stages':[],'scope':'L4.1 controlled analysis; no SNN training or dataset download'}
    for name,script,args in stages:
        start=time.perf_counter();cmd=[sys.executable,str(ROOT/script),*map(str,args)]
        with (out/(name+'.log')).open('w') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        journal['stages'].append({'name':name,'exit_code':code,'elapsed_seconds':time.perf_counter()-start})
        journal['status']='failed' if code else 'running';(out/'journal.json').write_text(json.dumps(journal,indent=2)+'\n')
        if code:raise RuntimeError(f'{name} failed; see log')
        print('completed',name,flush=True)
    journal['status']='complete';(out/'journal.json').write_text(json.dumps(journal,indent=2)+'\n')

if __name__=='__main__':main()
