#!/usr/bin/env python3
"""Resume the compiled stage-one analysis recipes without retraining completed runs.

Analysis-only normalization/dispatch repairs are backed up. The optional retry
is limited to the two audited CUDA-unavailable, never-started training jobs.
All outcomes in heldout_design remain sealed. Original analysis budgets remain.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,signal,sqlite3,subprocess,sys,time
from collections import Counter
from pathlib import Path

TRAIN_FILES=('engine.py','telemetry.py','utils.py','specs.py','models.py','extended_models.py','data.py','protocols.py','reference/__init__.py','reference/models.py','reference/neurons.py','legacy/tasks.py','legacy/l1_contracts.py')
CUDA_IDS={'00da6b6e70d0878d5e91cb3e','0159632d2ede0897a614c4f7'}
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temp,path)
def ro(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=60)
def source_hash(code,analysis=False):
    base=Path(code)/'snnbench'
    paths=sorted(base.rglob('*.py')) if analysis else [base/name for name in TRAIN_FILES]
    return digest({'scope':'analysis' if analysis else 'training','files':{str(p.relative_to(base)):sha(p) for p in paths if '__pycache__' not in str(p)}})
def patch_analysis(code):
    code=Path(code)/'snnbench'
    changes=[(code/'readouts.py',
        '    mean=xs[0].mean(0);scale=np.maximum(xs[0].std(0),1e-6);x,v,e=[torch.tensor((z-mean)/scale,dtype=torch.float32) for z in xs]',
        '''    mean=xs[0].mean(0);std=xs[0].std(0);constant=std<=1e-8
    scale=np.where(constant,1.,np.maximum(std,1e-6))
    obs['standardization']={'policy':'fit_only_unit_scale_for_near_constant_v1','fit_split':'probe_train',
        'fit_constant_coordinates':np.flatnonzero(constant).tolist(),
        'validation_changed_fit_constant_coordinates':np.flatnonzero(constant & np.any(np.abs(xs[1]-mean)>1e-8,axis=0)).tolist(),
        'evaluation_changed_fit_constant_coordinates':np.flatnonzero(constant & np.any(np.abs(xs[2]-mean)>1e-8,axis=0)).tolist()}
    x,v,e=[torch.tensor((z-mean)/scale,dtype=torch.float32) for z in xs]'''),
        (code/'analysis.py',
        '    torch.manual_seed(seed);mean=fit.mean(0);scale=np.maximum(fit.std(0),1e-6)',
        '    torch.manual_seed(seed);mean=fit.mean(0);std=fit.std(0);scale=np.where(std<=1e-8,1.,np.maximum(std,1e-6))'),
        (code/'analysis.py',
        '        T=I.shape[1];weights=[];contributions=[]',
        "        if cell.mode=='notd':b=np.zeros_like(b)  # NoTD has zero forward carry\n        T=I.shape[1];weights=[];contributions=[]"),
        (code/'campaign.py',
        "    override=replace(spec.analysis,modules=modules,eval_split='development')",
        '''    # compiled_stage1_recipe_dispatch_v1: preserve each compiled recipe.
    settings=dict(job.get('analysis') or {})
    if settings and settings.get('eval_split','development')!='development':
        raise ValueError('Compiled stage-one bridge opens development only')
    for field in ('modules','ridge_alphas'):
        if field in settings:settings[field]=tuple(settings[field])
    from .engine import implementation_hash
    expected=job.get('analysis_implementation_sha256')
    if expected and expected!=implementation_hash('analysis'):
        raise ValueError('Analysis source changed after staging; prepare a new revision queue')
    override=replace(spec.analysis,**settings) if settings else replace(spec.analysis,modules=modules,eval_split='development')''')]
    prepared={}
    for path,before,after in changes:
        original,text=prepared.get(path,(None,None))
        if original is None:original=text=path.read_text()
        if after in text:continue
        if text.count(before)!=1:raise RuntimeError('Unrecognized source; nothing patched: '+str(path))
        updated=text.replace(before,after);compile(updated,str(path),'exec');prepared[path]=(original,updated)
    for path,(old,new) in prepared.items():
        backup=path.with_name(path.name+'.before_stage1_bridge_'+hashlib.sha256(old.encode()).hexdigest()[:12]+'.bak')
        if not backup.exists():backup.write_text(old)
        temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_text(new);os.chmod(temp,path.stat().st_mode & 0o777);os.replace(temp,path)
        print('PATCHED',path,flush=True)
    return [str(x) for x in prepared]
def inspect(stage):
    stage=Path(stage);plan=json.loads((stage/'campaign.json').read_text())
    with ro(stage/'train.sqlite') as con:
        jobs={r[0]:{'state':r[1],'groups':json.loads(r[2] or '[]')} for r in con.execute('SELECT j.id,j.state,d.reasons FROM jobs j JOIN design d ON j.id=d.id')}
    recipes=[json.loads(line) for line in (stage/'analysis_jobs.jsonl').read_text().splitlines() if line.strip()]
    counts=Counter(x['state'] for x in jobs.values());ready=[];sealed=[];blocked=[]
    for recipe in recipes:
        state=jobs[recipe['run_id']]
        if state['groups'] and 'heldout_design' in state['groups'] or recipe.get('gate'):
            sealed.append(recipe['id']);continue
        if state['state']!='done':blocked.append({'id':recipe['id'],'run_id':recipe['run_id'],'training_state':state['state']});continue
        if recipe['analysis']['eval_split']!='development':raise ValueError('Unexpected role')
        ready.append(recipe)
    return plan,jobs,ready,{'training':dict(counts),'recipes':len(recipes),'ready':len(ready),'sealed':len(sealed),'blocked_training':blocked}
def prepare(stage,code):
    stage=Path(stage);plan,jobs,ready,summary=inspect(stage)
    expected=plan['training_implementation_sha256']
    if source_hash(code)!=expected:raise RuntimeError('Training source differs from saved campaign')
    changed=patch_analysis(code)
    if source_hash(code)!=expected:raise RuntimeError('Training fingerprint changed unexpectedly')
    revision=source_hash(code,True);database=stage/('post_stage1_'+revision[:12]+'.sqlite')
    con=sqlite3.connect(database,timeout=60);con.execute('PRAGMA journal_mode=WAL')
    con.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,spec TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',owner TEXT,lease REAL,attempts INTEGER NOT NULL DEFAULT 0,error TEXT)")
    missing=[];inserted=0
    for recipe in ready:
        run=Path(plan['training_output'])/recipe['run_id'];ck=run/recipe['checkpoint']
        if not ck.is_file():missing.append({'run_id':recipe['run_id'],'checkpoint':recipe['checkpoint']});continue
        job={'kind':'analyze','phase':'all_shared','run':str(run),'run_id':recipe['run_id'],
            'checkpoint':recipe['checkpoint'],'checkpoint_sha256':sha(ck),'data':plan['data_output'],
            'analysis':recipe['analysis'],'purpose':recipe['purpose'],'recipe_id':recipe['id'],
            'analysis_implementation_sha256':revision}
        priority='00' if recipe['purpose']=='main_shared_analysis' else '20'
        key=priority+digest(job)[:30];before=con.total_changes
        con.execute('INSERT OR IGNORE INTO jobs(id,spec) VALUES(?,?)',(key,canonical(job)));inserted+=con.total_changes-before
        if inserted and inserted%100==0:print('STAGED',inserted,flush=True)
    con.commit();summary.update(database=str(database.resolve()),analysis_revision=revision,new_jobs=inserted,
        post_counts=dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')),missing_checkpoints=missing,patched=changed,
        training_source_unchanged=True,specialized_sections_complete=False,capture_requests_complete=False)
    con.close();atomic(stage/'post_stage1.latest.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary
def retry_cuda(stage,code,gpus):
    stage=Path(stage);plan=json.loads((stage/'campaign.json').read_text());runs=Path(plan['training_output'])
    con=sqlite3.connect(stage/'train.sqlite',timeout=60)
    if con.execute('SELECT COUNT(*) FROM allocations').fetchone()[0] or con.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('pending','running')").fetchone()[0]:
        con.close();raise RuntimeError('Training is active; CUDA-only retry requires a drained queue')
    targets=[]
    for ident in sorted(CUDA_IDS):
        row=con.execute('SELECT spec,state,error,attempts FROM jobs WHERE id=?',(ident,)).fetchone()
        if row is None:raise RuntimeError('Audited ID absent')
        if row[1]=='done':continue
        if row[1]!='failed' or (runs/ident/'last.pt').exists():raise RuntimeError('Not a never-started audited CUDA failure: '+ident)
        log=runs/'_queue_specs'/(ident+'.log')
        if not log.is_file() or 'CUDA requested but unavailable' not in log.read_text(errors='replace')[-16384:]:
            raise RuntimeError('CUDA-unavailable evidence not found: '+ident)
        targets.append({'run_id':ident,'spec':json.loads(row[0]),'old_error':row[2],'attempts':row[3]})
    if not targets:con.close();return
    audit=stage/'cuda_retry_audit.json'
    if audit.exists():raise RuntimeError('Retry was already requested; inspect its audit instead of repeating')
    backup=stage/'train_before_cuda_retry.sqlite'
    if backup.exists():raise RuntimeError('Retry backup already exists')
    with sqlite3.connect(backup) as dest:con.backup(dest)
    atomic(audit,{'targets':targets,'backup':str(backup),'scope':'Only two startup CUDA failures; scientific specs and completed jobs unchanged'})
    with con:
        for target in targets:
            con.execute("UPDATE jobs SET state='pending',owner=NULL,lease=NULL,error=NULL WHERE id=? AND state='failed'",(target['run_id'],))
    con.close()
    helper=Path(os.environ.get('SNN_PROJECT_ROOT',stage.parents[1]))/'scripts/efficient_campaign.py'
    env=dict(os.environ,SNN_CODE=str(code),PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH',''),SNN_GPU_QUERY_IDS=','.join(gpus))
    command=[sys.executable,str(helper),'run','--root',str(stage),'--gpus',*gpus,'--max-slots','1','--cpu-budget','8','--resume']
    subprocess.run(command,env=env,check=True)
def launch(stage,code,gpus,slots,retry):
    stage=Path(stage)
    with (stage/'stage1_post_supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inventory={}
        for index in gpus:
            uuid=subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).strip()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
            subprocess.run([sys.executable,'-c',"import torch; x=torch.ones(4,device='cuda:0',requires_grad=True); x.sum().backward(); torch.cuda.synchronize(); print('GPU_OK',torch.cuda.get_device_name(0))"],env=env,check=True,timeout=60)
            inventory[index]=uuid
        if retry:retry_cuda(stage,code,gpus)
        receipt=prepare(stage,code);db=receipt['database'];logs=stage/'post_logs';logs.mkdir(exist_ok=True)
        envbase=dict(os.environ,PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        command=[sys.executable,'-u','-m','snnbench','post-worker','--db',db,'--device','cuda:0']
        if receipt['post_counts'].get('pending',0):
            with (logs/'first_recipe.log').open('a') as stream:
                subprocess.run(command+['--one'],env=dict(envbase,CUDA_VISIBLE_DEVICES=inventory[gpus[0]]),stdout=stream,stderr=subprocess.STDOUT,check=True)
            with ro(db) as con:failed=con.execute("SELECT id,error FROM jobs WHERE state='failed'").fetchall()
            if failed:raise RuntimeError('First recipe gate failed; see '+str(logs/'first_recipe.log')+' and '+repr(failed))
        children=[]
        def stop(signum,frame):
            for child in children:
                if child.poll() is None:child.terminate()
            raise SystemExit(128+signum)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        for gpu in gpus:
            for slot in range(slots):
                log=logs/f'gpu{gpu}_slot{slot}.log'
                with log.open('a') as stream:
                    child=subprocess.Popen(command,env=dict(envbase,CUDA_VISIBLE_DEVICES=inventory[gpu]),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                children.append(child);print('WORKER',gpu,slot,child.pid,log,flush=True)
        while any(p.poll() is None for p in children):time.sleep(5)
        with ro(db) as con:counts=dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        atomic(stage/'post_stage1.final.json',{'counts':counts,'database':db,'specialized_sections_complete':False,'capture_requests_complete':False})
        print('POST_FINAL',json.dumps(counts),flush=True)
        if counts.get('failed') or counts.get('pending') or counts.get('running'):raise RuntimeError('Some recipes remain unresolved; preserve logs and inspect failures')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('inspect','prepare','launch','status'))
    p.add_argument('--stage',default=str(Path(os.environ.get('SNN_PROJECT_ROOT','/home/bohan_zhang/research/benchmark_temporal'))/'results/stage1_all_layers'))
    p.add_argument('--code',default=os.environ.get('SNN_CODE','/home/bohan_zhang/research/benchmark_temporal/src/benchmark'))
    p.add_argument('--gpus',nargs='+',default=['1','3']);p.add_argument('--slots',type=int,default=2);p.add_argument('--retry-cuda',action='store_true');a=p.parse_args()
    if a.slots<1 or len(a.gpus)!=len(set(a.gpus)) or any(g not in ('1','2','3') for g in a.gpus):p.error('Use distinct physical GPU 1/2/3 and positive slots; default leaves GPU2 alone')
    if a.command=='inspect':print(json.dumps(inspect(a.stage)[-1],ensure_ascii=False,indent=2))
    elif a.command=='prepare':prepare(a.stage,a.code)
    elif a.command=='launch':launch(a.stage,a.code,a.gpus,a.slots,a.retry_cuda)
    else:
        receipt=json.loads((Path(a.stage)/'post_stage1.latest.json').read_text())
        with ro(receipt['database']) as con:
            print(json.dumps({'database':receipt['database'],'counts':dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')),'failures':con.execute("SELECT id,error FROM jobs WHERE state='failed' LIMIT 10").fetchall()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
