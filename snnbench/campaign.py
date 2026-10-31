"""Broad discovery first; all-checkpoint capture and analysis use separate queues."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
from .utils import atomic_json,file_hash,digest
from .queue import connect
from .specs import RunSpec,AnalysisSpec


def hardware_inventory(storage_root=None):
    import torch,shutil
    cards=[]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i);free,total=torch.cuda.mem_get_info(i);cards.append({'logical_index':i,'name':p.name,'total_memory_bytes':p.total_memory,'currently_free_bytes':free,'total_memory_GiB':p.total_memory/2**30,
                'compute_capability':[p.major,p.minor],'initial_process_budget_bytes':int(.8*p.total_memory)})
    storage=Path(storage_root or Path.cwd()).resolve();disk=shutil.disk_usage(storage)
    return {'host':socket.gethostname(),'python':sys.version,'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'visible_gpus':cards,
            'storage_root':str(storage),'disk_free_bytes':disk.free,'declared_gpu_memory_GB':84,'budget_rule':'Measure actual bytes; begin with 80% as a resource-planning budget, not a changed training batch or an enforced allocator limit'}


def campaign_plan(root,archives=(),seeds=(11,22,33,44,55),profiles=('full','paired','learning'),materialize=False,data_root=None):
    from .catalogue import task_bank,model_bank,compile_manifest
    from .coverage import plan_inventory
    if not profiles or len(set(profiles))!=len(profiles) or not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):raise ValueError('Distinct profiles and nonnegative seeds required')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);source=Path(__file__).resolve().parents[1];commands=[];counts={}
    data_output=Path(data_root).resolve() if data_root is not None else (root/'data').resolve()
    from .engine import implementation_hash
    if (root/'campaign.json').exists():
        previous=json.loads((root/'campaign.json').read_text())
        if previous['profiles']!=list(profiles) or previous['seeds']!=list(seeds) or previous['archives']!=[str(p) for p in archives] or previous['training_implementation_sha256']!=implementation_hash() or Path(previous['data_output']).resolve()!=data_output:raise ValueError('Campaign design/source/data location changed; use a new root to preserve prior evidence')
    for profile in profiles:
        if profile not in ('smoke','screen','full','expanded','paired','learning'):raise ValueError('Unknown campaign profile')
        manifest=root/'manifests'/(profile+'.jsonl');n=sum(len(model_bank(t,profile,'expanded')) for t in task_bank(profile))*len(seeds)
        counts[profile]={'synthetic_runs_before_cross_profile_dedup':n,'archives_requested':len(archives)}
        command=[sys.executable,'-m','snnbench','compile','--profile',profile,'--model-roster','expanded','--seeds',*[str(s) for s in seeds],'--out',str(manifest)]
        if archives:command+=['--archives',*[str(p) for p in archives]]
        commands.append(command)
        if materialize:
            receipt=compile_manifest(manifest,profile,seeds,archives,model_roster='expanded');counts[profile]['compiled_receipt']=receipt
    inventory=plan_inventory(source,root/'all_layer_coverage.json');atomic_json(root/'hardware.json',hardware_inventory(root))
    registry=json.loads((source/'snnbench/resources/datasets.json').read_text());models=json.loads((source/'snnbench/resources/model_sources.json').read_text())
    prepared={}
    for path in archives:
        manifest=Path(path)/'dataset.json'
        if manifest.exists():prepared[json.loads(manifest.read_text())['contract']['dataset_id']]={'root':str(Path(path).resolve()),'manifest_sha256':file_hash(manifest)}
    atomic_json(root/'source_roster.json',{'data_sources':registry['datasets'],'prepared_canonical_datasets':prepared,'models':models,
        'policy':'Keep unprepared sources and native-author integration tasks visible; absence of downloaded data is not permission to substitute synthetic inputs.'})
    training_queue=None
    if materialize:
        from .queue import enqueue
        for profile in profiles:training_queue=enqueue(root/'train.sqlite',root/'manifests'/(profile+'.jsonl'))
    plan={'schema':'four_phase_campaign_v1','profiles':list(profiles),'seeds':list(seeds),'archives':[str(p) for p in archives],
        'training_implementation_sha256':implementation_hash(),
        'training_output':str((root/'runs').resolve()),'data_output':str(data_output),'counts':counts,'compile_commands':commands,'materialized':materialize,
        'training_queue':training_queue,'training_database':str((root/'train.sqlite').resolve()),
        'execution_order':['broad_train_capture_preliminary','all_original_layer_analysis','discovery_mechanism_theory_design','evidence_led_paper_and_followups'],
        'selection_policy':'All declared configurations and all results remain in the discovery set; engineering calibration does not select scientific winners.',
        'capture_contract':'Every saved checkpoint remains replayable. Complete coordinate trajectories are archived on declared fixed scene banks; coordinate-limited probe caches remain separate.',
        'phase2_contract':'All six layers, original subsections and specialized controls remain in all_layer_coverage.json. A shared 14-module receipt is one part of that coverage.',
        'source_counts':inventory['counts'],'runtime':'Source code is local; this command does not connect to or start a remote GPU server.'}
    atomic_json(root/'campaign.json',plan)
    atomic_json(root/'discovery_schema.json',{'fields':['finding_id','observation','conditions','run_ids','artifact_paths','effect_direction','seed_variation','candidate_explanations','discriminating_experiments','theory_route','model_or_metric_design','followup_results'],
        'candidate_categories':['new_phenomenon','ranking_reversal','invariant_metric_changed_function','changed_geometry_same_function','phase_transition','representation_readout_gap','helpful_or_harmful_history','unexpected_failure','mechanistic_model_improvement','theory_or_counterexample'],
        'policy':'Keep surprising, weak, negative and conflicting observations. Elaborate competing explanations and new experiments before organizing the paper.'})
    return plan


def stage_postprocessing(root,phase='preliminary',checkpoints='all',archive_scenes=None,archive_batch=8):
    root=Path(root);plan=json.loads((root/'campaign.json').read_text());runs=Path(plan['training_output']);database=root/('post_'+phase+'.sqlite');inserted=0;waiting=[]
    if phase not in ('preliminary','all_shared'):raise ValueError('Phase preliminary or all_shared')
    with connect(database) as con:
        for spec_path in sorted(runs.glob('*/spec.json')):
            run=spec_path.parent;status_path=run/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('status')!='trained':waiting.append({'run':str(run),'status':status.get('status','incomplete')});continue
            paths=sorted(set([run/'best.pt',run/'last.pt',*run.glob('step_*.pt'),*run.glob('epoch_*.pt')])) if checkpoints=='all' else [run/'best.pt',run/'last.pt']
            for ck in paths:
                if not ck.exists():continue
                for kind in (('archive','analyze') if phase=='preliminary' else ('analyze',)):
                    job={'kind':kind,'phase':phase,'run':str(run.resolve()),'checkpoint':ck.name,'checkpoint_sha256':file_hash(ck),'data':plan['data_output'],
                         'archive_scenes':archive_scenes,'archive_batch':archive_batch}
                    key=digest(job)[:32];before=con.total_changes;con.execute('INSERT OR IGNORE INTO jobs(id,spec) VALUES(?,?)',(key,json.dumps(job)));inserted+=con.total_changes-before
        counts=dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())
    receipt={'database':str(database.resolve()),'phase':phase,'new_jobs':inserted,'counts':counts,'waiting_runs':waiting,
             'all_original_analysis_complete':False,'phase2_remaining':'Source-specific adapters in all_layer_coverage.json remain explicit until executed.'}
    atomic_json(root/('post_'+phase+'.json'),receipt);return receipt


def _claim_post(con,owner,lease,max_attempts):
    con.execute('BEGIN IMMEDIATE');now=time.time()
    con.execute("UPDATE jobs SET state='pending',owner=NULL WHERE state='running' AND lease<? AND attempts<?",(now,max_attempts))
    con.execute("UPDATE jobs SET state='failed',error='lease expired' WHERE state='running' AND lease<? AND attempts>=?",(now,max_attempts))
    busy={json.loads(row[0])['run'] for row in con.execute("SELECT spec FROM jobs WHERE state='running'")}
    selected=None
    for row in con.execute("SELECT id,spec FROM jobs WHERE state='pending' AND attempts<? ORDER BY id",(max_attempts,)):
        if json.loads(row[1])['run'] not in busy:selected=row;break
    if selected:con.execute("UPDATE jobs SET state='running',owner=?,lease=?,attempts=attempts+1 WHERE id=?",(owner,now+lease,selected[0]))
    con.commit();return selected


def run_post_job(job,device='cpu'):
    run=Path(job['run']);checkpoint=run/job['checkpoint']
    if file_hash(checkpoint)!=job['checkpoint_sha256']:raise ValueError('Checkpoint changed after job staging')
    if job['kind']=='archive':
        from .archive import archive_run
        out=archive_run(run,job['data'],job['checkpoint'],device,scenes=job.get('archive_scenes'),batch_size=job.get('archive_batch',8));return {'archive':str(out)}
    from .analysis import analyze
    spec=RunSpec.load(run/'spec.json');modules=('task_controls','working_point') if job['phase']=='preliminary' else AnalysisSpec().modules
    override=replace(spec.analysis,modules=modules,eval_split='development')
    out=analyze(run,job['data'],job['checkpoint'],device,override);return {'analysis':str(out),'components':list(modules)}


def post_worker(database,device='cuda:0',one=False,lease_seconds=300,max_attempts=2):
    con=connect(database);owner=socket.gethostname()+':'+str(os.getpid())+':'+uuid.uuid4().hex[:8];logs=Path(database).with_suffix('.logs');logs.mkdir(parents=True,exist_ok=True)
    active={'process':None,'id':None}
    def stop(signum,frame):
        p=active['process']
        if p is not None and p.poll() is None:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        if active['id'] is not None:con.execute("UPDATE jobs SET state='pending',owner=NULL,lease=NULL WHERE id=? AND owner=?",(active['id'],owner));con.commit()
        raise SystemExit(128+signum)
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        while True:
            row=_claim_post(con,owner,lease_seconds,max_attempts)
            if row is None:break
            key,payload=row;active['id']=key;path=logs/(key+'.json');path.write_text(payload);log=logs/(key+'.log')
            with log.open('a') as f:
                p=subprocess.Popen([sys.executable,'-m','snnbench','post-job','--job',str(path),'--device',device],stdout=f,stderr=subprocess.STDOUT);active['process']=p
                while p.poll() is None:
                    time.sleep(2);con.execute("UPDATE jobs SET lease=? WHERE id=? AND owner=?",(time.time()+lease_seconds,key,owner));con.commit()
            error=None if p.returncode==0 else 'postprocess exit '+str(p.returncode)+'; '+str(log)
            con.execute("UPDATE jobs SET state=?,error=?,lease=NULL WHERE id=? AND owner=?",('failed' if error else 'done',error,key,owner));con.commit();active.update(process=None,id=None)
            if one:break
        return dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())
    finally:
        for s,h in handlers.items():signal.signal(s,h)
        con.close()
