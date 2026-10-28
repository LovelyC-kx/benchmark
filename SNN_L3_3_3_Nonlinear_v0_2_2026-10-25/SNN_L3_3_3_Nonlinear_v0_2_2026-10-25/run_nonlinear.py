"""One frozen checkpoint, grouped nonlinear dependence and fixed-reader risk."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from bridge import load_model,collect,flatten_contents
from parent_sampling import fixed_batch,SCHEDULES
from factor_sampling import full_lattice,conditional_panels,factor_schema
from functional_anova import decompose,power_summary,error_risk_budget,paired_bootstrap,conditional_risk,conditional_mean_lattice,mask_for
from decoding import fit_probe,fit_mlp,predict,mse,probe_metadata
from context_audit import audit_contexts
from task_objective import requested_objective,whole_task_report
from run_calibrations import serial


def targets_of(batch,cfg,schema):
    z=flatten_contents(batch,cfg)
    indices={name:np.array([i for i,row in enumerate(schema['coordinate_roles']) if row['role']==role],int) for name,role in [('F','focal'),('G','cotarget')]}
    return z,indices


def native_map(model,schema,group,observation):
    role='focal' if group=='F' else 'cotarget'
    rows=[r['source_id']*4+r['coordinate'] for r in schema['coordinate_roles'] if r['role']==role]
    w=model.core.head.weight.detach().cpu().numpy()[rows].astype(float)
    b=model.core.head.bias.detach().cpu().numpy()[rows].astype(float)
    if observation=='joint_post':w=np.concatenate((np.zeros_like(w),w),axis=1)
    return w,b,rows


def analyze(model,run,metadata,schedule,focal,observation,smoke,nested):
    start=time.perf_counter();cfg=run.data
    if run.objective!='regression' or cfg.extra_writes!=3:
        raise ValueError('Initial pipeline requires regression and every source present')
    nfit,nselect,neval,npairs=(128,64,256,256) if smoke else (1024,512,4096,4096)
    nscenes,repeats=(32,4) if smoke else (256,8)
    counts=(32,128) if smoke else (64,256,1024)
    probe_seeds=(1,) if smoke else (1,2,3)
    batches={name:fixed_batch(cfg,n,split,first,schedule) for name,n,split,first in [
        ('train',nfit,'probe_train',3100000),('selection',nselect,'probe_val',3200000),
        ('evaluation',neval,'development_test',3300000)]}
    collected={name:collect(model,batch['x']) for name,batch in batches.items()}
    r={name:values[observation].astype(float) for name,values in collected.items()}
    schema=factor_schema(cfg,focal,batches['train']['source_present']);names=schema['names']
    z={name:flatten_contents(batch,cfg) for name,batch in batches.items()}
    _,indices=targets_of(batches['train'],cfg,schema)
    lattice,_,pairmeta=full_lattice(cfg,npairs,'development_test',3400000,3500000,schedule,focal)
    lattice_r=np.stack([collect(model,lattice[mask]['x'])[observation] for mask in range(len(lattice))]).astype(float)
    lattice_z=np.stack([flatten_contents(lattice[mask],cfg) for mask in range(len(lattice))])
    statefit=decompose(lattice_r,names)
    hold_masks=[mask_for((name,),names) for name in ('F','G')]
    panels,scene_targets,_=conditional_panels(cfg,nscenes,repeats,'development_test',3600000,schedule,focal,hold_masks)
    repeat_r={}
    for mask,x in panels.items():
        shape=x.shape;repeat_r[mask]=collect(model,x.reshape(-1,*shape[2:]))[observation].astype(float).reshape(nscenes,repeats,-1)
    reports={};arrays={'lattice_r':lattice_r,'lattice_z':lattice_z,
        'lattice_component_covariances':statefit['components'],
        'train_r':r['train'],'selection_r':r['selection'],'evaluation_r':r['evaluation'],
        'train_z':z['train'],'selection_z':z['selection'],'evaluation_z':z['evaluation'],
        'a_scene_ids':pairmeta['a_scene_ids'],'b_scene_ids':pairmeta['b_scene_ids']}
    readers_by_group={};max_embedding=0.;max_transport=0.;max_risk_closure=0.;max_repeat_closure=0.;probe_updates=0;probe_fit_count=0
    for group,columns in indices.items():
        if not len(columns):continue
        yt,yv,ye=[z[name][:,columns] for name in ('train','selection','evaluation')]
        reader_fits={};curves={}
        for family in ('linear','quadratic'):
            curve=[]
            for n in counts:
                fit=fit_probe(r['train'][:n],yt[:n],r['selection'],yv,family)
                curve.append({'train_count':n,'fit':probe_metadata(fit),'evaluation_mse':mse(predict(fit,r['evaluation']),ye)})
                probe_fit_count+=1
                if n==counts[-1]:reader_fits[family]=fit
            curves[family]=curve
        for seed in probe_seeds:
            curve=[]
            for n in counts:
                fit=fit_mlp(r['train'][:n],yt[:n],r['selection'],yv,seed=seed,width=64,epochs=80 if smoke else 200)
                curve.append({'train_count':n,'fit':probe_metadata(fit),'evaluation_mse':mse(predict(fit,r['evaluation']),ye)})
                probe_updates+=fit['total_optimizer_updates'];probe_fit_count+=1
                if n==counts[-1]:reader_fits['mlp_seed_'+str(seed)]=fit
            curves['mlp_seed_'+str(seed)]=curve
        w,b,headrows=native_map(model,schema,group,observation)
        native=r['evaluation'] @ w.T+b
        embedding=float(np.max(np.abs(native-collected['evaluation']['prediction'][:,headrows])))
        max_embedding=max(max_embedding,embedding)
        if embedding>1e-6:raise AssertionError('Original task-head embedding failed')
        readers_by_group[group]=reader_fits
        fixed={'original_task_head':lambda x,w=w,b=b:x @ w.T+b}
        fixed.update({name:(lambda x,f=f:predict(f,x)) for name,f in reader_fits.items()})
        fixed_reports={}
        for name,reader in fixed.items():
            predictions=reader(lattice_r.reshape(-1,lattice_r.shape[-1])).reshape(len(lattice),npairs,len(columns))
            errors=predictions-lattice_z[:,:,columns]
            prediction_fit=decompose(predictions,names)
            error_fit,budget=error_risk_budget(errors,names,(group,))
            interval=paired_bootstrap(errors,names,(group,),draws=32 if smoke else 200,seed=33371)
            held=mask_for((group,),names);rr=repeat_r[held]
            pr=reader(rr.reshape(-1,rr.shape[-1])).reshape(nscenes,repeats,len(columns))
            conditional=conditional_risk(pr,scene_targets[group])
            difference=budget['target_contract']['non_target_conditional_variability_power']-conditional['conditional_variability']
            if name=='original_task_head':
                transported=np.einsum('ai,sij,bj->sab',w,statefit['components'],w)
                transport=float(np.max(np.abs(transported-prediction_fit['components'])))
                max_transport=max(max_transport,transport)
            else:transport=None
            max_risk_closure=max(max_risk_closure,abs(budget['mse_identity_error']))
            max_repeat_closure=max(max_repeat_closure,abs(conditional['identity_error']))
            fixed_reports[name]={'natural_evaluation_mse':mse(reader(r['evaluation']),ye),
                'prediction_anova':power_summary(prediction_fit,(group,)),
                'error_anova_risk_budget':budget,'paired_row_bootstrap':interval,
                'independent_conditioned_repeat_risk':{k:v for k,v in conditional.items() if not k.startswith('per_scene')},
                'independent_estimator_non_target_variability_difference':difference,
                'native_linear_covariance_transport_max_error':transport,
                'warning':'Independent ANOVA and repeat estimates need not match exactly; tiny smoke estimates are not scientific effects.'}
            arrays[group+'__'+name+'__lattice_predictions']=predictions
            arrays[group+'__'+name+'__error_component_covariances']=error_fit['components']
            arrays[group+'__'+name+'__repeat_predictions']=pr
        reports[group]={'target_coordinate_indices':columns,'learning_curves':curves,
            'fixed_readers':fixed_reports,'native_head_embedding_error':embedding,
            'interpretation':'G remains a useful requested target. Lower variability for F alone is insufficient evidence of globally better memory.'}
        for reader_name,fit in reader_fits.items():
            for key,value in fit.items():
                if isinstance(value,np.ndarray):arrays['probe__'+group+'__'+reader_name+'__'+key]=value
    nested_report={'status':'not_requested'}
    if nested:
        ns,nb=(8,4) if smoke else (64,16)
        cp,_,_=conditional_panels(cfg,ns,nb,'development_test',3700000,schedule,focal,list(range(2**len(names))),panel_seed=33302)
        nested_r={}
        for mask,x in cp.items():
            nested_r[mask]=collect(model,x.reshape(-1,*x.shape[2:]))[observation].astype(float).reshape(ns,nb,-1)
        fn=conditional_mean_lattice(nested_r,names)
        nested_report={'status':'computed','independent_base_scenes':ns,'iid_completions':nb,
            'reconstruction_max_error':fn['reconstruction_max_error'],'scope':fn['scope'],
            'warning':'Mobius reconstruction closes by algebra. It does not validate finite-completion orthogonality, neuron assignments or population effect accuracy.'}
        arrays['nested_conditional_means']=fn['conditional_means'];arrays['nested_component_functions']=fn['component_functions_estimated']
        arrays['nested_conditional_mean_mc_variance']=fn['conditional_mean_mc_variance']
    objective=requested_objective(cfg,schema['coordinate_roles'],run.loss_mode,run.lambda_extra)
    ow=model.core.head.weight.detach().cpu().numpy()[objective['native_head_rows']].astype(float)
    ob=model.core.head.bias.detach().cpu().numpy()[objective['native_head_rows']].astype(float)
    if observation=='joint_post':ow=np.concatenate((np.zeros_like(ow),ow),axis=1)
    original_objective,weighted_components=whole_task_report(lattice_r @ ow.T+ob,
        lattice_z[:,:,objective['visible_latent_indices']],r['evaluation'] @ ow.T+ob,
        z['evaluation'][:,objective['visible_latent_indices']],names,objective)
    arrays['original_whole_task_weighted_error_component_covariances']=weighted_components
    report={'module':'L3.3.3','version':'0.2','date':'2026-10-25','analysis_kind':'pipeline_smoke' if smoke else 'development_analysis',
        'source_checkpoint_id':metadata['effective_id'],'source_checkpoint_original_optimizer_updates':metadata.get('optimizer_updates'),
        'schedule_index':schedule,'schedule_times':SCHEDULES[schedule],'focal':focal,'observation':observation,
        'factor_schema':schema,'observation_dimension':r['train'].shape[1],
        'sampling':{'probe_fit':{'split':'probe_train','first_id':3100000,'n':nfit},'probe_selection':{'split':'probe_val','first_id':3200000,'n':nselect},
                    'natural_evaluation':{'split':'development_test','first_id':3300000,'n':neval},'anova_evaluation':{'split':'development_test','a_first':3400000,'b_first':3500000,'n_independent_pairs':npairs,'hybrid_count':len(lattice)},
                    'conditional_repeat_evaluation':{'split':'development_test','first_id':3600000,'scenes':nscenes,'repeats':repeats},
                    'nested_conditional_functions':{'split':'development_test','first_id':3700000,'requested':nested}},
        'representation_grouped_anova':power_summary(statefit), 'target_readers':reports,'nested_conditional_function_audit':nested_report,
        'original_whole_task_objective':original_objective,
        'checks':{'max_original_head_embedding_error':max_embedding,'max_native_covariance_transport_error':max_transport,
                  'max_error_risk_closure':max_risk_closure,'max_conditional_risk_closure':max_repeat_closure,
                  'representation_covariance_closure':statefit['covariance_closure_max_error']},
        'new_snn_training_runs':0,'new_snn_optimizer_updates':0,'diagnostic_probe_fit_count':probe_fit_count,
        'diagnostic_mlp_optimizer_updates':probe_updates,'audit_test_accessed':False,'elapsed_seconds':time.perf_counter()-start,
        'scope':'Nonlinear grouped dependence, finite reader recovery and fixed-reader error allocation; not information absence, synaptic mechanism, Sobol-as-MI, causal deletion benefit or architecture ranking.'}
    context={'schedule_index':schedule,'train_r':r['train'],'select_r':r['selection'],'evaluation_r':r['evaluation'],
             'train_y':z['train'],'select_y':z['selection'],'evaluation_y':z['evaluation'],'target_slices':indices,'readers':readers_by_group}
    return report,arrays,context


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--smoke',action='store_true');p.add_argument('--all-schedules',action='store_true');p.add_argument('--schedule-index',type=int,default=0)
    p.add_argument('--focal',choices=('C','H1'),default='C');p.add_argument('--observation',choices=('last_post','joint_post'),default='last_post');p.add_argument('--nested',action='store_true');args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    if args.schedule_index not in range(6):raise ValueError('Schedule 0..5 required')
    model,run,metadata=load_model(args.run_directory,args.device)
    before=hashlib.sha256((Path(args.run_directory)/'best.pt').read_bytes()).hexdigest()
    out.mkdir(parents=True)
    reports=[];contexts=[]
    for schedule in (range(6) if args.all_schedules else (args.schedule_index,)):
        report,arrays,context=analyze(model,run,metadata,schedule,args.focal,args.observation,args.smoke,args.nested)
        folder=out/f'schedule_{schedule}';folder.mkdir()
        (folder/'report.json').write_text(json.dumps(serial(report),ensure_ascii=False,indent=2))
        np.savez_compressed(folder/'analysis_arrays.npz',**arrays)
        reports.append(report);contexts.append(context)
        print(json.dumps({'schedule':schedule,'focal':args.focal,'observation':args.observation,'checks':report['checks'],'probe_fits':report['diagnostic_probe_fit_count']},ensure_ascii=False),flush=True)
    transfer=audit_contexts(contexts) if args.all_schedules else {'status':'requires_all_six_schedules'}
    after=hashlib.sha256((Path(args.run_directory)/'best.pt').read_bytes()).hexdigest()
    if before!=after:raise AssertionError('Source checkpoint bytes changed')
    summary={'module':'L3.3.3','analysis_kind':'pipeline_smoke' if args.smoke else 'development_analysis',
        'checkpoint_id':metadata['effective_id'],'checkpoint_sha256':before,'checkpoint_unchanged':before==after,
        'analysis_view_count':len(reports),'independent_checkpoints':1,'source_optimizer_updates':metadata.get('optimizer_updates'),
        'new_snn_training_runs':0,'new_snn_optimizer_updates':0,'new_probe_fit_calls_excluding_context_shared':sum(r['diagnostic_probe_fit_count'] for r in reports),
        'diagnostic_mlp_optimizer_updates':sum(r['diagnostic_mlp_optimizer_updates'] for r in reports),
        'context_shared_ridge_fit_calls':8 if args.all_schedules else 0,
        'cross_schedule_audit':transfer,'checks':[r['checks'] for r in reports],
        'audit_test_accessed':False,'formal_benchmark_completed':0,'formal_benchmark_planned':435}
    (out/'summary.json').write_text(json.dumps(serial(summary),ensure_ascii=False,indent=2))
    from robust_review import augment_folder
    augment_folder(out,draws=32 if args.smoke else 200)
    print(json.dumps({'summary':str(out/'summary.json'),'unchanged_checkpoint':before==after,'views':len(reports)},ensure_ascii=False))
