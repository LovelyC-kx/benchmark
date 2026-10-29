"""L3.4 v0.2: paired readers, independent spectra and fresh scene holdout.

Selection and all reader weights are frozen and hashed BEFORE generating the
evaluation pool for that view. This is fresh development evidence from one
old one-update checkpoint, not the sealed audit split or formal benchmark.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from bridge import load_model,collect,flatten_contents
from parent_sampling import fixed_batch
from factor_sampling import factor_schema
from geometry_generalization import sample_geometry,geometry_terms,predict_error,gaussian_surrogate
from task_moments import teachers,labels,empirical_task_moments,conditional_forecast,gaussian_task_mean
from task_structure import support_tasks,crossfit_higher_alignment
from task_spectrum import crossfit_task_spectrum
from finite_gaussian import finite_gaussian_forecast
from approximation_diagnostics import berry_esseen_diagnostic,prediction_gap
from readout_design import spectral_candidates,select_before_evaluation
from paired_readouts import draw_plan,fit_linear_family,fit_independent_mlps,evaluate,paired_gain_certificate
from run_generalization import serial


def public_loss(e):return {k:v for k,v in e.items() if k!='per_independent_scene_loss'}


def task_directions(d,prior,count,seed):
    if prior=='sphere':return teachers(d,count,seed)
    if prior=='axis':return np.eye(d)
    if prior=='equal_support':return support_tasks(d,count,d,seed)
    raise ValueError('Declared task prior')


def weight_digest(models):
    h=hashlib.sha256()
    for name,curves in sorted(models.items()):
        h.update(name.encode())
        for p,fit in sorted(curves.items()):
            h.update(str(p).encode())
            if fit['family']=='mlp':
                for state in fit['models']:
                    for key in ('w1','b1','w2','b2'):h.update(state[key].tobytes())
            else:h.update(fit['weights'].tobytes())
    return h.hexdigest()


def prepare(raw,z,columns,group,prior,args,with_mlp=False):
    initial=sample_geometry(raw['geometry'],z['geometry'][:,columns]);origin=initial['response_mean']
    scale=float(np.sqrt(np.trace(initial['Psi'])/len(initial['Psi'])))
    if scale==0:scale=1.
    x={k:(a-origin)/scale for k,a in raw.items()};zg={k:a[:,columns] for k,a in z.items()}
    geometry=sample_geometry(x['geometry'],zg['geometry']);terms=geometry_terms(geometry['Psi'],geometry['Phi'],geometry['Omega'])
    count=16 if args.smoke else 64;sizes=(1,8,32,128) if args.smoke else (4,16,64,256)
    seed=35101+sum(map(ord,group));theta=task_directions(len(columns),prior,count,seed)
    dev=task_directions(len(columns),prior,count,seed+1000)
    # Axis family is a finite exhaustive prior, not a claim of unseen axes.
    prior_contract='Independent held-out directions from the same prior' if prior=='sphere' else ('Fixed finite axis family; directions are exhaustive, only scenes/readers held out' if prior=='axis' else 'Independent sign-balanced equal-support directions; semantic duplicates up to sign possible and recorded')
    y={k:labels(v,theta) for k,v in zg.items()};dy={k:labels(v,dev) for k,v in zg.items()}
    plan=draw_plan(len(x['train']),sizes,repeats=args.readout_repeats)
    candidates=spectral_candidates(geometry['Psi']);models={};dev_risks={}
    began=time.perf_counter()
    moment_x=np.concatenate((x['moment_a'],x['moment_b']));moment_y=np.concatenate((y['moment_a'],y['moment_b']))
    dev_my=np.concatenate((dy['moment_a'],dy['moment_b']))
    selectors=select_before_evaluation(geometry,candidates,moment_x,dev_my,x['forecast'],dy['forecast'],sizes)
    forecast_selector_seconds=time.perf_counter()-began;began=time.perf_counter()
    for name,b in candidates.items():
        models[name]=fit_linear_family(x['train']@b.T,y['train'],plan)
        fit=fit_linear_family(x['train']@b.T,dy['train'],plan)
        dev_risks[name]={p:evaluate(fit[p],x['forecast']@b.T,dy['forecast'])['mean_error'] for p in sizes}
    direct_selector_seconds=time.perf_counter()-began
    choices={'four_statistic':selectors['four_statistic_choices'],
        'development_moment':selectors['development_task_moment_choices'],
        'direct_development':{str(p):min(candidates,key=lambda name:(dev_risks[name][p],name!='identity',name)) for p in sizes}}
    models['ridge']=fit_linear_family(x['train'],y['train'],plan,'ridge')
    if with_mlp:models['mlp']=fit_independent_mlps(x['train'],y['train'],plan,epochs=80)
    ma=empirical_task_moments(x['moment_a'],y['moment_a']);mb=empirical_task_moments(x['moment_b'],y['moment_b'])
    moment=empirical_task_moments(moment_x,moment_y)
    control=np.linalg.solve(geometry['Omega'],geometry['Phi'].T).T
    higher=crossfit_higher_alignment(x['moment_a'],zg['moment_a'],x['moment_b'],zg['moment_b'],theta,
        Omega=np.eye(len(columns)),control_encoding=control)
    spectrum=crossfit_task_spectrum(geometry['Psi'],ma['m'],mb['m'])
    proxy_offset=moment_x.mean(0)
    # Exact Gaussian TRAINING reference, estimated parameters: no claim that
    # the real SNN response has become Gaussian.
    gx,gz=gaussian_surrogate(geometry,256,35302);gx+=proxy_offset;gy=labels(gz,theta)
    pre={}
    for p in sizes:
        exact_gauss=finite_gaussian_forecast(geometry['Psi'],geometry['Phi'],geometry['Omega'],theta,gx,gy,p,
            response_mean=proxy_offset,draws=args.gaussian_draws)
        gaussian_real=finite_gaussian_forecast(geometry['Psi'],geometry['Phi'],geometry['Omega'],theta,x['forecast'][:256],y['forecast'][:256],p,
            response_mean=proxy_offset,draws=args.gaussian_draws)
        pre[p]={'four_statistic':predict_error(terms,p),'actual_task_moment_normal':conditional_forecast(moment,x['forecast'],y['forecast'],p),
            'Gaussian_training_real_reference':gaussian_real,'Gaussian_training_Gaussian_reference':exact_gauss,
            'BE_diagnostic':berry_esseen_diagnostic(moment_x,moment_y,x['forecast'][:128],p)}
    canonical=theta.copy()
    for j in range(canonical.shape[1]):
        nonzero=np.flatnonzero(canonical[:,j]);canonical[:,j]*=np.sign(canonical[nonzero[0],j])
    unique=len(np.unique(np.round(canonical.T,12),axis=0))
    return {'models':models,'candidates':candidates,'theta':theta,'development_theta':dev,'plan':plan,'choices':choices,
        'origin':origin,'scale':scale,'geometry':geometry,'terms':terms,'higher':higher,'spectrum':spectrum,
        'moment':moment,'proxy_offset':proxy_offset,'pre_evaluation_forecasts':pre,'columns':columns,
        'prior':prior,'prior_contract':prior_contract,'unique_directions_up_to_sign':unique,
        'selection_details':selectors,'direct_development_risks':dev_risks,
        'selection_cost':{'theory_and_moment_selection_seconds':forecast_selector_seconds,
            'direct_development_fit_and_selection_seconds':direct_selector_seconds,
            'moment_oracle_scene_count':len(moment_x),'forecast_scene_count':len(x['forecast']),
            'structural_geometry_latent_scenes':len(x['geometry']),
            'per_task_unique_train_labels_across_frozen_readers':{str(p):len(np.unique(plan['indices'][:,:p])) for p in sizes}},
        'source_normalized_forecast_x':x['forecast'][:256],
        'source_normalized_forecast_y':y['forecast'][:256]}


def run(args):
    import torch
    torch.set_num_threads(1);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);model,cfg,meta=load_model(args.run_directory)
    checkpoint=Path(args.run_directory)/'best.pt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract=[('geometry',512,'geometry_fit',5000000),('moment_a',1024,'probe_train',5100000),
        ('moment_b',1024,'probe_train',5200000),('forecast',512,'development_test',5300000),('train',512,'probe_train',5400000)]
    # Prespecified family size: 9 sphere views plus two priors on two carrier
    # views; 3 role groups, 4 p values, 3 selectors. Dependence across views
    # or tasks does not invalidate this union bound.
    view_count=0
    for schedule in (range(6) if args.all_schedules else (0,)):
        base=[('C','last_post')]
        if schedule==0:
            base+=[('H1','last_post')]
            if args.carriers:base+=[('C','last_output'),('C','joint_post')]
        for focal,observation in base:
            view_count+=3 if args.task_priors and schedule==0 and focal=='C' and observation in ('last_post','joint_post') else 1
    family_size=view_count*3*4*3
    records=[];start=time.perf_counter();mlp_steps=0
    for schedule in (range(6) if args.all_schedules else (0,)):
        batches={k:fixed_batch(cfg.data,n,split,first,schedule) for k,n,split,first in contract}
        features={k:collect(model,b['x']) for k,b in batches.items()};z={k:flatten_contents(b,cfg.data) for k,b in batches.items()}
        views=[('C','last_post')]
        if schedule==0:
            views+=[('H1','last_post')]
            if args.carriers:views+=[('C','last_output'),('C','joint_post')]
        for focal,observation in views:
            schema=factor_schema(cfg.data,focal,batches['geometry']['source_present'])
            priors=('sphere','axis','equal_support') if args.task_priors and schedule==0 and focal=='C' and observation in ('last_post','joint_post') else ('sphere',)
            for prior in priors:
                folder=out/f'{focal}_{observation}_{prior}_schedule_{schedule}';folder.mkdir()
                raw={k:a[observation].astype(float) for k,a in features.items()};prepared={};freeze={}
                for group,roles in [('F',('focal',)),('G',('cotarget',)),('FG',('focal','cotarget'))]:
                    columns=[i for i,row in enumerate(schema['coordinate_roles']) if row['role'] in roles]
                    with_mlp=args.mlp and schedule==0 and focal=='C' and observation=='last_post' and prior=='sphere'
                    a=prepare(raw,z,columns,group,prior,args,with_mlp)
                    if with_mlp:mlp_steps+=sum(f['vectorized_optimizer_steps'] for f in a['models']['mlp'].values())
                    prepared[group]=a
                    freeze[group]={'choices':a['choices'],'weight_sha256':weight_digest(a['models']),
                        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'draw_plan_sha256':a['plan']['sha256'],'theta':a['theta'],'development_theta':a['development_theta'],
                        'task_prior':prior,'unique_directions_up_to_sign':a['unique_directions_up_to_sign']}
                freeze_text=json.dumps(serial(freeze),sort_keys=True,allow_nan=False)
                (folder/'selection_freeze.json').write_text(freeze_text)
                freeze_sha=hashlib.sha256(freeze_text.encode()).hexdigest()
                # Evaluation generation and SNN forward occur only AFTER all
                # three role systems have been selected/fitted and frozen.
                evbatch=fixed_batch(cfg.data,args.evaluation_scenes,'development_test',args.evaluation_first_id,schedule)
                ev=collect(model,evbatch['x']);evz=flatten_contents(evbatch,cfg.data);group_results={};arrays={}
                for group,a in prepared.items():
                    x=(ev[observation].astype(float)-a['origin'])/a['scale'];y=labels(evz[:,a['columns']],a['theta'])
                    family_results={};full_results={}
                    for name,curves in a['models'].items():
                        b=a['candidates'].get(name,np.eye(x.shape[1]));results={p:evaluate(f,x@b.T,y) for p,f in curves.items()}
                        full_results[name]=results;family_results[name]={p:public_loss(v) for p,v in results.items()}
                        for p,result in results.items():arrays[f'{group}__{name}__p{p}__scene_loss']=result['per_independent_scene_loss']
                        for p,f in curves.items():
                            if f['family']!='mlp':arrays[f'{group}__{name}__p{p}__weights']=f['weights']
                            else:
                                for j,state in enumerate(f['models']):
                                    for key in ('w1','b1','w2','b2'):arrays[f'{group}__mlp__p{p}__model{j}__{key}']=state[key]
                    gains={}
                    for selector,choices in a['choices'].items():
                        gains[selector]={p:dict(paired_gain_certificate(full_results['identity'][p]['per_independent_scene_loss'],
                            full_results[choices[str(p)]][p]['per_independent_scene_loss'],family_size=family_size),selected_candidate=choices[str(p)]) for p in a['plan']['sample_sizes']}
                    pre=a['pre_evaluation_forecasts'];gap={}
                    for p in a['plan']['sample_sizes']:
                        f=pre[p]
                        gap[p]=prediction_gap(full_results['identity'][p]['mean_error'],f['actual_task_moment_normal']['normal_weight_forecast'],
                            f['Gaussian_training_real_reference']['predicted_error'],f['Gaussian_training_Gaussian_reference']['predicted_error'],f['four_statistic']['predicted_error'])
                    group_results[group]={'prior':prior,'prior_contract':a['prior_contract'],
                        'four_statistic_scope':'Uniform sphere-task reference; not calibrated for this non-spherical prior' if prior!='sphere' else 'Uniform sphere-task approximation; finite task sampling remains',
                        'prior_direction_count':a['theta'].shape[1],'unique_directions_up_to_sign':a['unique_directions_up_to_sign'],
                        'observation_origin':a['origin'],'shared_geometry_fit_scalar_scale':a['scale'],
                        'four_statistics':a['terms'],'orthogonalized_higher_order_alignment':a['higher'],
                        'state_energy_and_task_spectrum':a['spectrum'],'pre_evaluation_forecasts':pre,
                        'reader_results':family_results,'selected_reader_gains':gains,
                        'distribution_replacement_gap_with_independent_reference_sampling':gap,
                        'selection_cost':a['selection_cost'],'direct_development_risks':a['direct_development_risks'],
                        'uniform_random_candidate_expected_error':{p:float(np.mean([family_results[k][p]['mean_error'] for k in a['candidates']])) for p in a['plan']['sample_sizes']},
                        'label_budget_contract':'Each fitted reader sees exactly p labels per task; same nested p-example plans across candidates, ridge and MLP initializations. Structural/moment/development labels are separate extra analysis access.'}
                    arrays[group+'__theta']=a['theta'];arrays[group+'__train_indices']=a['plan']['indices']
                    arrays[group+'__evaluation_labels']=y
                if any(freeze[g]['weight_sha256']!=weight_digest(prepared[g]['models']) for g in prepared):raise AssertionError('Frozen readers mutated during evaluation')
                report={'module':'L3.4','version':'0.2','checkpoint_id':meta['effective_id'],'schedule':schedule,
                    'focal':focal,'observation':observation,'task_prior':prior,'selection_freeze_sha256':freeze_sha,
                    'selection_frozen_before_evaluation_generation':True,'evaluation_scene_first_id':args.evaluation_first_id,
                    'evaluation_scenes':args.evaluation_scenes,'planned_confidence_family_size':family_size,
                    'independent_pool_contract':contract,'target_groups':group_results,'audit_test_accessed':False,
                    'new_SNN_optimizer_updates':0,'kind':'fresh_development_pipeline_evidence_one_old_one_update_checkpoint'}
                (folder/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False));np.savez_compressed(folder/'frozen_readers_and_scene_losses.npz',**arrays)
                records.append({'folder':folder.name,'prior':prior,'groups':{g:{k:group_results[g]['selected_reader_gains'][k][128 if args.smoke else 256] for k in group_results[g]['selected_reader_gains']} for g in group_results}})
                print(json.dumps({'completed_view':folder.name,'eval_scenes':args.evaluation_scenes}),flush=True)
    if before!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():raise AssertionError('SNN checkpoint changed')
    result={'module':'L3.4','version':'0.2','semantic_view_count':len(records),'target_group_views':3*len(records),
        'independent_SNN_checkpoints':1,'source_original_optimizer_updates':meta['optimizer_updates'],
        'checkpoint_sha256':before,'checkpoint_unchanged':True,'new_SNN_training_runs':0,'new_SNN_optimizer_updates':0,
        'MLP_vectorized_optimizer_steps':mlp_steps,'audit_test_accessed':False,'formal_benchmark_completed':0,'formal_benchmark_planned':435,
        'all_selections_frozen_before_evaluation':True,'planned_confidence_family_size':family_size,
        'views':records,'elapsed_seconds':time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--all-schedules',action='store_true');p.add_argument('--carriers',action='store_true');p.add_argument('--task-priors',action='store_true');p.add_argument('--mlp',action='store_true')
    p.add_argument('--evaluation-scenes',type=int,default=8192);p.add_argument('--readout-repeats',type=int,default=4);p.add_argument('--gaussian-draws',type=int,default=512)
    p.add_argument('--evaluation-first-id',type=int,default=5600000)
    args=p.parse_args();s=run(args);print(json.dumps({k:s[k] for k in ('semantic_view_count','target_group_views','elapsed_seconds','checkpoint_unchanged')}))
