"""Independent forecast and held-out-task learning on one frozen SNN.

Ridge has a fixed regularizer and no validation labels. Optional MLPs have
independent parameter blocks per task, fixed epochs and no evaluation access.
All task directions remain in the evaluation, including imbalanced tasks.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from bridge import load_model,collect,flatten_contents
from parent_sampling import fixed_batch
from factor_sampling import factor_schema
from geometry_generalization import sample_geometry,geometry_terms,predict_error,gaussian_surrogate,crossfit_signal_power
from task_moments import teachers,labels,empirical_task_moments,conditional_forecast,nonlinear_alignment,hebbian_curves,classification_error
from readout_design import spectral_candidates,select_before_evaluation


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x


def ridge_curves(tx,ty,ex,ey,sizes,repeats=8,seed=34003,alpha=1e-3):
    rng=np.random.default_rng(seed);risks={p:[] for p in sizes}
    for _ in range(repeats):
        idx=rng.permutation(len(tx))[:max(sizes)]
        for p in sizes:
            x,y=tx[idx[:p]],ty[idx[:p]]
            w=np.linalg.solve(x.T@x/p+alpha*np.eye(x.shape[1]),x.T@y/p)
            risks[p].append(classification_error(ex@w,ey)['per_task_error'])
    return [{'samples':p,'mean_error':float(np.mean(risks[p])),
        'per_task_mean_error':np.mean(risks[p],axis=0),'new_task_label_budget':p,
        'regularizer':alpha,'validation_labels':0,'intercept':False} for p in sizes]


def mlp_curves(tx,ty,ex,ey,sizes,epochs=80,seeds=(1,2),width=16):
    import torch
    results=[];tasks=ty.shape[1];dim=tx.shape[1]
    # Covariance-only global scaling, common to every task, is fit on the
    # training pool. Report this unlabelled access separately from p labels.
    scale=max(float(np.sqrt(np.mean(tx*tx))),1e-12)
    tx,ex=tx/scale,ex/scale
    for p in sizes:
        risks=[]
        for seed in seeds:
            torch.manual_seed(seed);idx=np.random.default_rng(34003+seed).permutation(len(tx))[:p]
            x=torch.tensor(tx[idx],dtype=torch.float32);y=torch.tensor(ty[idx],dtype=torch.float32)
            xe=torch.tensor(ex,dtype=torch.float32)
            w1=torch.nn.Parameter(torch.randn(tasks,dim,width)/np.sqrt(dim));b1=torch.nn.Parameter(torch.zeros(tasks,width))
            w2=torch.nn.Parameter(torch.randn(tasks,width)/np.sqrt(width));b2=torch.nn.Parameter(torch.zeros(tasks))
            opt=torch.optim.Adam((w1,b1,w2,b2),lr=.02)
            def score(a):return (torch.tanh(torch.einsum('nd,tdh->nth',a,w1)+b1)*w2).sum(-1)+b2
            for _ in range(epochs):
                opt.zero_grad();s=score(x)
                # Sum of independent task objectives; no shared hidden weights.
                loss=torch.nn.functional.softplus(-y*s).mean(0).sum()+1e-4*(w1.square().sum()+w2.square().sum())
                loss.backward();opt.step()
            with torch.no_grad():risks.append(classification_error(score(xe).numpy(),ey)['per_task_error'])
        results.append({'samples':p,'mean_error':float(np.mean(risks)),
            'per_task_seed_mean_error':np.mean(risks,axis=0),'new_task_label_budget':p,
            'validation_labels':0,'epochs':epochs,'seeds':list(seeds),'width':width,
            'independent_task_parameter_blocks':tasks,'vectorized_optimizer_steps':epochs*len(seeds),
            'training_index_seed_by_initialization':{str(seed):34003+seed for seed in seeds},
            'sampling_scope':'Same p-label count and train pool as other families; these MLP index seeds are not paired split-by-split with the Hebbian/ridge curves.',
            'unlabelled_pool_for_global_scale':len(tx),'scale':scale})
    return results


def analyze_group(raw,z,columns,group,smoke,native_predictions=None,mlp=False):
    sizes=(1,8,32,128) if smoke else (4,16,64,256);repeats=8 if smoke else 32
    g=sample_geometry(raw['geometry'],z['geometry'][:,columns]);terms=geometry_terms(g['Psi'],g['Phi'],g['Omega'])
    x={k:v-g['response_mean'] for k,v in raw.items()};zg={k:v[:,columns] for k,v in z.items()}
    count=16 if smoke else 128
    theta=teachers(len(columns),count,34101+sum(map(ord,group)))
    dev_theta=teachers(len(columns),count,34901+sum(map(ord,group)))
    y={k:labels(v,theta) for k,v in zg.items()};devy={k:labels(v,dev_theta) for k,v in zg.items()}
    moment=empirical_task_moments(x['moments'],y['moments'])
    alignment=nonlinear_alignment(x['moments'],zg['moments'],y['moments'],theta,g)
    candidates=spectral_candidates(g['Psi'])
    selection=select_before_evaluation(g,candidates,x['moments'],devy['moments'],x['forecast'],devy['forecast'],sizes)
    empirical={name:hebbian_curves(x['train']@b.T,y['train'],x['evaluation']@b.T,y['evaluation'],sizes,repeats) for name,b in candidates.items()}
    decisions=[]
    for i,p in enumerate(sizes):
        identity=empirical['identity'][i]['mean_error'];choice=selection['four_statistic_choices'][str(p)];devchoice=selection['development_task_moment_choices'][str(p)]
        decisions.append({'samples':p,'identity_error':identity,'four_stat_choice':choice,
            'four_stat_selected_error':empirical[choice][i]['mean_error'],
            'development_moment_choice':devchoice,'development_moment_selected_error':empirical[devchoice][i]['mean_error'],
            'four_stat_error_reduction':identity-empirical[choice][i]['mean_error'],
            'development_moment_error_reduction':identity-empirical[devchoice][i]['mean_error'],
            'paired_readout_split_error_reductions_four_stat':empirical['identity'][i]['per_readout_split_mean_error']-empirical[choice][i]['per_readout_split_mean_error'],
            'paired_split_uncertainty_scope':'Fixed task/scene pools and one checkpoint; not architecture or population confidence.',
            'uniform_random_candidate_expected_error':float(np.mean([v[i]['mean_error'] for v in empirical.values()]))})
    surrogate={};surz={}
    for i,(name,n) in enumerate((k,len(v)) for k,v in x.items()):
        surrogate[name],surz[name]=gaussian_surrogate(g,n,34500+i)
    sy={k:labels(v,theta) for k,v in surz.items()}
    smoment=empirical_task_moments(surrogate['moments'],sy['moments'])
    report={'target_group':group,'latent_coordinate_indices':list(map(int,columns)),
        'latent_dimension':len(columns),'response_dimension':raw['geometry'].shape[1],
        'random_tasks':count,'development_tasks':count,'task_direction_seed':34101+sum(map(ord,group)),
        'geometry':g,'four_statistics':terms,'four_statistic_forecast':[predict_error(terms,p) for p in sizes],
        'task_moment_forecast':[conditional_forecast(moment,x['forecast'],y['forecast'],p) for p in sizes],
        'nonlinear_task_alignment':alignment,
        'independent_fit_signal_norm':crossfit_signal_power(raw['geometry'],zg['geometry'],raw['moments'],zg['moments']),
        'hebbian_candidate_curves':empirical,'ridge_curve':ridge_curves(x['train'],y['train'],x['evaluation'],y['evaluation'],sizes,repeats),
        'readout_design':{'selection':selection,'held_out_task_results':decisions},
        'matched_Gaussian_control':{'empirical_Hebbian':hebbian_curves(surrogate['train'],sy['train'],surrogate['evaluation'],sy['evaluation'],sizes,repeats),
            'task_moment_forecast':[conditional_forecast(smoment,surrogate['forecast'],sy['forecast'],p) for p in sizes],
            'nonlinear_alignment':nonlinear_alignment(surrogate['moments'],surz['moments'],sy['moments'],theta,g),
            'scope':'Population Gaussian moments equal the finite geometry-fit moments. Separate fresh pools; finite response/latent means and sample errors are not matched exactly.'},
        'label_and_access_contract':{'per_new_task_labels_for_Hebb_ridge_MLP':'p; fixed hyperparameters, no validation labels',
            'moment_forecast_labels':'Independent moment and forecast pools, not supplied to fitted evaluation readers.',
            'selector_labels':'Separate development task directions on moment/forecast pools.',
            'response_origin':'Geometry-fit mean only; evaluation never centered on itself.',
            'latent_tasks':'Original latent units and zero threshold; no latent whitening.',
            'all_evaluation_tasks_retained':True,'tie_policy':'half'}}
    if native_predictions is not None:
        report['derived_native_regression_composition']=classification_error(native_predictions[:,columns]@theta,y['evaluation'])
        report['derived_native_regression_composition']['scope']='Original regression head composed with new teacher directions; original regression supervision is a different budget. This is not a p-label trained classifier, and may use a different carrier from the probed one.'
    if mlp:report['restricted_MLP_curve']=mlp_curves(x['train'],y['train'],x['evaluation'],y['evaluation'],sizes)
    arrays={'theta':theta,'development_theta':dev_theta,'moment_M':moment['M'],'moment_m':moment['m']}
    return report,arrays


def run(args):
    import torch
    torch.set_num_threads(1);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);model,cfg,meta=load_model(args.run_directory)
    checkpoint=Path(args.run_directory)/'best.pt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    n=(256,512,512,512,512) if args.smoke else (2048,4096,4096,2048,4096)
    contract=[('geometry',n[0],'geometry_fit',4000000),('moments',n[1],'probe_train',4100000),
        ('forecast',n[2],'development_test',4200000),('train',n[3],'probe_train',4300000),('evaluation',n[4],'development_test',4400000)]
    views=[];start=time.perf_counter()
    observations=('last_post','last_output','joint_post') if args.carriers else ('last_post',)
    for schedule in (range(6) if args.all_schedules else (0,)):
        batches={k:fixed_batch(cfg.data,count,split,first,schedule) for k,count,split,first in contract}
        responses={k:collect(model,b['x']) for k,b in batches.items()};z={k:flatten_contents(b,cfg.data) for k,b in batches.items()}
        for focal in (('C','H1') if schedule==0 else ('C',)):
            schema=factor_schema(cfg.data,focal,batches['geometry']['source_present'])
            for observation in (observations if focal=='C' and schedule==0 else ('last_post',)):
                raw={k:v[observation].astype(float) for k,v in responses.items()};group_reports={};arrays={}
                for group,roles in [('F',('focal',)),('G',('cotarget',)),('FG',('focal','cotarget'))]:
                    columns=[i for i,row in enumerate(schema['coordinate_roles']) if row['role'] in roles]
                    if not columns:continue
                    report,a=analyze_group(raw,z,columns,group,args.smoke,responses['evaluation']['prediction'],args.mlp and schedule==0 and focal=='C' and observation=='last_post')
                    group_reports[group]=report;arrays.update({group+'__'+k:v for k,v in a.items()})
                folder=out/f'{focal}_{observation}_schedule_{schedule}';folder.mkdir()
                report={'module':'L3.4','kind':'frozen_checkpoint_pipeline_smoke' if args.smoke else 'development_analysis',
                    'schedule':schedule,'focal':focal,'observation':observation,'target_groups':group_reports,
                    'independent_sampling_pools':[{ 'role':k,'count':cnt,'split':split,'first_id':first} for k,cnt,split,first in contract],
                    'checkpoint_id':meta['effective_id'],'new_SNN_optimizer_updates':0,'audit_test_accessed':False}
                (folder/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False))
                np.savez_compressed(folder/'moments_and_tasks.npz',**arrays,**{k+'__response':v for k,v in raw.items()},**{k+'__latents':v for k,v in z.items()})
                views.append({'focal':focal,'observation':observation,'schedule':schedule,'folder':folder.name,
                    'group_summaries':{g:{'four_stat':r['four_statistic_forecast'],'actual':r['hebbian_candidate_curves']['identity'],'design':r['readout_design']['held_out_task_results']} for g,r in group_reports.items()}})
                print(json.dumps({'view':folder.name,'groups':list(group_reports)}),flush=True)
    after=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if before!=after:raise AssertionError('Checkpoint changed')
    summary={'module':'L3.4','view_count':len(views),'independent_checkpoints':1,'checkpoint_id':meta['effective_id'],
        'checkpoint_sha256':before,'checkpoint_unchanged':True,'source_original_optimizer_updates':meta['optimizer_updates'],
        'new_SNN_training_runs':0,'new_SNN_optimizer_updates':0,'formal_benchmark_completed':0,'formal_benchmark_planned':435,
        'audit_test_accessed':False,'views':views,'elapsed_seconds':time.perf_counter()-start,
        'scope':'Pipeline/calibration evidence from one originally one-update checkpoint. Repeated tasks and readout splits do not establish architecture-level conclusions.'}
    (out/'summary.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--all-schedules',action='store_true');p.add_argument('--carriers',action='store_true');p.add_argument('--mlp',action='store_true');args=p.parse_args()
    s=run(args);print(json.dumps({k:s[k] for k in ('view_count','checkpoint_unchanged','elapsed_seconds')}))
