"""Freeze dynamics; compare a declared 3-carrier x 5-summary access grid."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from bridge35 import load_model,fixed_batch
from capture35 import capture
from access_operators import aggregate,permission_spec,MODES,output_subspaces
from native_head_audit import parameter_hash
from temporal_probes import fit_head,loss_by_group,paired_bound,tree_hash
from temporal_contracts import independent_ids
from artifacts35 import save_tree,load_tree,write_json
ROOT=Path(__file__).resolve().parent
ALPHAS=(.0001,.001,.01)

def targets(batch,k):
    z=np.concatenate([batch['latents'][:,0,:k],batch['latents'][:,4,:k]],1).astype(float)
    rng=np.random.default_rng(41042);theta=rng.normal(size=(2,k,16));theta/=np.linalg.norm(theta,axis=1,keepdims=True)
    binary=np.concatenate([np.where(z[:,j*k:(j+1)*k]@theta[j]>=0,1.,-1.) for j in range(2)],1)
    groups=[{'source':s,'continuous_columns':list(range(j*k,(j+1)*k)),
        'binary_columns':list(range(2*k+16*j,2*k+16*(j+1)))} for j,s in enumerate(('H1','C'))]
    return np.c_[z,binary],theta,groups

def collect(model,batch):
    r=capture(model,batch['x']);c=r['carriers']
    wanted={'output':c['backbone.cells.1/output'],'pre':c['backbone.cells.1/pre'],'post':c['backbone.cells.1/post']}
    views={p+'__'+m:aggregate(x,m) for p,x in wanted.items() for m in MODES}
    return views,r['prediction']

def losses(head,x,y,g):
    w,o,b=head;pred=((x-o)@w+b)[None]
    return loss_by_group(pred,y,g['binary_columns'],g['continuous_columns'])

def run(output,run_directory=None,evaluation_scenes=8192):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);start=time.perf_counter()
    directory=Path(run_directory) if run_directory else ROOT/'evidence/source_smoke_run'
    model,cfg,meta=load_model(directory)
    if cfg.objective!='regression' or cfg.data.required_history!=1 or cfg.data.extra_writes!=3 or cfg.data.visible_content_dim!=cfg.data.k:
        raise ValueError('This native-task adapter requires physical_k all-present H1/C regression. Other tasks need an explicit native target/observation mapping.')
    before=parameter_hash(model);checkpoint=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest();k=cfg.data.k
    cb=fixed_batch(cfg.data,512,'geometry_fit',8400000,0);tb=fixed_batch(cfg.data,1024,'probe_train',8410000,0);sb=fixed_batch(cfg.data,2048,'probe_val',8420000,0)
    independent_ids(cb['sample_ids'],tb['sample_ids'],sb['sample_ids'],np.arange(8430000,8430000+evaluation_scenes))
    cv,_=collect(model,cb);tv,_=collect(model,tb);sv,_=collect(model,sb)
    y,theta,groups=targets(tb,k);sy,_,_=targets(sb,k);T=tb['x'].shape[1];D=cfg.width
    # One nested draw. Labels are unique scenes, never multiplied by time.
    indices=np.random.default_rng(410128).permutation(1024);budgets=(128,512)
    scales={name:max(float(np.sqrt(np.mean(np.asarray(x,float)**2))),1e-8) for name,x in cv.items()}
    bank={};selection={};spec={}
    for name,x in tv.items():
        phase,mode=name.split('__');spec[name]=permission_spec(phase,mode,T,D,y.shape[1])
        d=spec[name]['feature_dimension'];spec[name].update(
            joint_continuous_and_binary_head_parameters=d*y.shape[1]+y.shape[1],
            continuous_task_head_parameters=d*(2*k)+2*k,
            auxiliary_binary_probe_parameters=d*32+32,
            binary_labels_derived_from_same_latents_not_independent_measurements=True)
        bank[name]={};selection[name]={}
        for p in budgets:
            ids=indices[:p];heads={str(alpha):fit_head(np.asarray(x,float)[ids]/scales[name],y[ids],alpha) for alpha in ALPHAS}
            bank[name][str(p)]=heads;selection[name][str(p)]={}
            for g in groups:
                values={a:losses(h,np.asarray(sv[name],float)/scales[name],sy,g) for a,h in heads.items()}
                selection[name][str(p)][g['source']]={objective:min(heads,key=lambda a:(float(values[a][j].mean()),ALPHAS.index(float(a)))) for j,objective in enumerate(('classification','MSE'))}
        print('prepared access',name,flush=True)
    save_tree(out/'frozen_readers',bank)
    freeze={'module':'L4.1','checkpoint_sha256':checkpoint,'source_optimizer_updates':meta['optimizer_updates'],
        'reader_sha256':tree_hash(bank),'selection':selection,'scales':scales,'permission_specifications':spec,
        'calibration':[512,'geometry_fit',8400000],'training_pool':[1024,'probe_train',8410000],
        'unique_training_labels_per_head':budgets,'one_nested_training_draw':indices[:512],
        'selection_pool':[2048,'probe_val',8420000],'evaluation_pool':[evaluation_scenes,'development_test',8430000],
        'theta':theta,'groups':groups,'alphas':ALPHAS,'feature_normalization':'one calibration RMS per view, including fixed-T sum/mean scale',
        'choices_frozen_before_evaluation_generation':True,
        'constant_reader_training_means':{str(p):y[indices[:p]].mean(0) for p in budgets},
        'objectives':'Native head is H1/C regression; sign teachers are separate auxiliary classification probes, not a newly trained native classifier.'}
    write_json(out/'freeze_contract.json',freeze);del cv,tv,sv
    eb=fixed_batch(cfg.data,evaluation_scenes,'development_test',8430000,0);ev,native=collect(model,eb);ey,_,_=targets(eb,k)
    result={};scene={};constant_risks={};constant_losses={}
    for p in budgets:
        pred=np.broadcast_to(np.asarray(freeze['constant_reader_training_means'][str(p)])[None,None],(1,len(ey),ey.shape[1]))
        constant_risks[str(p)]={}
        for g in groups:
            values=loss_by_group(pred,ey,g['binary_columns'],g['continuous_columns'])
            constant_risks[str(p)][g['source']]={obj:float(values[j].mean()) for j,obj in enumerate(('classification','MSE'))}
            for j,obj in enumerate(('classification','MSE')):constant_losses[f'p{p}__{g["source"]}__{obj}']=values[j]
    for name,x in ev.items():
        x=np.asarray(x,float)/scales[name];result[name]={}
        for p in budgets:
            result[name][str(p)]={}
            for g in groups:
                row={}
                for j,objective in enumerate(('classification','MSE')):
                    alpha=selection[name][str(p)][g['source']][objective]
                    value=losses(bank[name][str(p)][alpha],x,ey,g)[j];row[objective]=float(value.mean());row[objective+'_alpha']=float(alpha)
                    scene[f'{name}__p{p}__{g["source"]}__{objective}']=value
                result[name][str(p)][g['source']]=row
    comparisons={};pairs=[('output__last','output__trajectory'),('output__last','output__mean'),('output__last','post__last')]
    family_size=len(pairs)*len(budgets)*len(groups)
    for a,b in pairs:
        for p in budgets:
            for g in groups:
                source=g['source'];key=f'{a}_vs_{b}__p{p}__{source}'
                comparisons[key]=paired_bound(scene[f'{a}__p{p}__{source}__classification'],scene[f'{b}__p{p}__{source}__classification'],family_size)
    # Same original W/b on each SAME-DIMENSION alternate observation/summary.
    # Trajectory has no compatible original head and is not silently padded.
    W=model.core.head.weight.detach().cpu().numpy();bias=model.core.head.bias.detach().cpu().numpy()
    cols=list(range(k))+list(range(16,16+k));native_rows={}
    for name,x in ev.items():
        if name.endswith('__trajectory'):continue
        pred=np.asarray(x,float)@W.T+bias;native_rows[name]={}
        for j,g in enumerate(groups):
            rows=cols[j*k:(j+1)*k];native_rows[name][g['source']]=float(np.mean((pred[:,rows]-ey[:,j*k:(j+1)*k])**2))
    replay=np.asarray(ev['post__last'],float)@W.T+bias
    maxgap=float(np.max(np.abs(replay-native)))
    nulls={}
    rng=np.random.default_rng(41024)
    for label,rows in [('all_head_outputs',list(range(len(W)))),('requested_H1_C_outputs',cols)]:
        info=output_subspaces(W[rows]);basis=info['null_basis'];delta=rng.normal(size=(128,basis.shape[1]))@basis.T
        delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-30)
        nulls[label]={'declared_output_count':len(rows),'rank':info['rank'],'null_dimension':basis.shape[1],
            'unit_feature_perturbation_head_change_max_abs':float(np.max(np.abs(delta@W[rows].T))),
            'scope':'Offline head-input perturbation only; not a cell-state or future-trajectory intervention.'}
    np.savez_compressed(out/'scene_losses.npz',**scene)
    np.savez_compressed(out/'constant_baseline_losses.npz',**constant_losses)
    assert tree_hash(bank)==freeze['reader_sha256'] and parameter_hash(model)==before
    r={'module':'L4.1 access/readout development analysis','views':len(ev),'head_matrix_solves':len(ev)*len(budgets)*len(ALPHAS),
       'risk_cells':len(ev)*len(budgets)*len(groups)*2,'risks':result,'constant_reader_risks':constant_risks,
       'constant_baseline_risk_cells':8,'native_and_same_head_counterfactual_MSE':native_rows,
       'native_actual_permission':'post__last','native_head_replay_max_abs':maxgap,'head_null_checks':nulls,
       'classification_comparisons':comparisons,'comparison_family_size':family_size,
       'positive_lower_bounds':sum(c['interval'][0]>0 for c in comparisons.values()),
       'evaluation_scenes':evaluation_scenes,'elapsed_seconds':time.perf_counter()-start,
       'checkpoint_sha256':checkpoint,'parameters_unchanged':True,'new_SNN_optimizer_updates':0,
       'formal_benchmark_completed':0,'formal_benchmark_planned':435,'audit_test_accessed':False,
       'scope':'One frozen project MLP; original checkpoint has one optimizer update. Internal pre/post and trajectory have distinct permissions/costs. Auxiliary-head gains are not evidence that the native system used the same content or that a neuron gained intrinsic memory.'}
    write_json(out/'report.json',r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--run-directory');p.add_argument('--evaluation-scenes',type=int,default=8192);a=p.parse_args()
    r=run(a.output,a.run_directory,a.evaluation_scenes);print('Access views',r['views'],'solves',r['head_matrix_solves'],'risk cells',r['risk_cells'])
