"""Native head, same-access refits and restricted nonlinear reader controls."""
import argparse, hashlib, time
from pathlib import Path
import numpy as np
import torch
from bridge42 import (ROOT, PARENT, load_model, fixed_batch, collect, targets,
    parameter_hash, independent_ids, fit_head, save_tree, write_json, tree_hash, paired_bound)
from readers42 import predict, predict_linear, linear_bundle, fit_mlp, ridge_optimization_control

VIEWS=('output__last','output__mean','output__trajectory','post__last','post__mean','post__trajectory')
BUDGETS=(128,512)
ALPHAS=(.0001,.001,.01)
DRAW_SEEDS=(420128,420129)

def group_losses(pred,y,theta,k):
    out={}
    for j,name in enumerate(('H1','C')):
        a,b=pred[:,j*k:(j+1)*k],y[:,j*k:(j+1)*k]
        sign_a,sign_b=a@theta[j],b@theta[j]
        out[name]={'MSE':((a-b)**2).mean(1),
            'derived_classification':np.where(sign_a==0,.5,(np.sign(sign_a)!=np.sign(sign_b)).astype(float)).mean(1)}
    return out

def fit(output,steps=400,evaluation_scenes=8192):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);start=time.perf_counter()
    directory=PARENT/'evidence/source_smoke_run'
    model,cfg,meta=load_model(directory);before=parameter_hash(model)
    if cfg.objective!='regression' or cfg.data.k!=4 or cfg.data.required_history!=1 or cfg.data.extra_writes!=3:
        raise ValueError('Only the declared all-present k=4 H1/C native regression adapter is implemented')
    checkpoint=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest();k=cfg.data.k
    cb=fixed_batch(cfg.data,512,'geometry_fit',8500000,0)
    tb=fixed_batch(cfg.data,1024,'probe_train',8510000,0)
    sb=fixed_batch(cfg.data,256,'probe_val',8520000,0)
    # Independent diagnostic reader, reserved for later intervention validation.
    vb=fixed_batch(cfg.data,512,'probe_train',8530000,0)
    vsb=fixed_batch(cfg.data,128,'probe_val',8531000,0)
    independent_ids(cb['sample_ids'],tb['sample_ids'],sb['sample_ids'],vb['sample_ids'],vsb['sample_ids'],np.arange(8540000,8540000+evaluation_scenes))
    cv,_=collect(model,cb);tv,_=collect(model,tb);sv,_=collect(model,sb)
    vv,_=collect(model,vb);vsv,_=collect(model,vsb)
    y,theta,_=targets(tb,k);y=y[:,:2*k]
    sy=targets(sb,k)[0][:,:2*k];vy=targets(vb,k)[0][:,:2*k];vsy=targets(vsb,k)[0][:,:2*k]
    scale={name:max(float(np.sqrt(np.mean(cv[name]**2))),1e-8) for name in VIEWS}
    bank={};cases=[];selection={};opt={};solves=0;mlp_fits=0;mlp_updates=0
    W=model.core.head.weight.detach().cpu().numpy();bias=model.core.head.bias.detach().cpu().numpy()
    native_cols=list(range(k))+list(range(16,16+k))
    for name in VIEWS:
        tx=np.asarray(tv[name],float)/scale[name];sx=np.asarray(sv[name],float)/scale[name]
        for draw,seed in enumerate(DRAW_SEEDS):
            rng=np.random.default_rng(seed);idx=rng.permutation(len(y));val_idx=rng.permutation(len(sy))
            for p in BUDGETS:
                ids=idx[:p];sel=val_idx[:p//4];a,b=tx[ids],y[ids];c,d=sx[sel],sy[sel]
                key=f'{name}__draw{draw}__p{p}'
                heads={str(alpha):fit_head(a,b,alpha) for alpha in ALPHAS};solves+=len(heads)
                alpha=min(heads,key=lambda z:(float(np.mean((predict_linear(heads[z],c)-d)**2)),float(z)))
                chosen=heads[alpha];selection[key]={'alpha':float(alpha)}
                for kind in ('ridge','budget_mlp','residual'):
                    case=key+'__'+kind
                    if kind=='ridge':bundle=linear_bundle(chosen)
                    else:
                        bundle,receipt=fit_mlp(a,b,c,d,kind,seed+42,chosen if kind=='residual' else None,steps)
                        mlp_fits+=receipt['optimization_fits'];mlp_updates+=receipt['optimizer_updates']
                        selection[key][kind]=receipt
                    bank[case]=bundle
                    cases.append({'key':case,'view':name,'draw':draw,'fit_labels':p,'selection_labels':len(sel),
                        'total_head_supervised_labels':p+len(sel),'kind':kind,'parameters':bundle['parameters'],
                        'input_dimension':tx.shape[1],'fit_indices':ids,'selection_indices':sel})
                shuffle=np.random.default_rng(seed+50000+p)
                sh_y=b[shuffle.permutation(len(b))];sh_d=d[shuffle.permutation(len(d))]
                sh_heads={str(alpha):fit_head(a,sh_y,alpha) for alpha in ALPHAS};solves+=len(sh_heads)
                sh_alpha=min(sh_heads,key=lambda z:(float(np.mean((predict_linear(sh_heads[z],c)-sh_d)**2)),float(z)))
                case=key+'__shuffled_ridge';bundle=linear_bundle(sh_heads[sh_alpha]);bank[case]=bundle
                cases.append({'key':case,'view':name,'draw':draw,'fit_labels':p,'selection_labels':len(sel),
                    'total_head_supervised_labels':p+len(sel),'kind':'shuffled_ridge','parameters':bundle['parameters'],
                    'input_dimension':tx.shape[1],'fit_indices':ids,'selection_indices':sel})
                selection[key]['shuffled_alpha']=float(sh_alpha)
                if name=='post__last':
                    opt[key]=ridge_optimization_control(a,b,float(alpha),(W[native_cols]*scale[name]).T,bias[native_cols])
                print('frozen reader case',key,flush=True)
    # Independently fitted verification probes are not used to choose the main reader.
    verification={}
    for name in ('output__last','post__last'):
        a=np.asarray(vv[name],float)/scale[name];c=np.asarray(vsv[name],float)/scale[name]
        hs={str(alpha):fit_head(a,vy,alpha) for alpha in ALPHAS};solves+=len(hs)
        alpha=min(hs,key=lambda z:float(np.mean((predict_linear(hs[z],c)-vsy)**2)))
        verification[name]=linear_bundle(hs[alpha]);verification[name]['alpha']=float(alpha)
    save_tree(out/'frozen_readers',bank);save_tree(out/'independent_verification_readers',verification)
    freeze={'module':'L4.2','checkpoint_sha256':checkpoint,'reader_sha256':tree_hash(bank),'verification_sha256':tree_hash(verification),
        'original_optimizer_updates':meta['optimizer_updates'],'views':VIEWS,'scales':scale,'cases':cases,'selection':selection,
        'calibration':[512,'geometry_fit',8500000],'fit_pool':[1024,'probe_train',8510000],
        'selection_pool':[256,'probe_val',8520000],'verification_fit_pool':[512,'probe_train',8530000],
        'verification_selection_pool':[128,'probe_val',8531000],'evaluation':[evaluation_scenes,'development_test',8540000],
        'head_supervised_budgets':{str(p):{'fit':p,'select':p//4,'total':p+p//4} for p in BUDGETS},
        'independent_verification_supervision':{'fit':512,'select':128,'total':640,'shared_by_two_views':True},
        'scales_use_unlabeled_calibration':True,'theta':theta,'k':k,'alphas':ALPHAS,'steps_per_MLP_fit':steps,
        'main_objective':'Original eight H1/C continuous targets; all selection uses continuous MSE.',
        'classification':'Fixed sign projections of predicted continuous latents; auxiliary derived task, not a native trained classifier.',
        'choices_frozen_before_evaluation_generation':True}
    write_json(out/'freeze_contract.json',freeze);write_json(out/'optimization_control.json',opt)
    del cv,tv,sv,vv,vsv,cb,tb,sb,vb,vsb
    eb=fixed_batch(cfg.data,evaluation_scenes,'development_test',8540000,0);ev,native=collect(model,eb);ey=targets(eb,k)[0][:,:2*k]
    native_pred=np.asarray(native)[:,native_cols];native_loss=group_losses(native_pred,ey,theta,k)
    scene={};risks={};verification_risks={};constants={}
    for case in cases:
        key,name=case['key'],case['view'];pred=predict(bank[key],np.asarray(ev[name],float)/scale[name]);lo=group_losses(pred,ey,theta,k)
        risks[key]={}
        for source,row in lo.items():
            risks[key][source]={obj:float(a.mean()) for obj,a in row.items()}
            for obj,a in row.items():scene[key+'::'+source+'::'+obj]=a
        if case['kind']=='ridge':
            head=bank[key]['head'];co=np.broadcast_to(head[2],ey.shape);cl=group_losses(co,ey,theta,k)
            constants[key]={s:{o:float(a.mean()) for o,a in row.items()} for s,row in cl.items()}
    for name,bundle in verification.items():
        lo=group_losses(predict(bundle,np.asarray(ev[name],float)/scale[name]),ey,theta,k)
        verification_risks[name]={s:{o:float(a.mean()) for o,a in row.items()} for s,row in lo.items()}
    comparisons={};comparison_pairs=[]
    for draw in range(len(DRAW_SEEDS)):
        for p in BUDGETS:
            for name in VIEWS:
                base=f'{name}__draw{draw}__p{p}'
                for other in ('budget_mlp','residual'):
                    for source in ('H1','C'):
                        comparison_pairs.append((base+'__ridge',base+'__'+other,source))
    family=len(comparison_pairs)
    for a,b,source in comparison_pairs:
        comparisons[a+'_vs_'+b+'::'+source]=paired_bound(scene[a+'::'+source+'::derived_classification'],scene[b+'::'+source+'::derived_classification'],family)
    replay=np.asarray(ev['post__last'],float)@W[native_cols].T+bias[native_cols]
    assert np.max(np.abs(replay-native_pred))<1e-5
    assert parameter_hash(model)==before and tree_hash(bank)==freeze['reader_sha256']
    np.savez_compressed(out/'scene_losses.npz',**scene)
    report={'module':'L4.2 development reader ladder','checkpoint_sha256':checkpoint,'original_optimizer_updates':meta['optimizer_updates'],
        'view_count':len(VIEWS),'draws':len(DRAW_SEEDS),'head_fit_budgets':BUDGETS,'selected_main_readers':len(cases),
        'ridge_matrix_solves_including_shuffle_and_verification':solves,'MLP_optimization_fits':mlp_fits,'MLP_optimizer_updates':mlp_updates,
        'risk_cells':len(scene),'evaluation_scenes':evaluation_scenes,'risks':risks,'constant_baselines':constants,
        'native_risks':{s:{o:float(a.mean()) for o,a in row.items()} for s,row in native_loss.items()},
        'native_replay_max_abs':float(np.max(np.abs(replay-native_pred))),
        'independent_verification_reader_risks':verification_risks,'classification_comparison_family':family,
        'classification_comparisons':comparisons,'positive_classification_lower_bounds':sum(c['interval'][0]>0 for c in comparisons.values()),
        'core_parameters_unchanged':True,'new_SNN_training':0,'formal_benchmark_completed':0,'formal_benchmark_planned':435,
        'audit_test_accessed':False,'real_datasets_downloaded':False,'elapsed_seconds':time.perf_counter()-start,
        'scope':'One old one-update model, frozen core. Two reader draws are not two SNN training seeds. No native content intervention or full-network optimization adequacy claim.'}
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=400);p.add_argument('--evaluation-scenes',type=int,default=8192)
    a=p.parse_args();r=fit(a.output,a.steps,a.evaluation_scenes);print('Readers',r['selected_main_readers'],'risk cells',r['risk_cells'])
