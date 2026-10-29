"""Post-review stability diagnosis, explicitly separate from the initial frozen family."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge43 import (ROOT,RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,labels23,baseline_sites,
    fit_mlp,predict,save_tree,load_tree,write_json,tree_hash,parameter_hash)
from fit_intervention_bank import choose_ridge
from regularized_intervention43 import fit_covariance_regularized_eraser
from intervention_math43 import apply_eraser,random_eraser,norm_matched,transport,decomposition
from runtime_interventions43 import run_post_edit
from run_state_interventions import native_losses,probe_report
from audit_author_sources import original_leace,check_sources

def fit(out,source_root):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);check_sources(source_root)
    original=original_leace(source_root);model,cfg,_=load_model(RUN_DIRECTORY);before=parameter_hash(model)
    initial=json.loads((ROOT/'evidence/fitted/freeze_contract.json').read_text())
    batches={n:fixed_batch(cfg.data,count,split,first,0) for n,count,split,first in [('fit',2048,'geometry_fit',9000000),('probe',1024,'probe_train',9020000),('select',256,'probe_val',9040000)]}
    sites={n:baseline_sites(model,b)[0] for n,b in batches.items()};yp,ys=labels23(batches['probe']),labels23(batches['select'])
    bank={'erasers':{},'refit_probes':{}};selection={};diagnostics={};solves=0;fits=0;updates=0
    old=load_tree(ROOT/'evidence/fitted/frozen_bank')
    for c in initial['contexts']:
        key,site,source=c['key'],c['site'],c['source'];j=SOURCES.index(source);h=sites['fit'][site];f=batches['fit']['latents'][:,j]
        eraser=fit_covariance_regularized_eraser(h,f,.01)
        ae=original.LeaceEraser.fit(torch.tensor(h),torch.tensor(f))
        author={'method':'author_LEACE_default','mean':ae.bias.numpy().copy(),'matrix':ae.P.numpy().T.copy(),
            'removed_rank':int(torch.linalg.matrix_rank(torch.eye(h.shape[1],dtype=torch.float64)-ae.P))}
        bank['erasers'][key]={'shrink_fixed':eraser,'rank_random':random_eraser(eraser,438043),'author_default':author}
        scale=initial['scales'][site];x=apply_eraser(sites['probe'][site],eraser)/scale;vx=apply_eraser(sites['select'][site],eraser)/scale
        ridge,r=choose_ridge(x,yp,vx,ys);mlp,m=fit_mlp(x,yp,vx,ys,'budget_mlp',43042,steps=400)
        bank['refit_probes'][key]={'ridge':ridge,'mlp':mlp};selection[key]={'ridge':r,'mlp':m};solves+=3;fits+=m['optimization_fits'];updates+=m['optimizer_updates']
        d={}
        for name,e in [('initial_unregularized',old['erasers'][key]['leace']),('shrink_fixed',eraser),('author_default',author)]:
            edited=apply_eraser(h,e);v=sites['select'][site];ve=apply_eraser(v,e)
            d[name]={'operator_norm':float(np.linalg.norm(e['matrix'],2)),'fit_state_displacement_MSE':float(np.mean((edited-h)**2)),
                'independent_selection_displacement_MSE':float(np.mean((ve-v)**2)),
                'fit_cross_covariance_after_norm':float(np.linalg.norm((edited-edited.mean(0)).T@(f-f.mean(0))/(len(f)-1))),
                'removed_rank':e['removed_rank']}
        diagnostics[key]=d
    save_tree(out/'frozen_bank',bank)
    record={'module':'L4.3 post-review stability follow-up','adaptive_reason':'Large unregularized operator amplification observed in the initial development run. Methods frozen next, before new development scenes; not preregistered confirmation of the original family.',
        'bank_sha256':tree_hash(bank),'original_bank_sha256':initial['bank_sha256'],'contexts':initial['contexts'],'selection':selection,'fit_diagnostics':diagnostics,
        'shrinkage':.01,'author_default':{'shrinkage':True,'constrain_cov_trace':True,'svd_tol':.01,'pinned_commit':'9f51753821316a1edacf78b52b464ab26d40e60a'},
        'fresh_evaluation_recipient_pool':[1024,'development_test',9100000],'fresh_evaluation_donor_pool':[1024,'development_test',9110000],
        'ridge_matrix_solves':solves,'MLP_optimization_fits':fits,'MLP_optimizer_updates':updates,'selected_refit_probes':28,
        'core_parameters_unchanged':parameter_hash(model)==before,'formal_confirmation':False}
    assert record['core_parameters_unchanged'];write_json(out/'freeze_contract.json',record);return record

def evaluate(out,fitdir):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);fitdir=Path(fitdir)
    f=json.loads((fitdir/'freeze_contract.json').read_text());bank=load_tree(fitdir/'frozen_bank');assert tree_hash(bank)==f['bank_sha256']
    initial=json.loads((ROOT/'evidence/fitted/freeze_contract.json').read_text());old=load_tree(ROOT/'evidence/fitted/frozen_bank')
    model,cfg,_=load_model(RUN_DIRECTORY);before=parameter_hash(model)
    a=fixed_batch(cfg.data,1024,'development_test',9100000,0);b=fixed_batch(cfg.data,1024,'development_test',9110000,0);av,ap=baseline_sites(model,a)
    donors={s:hybrid(cfg.data,a,b,s) for s in ('H1','H4','C')};neutral={s:hybrid(cfg.data,a,b,s,zero=True) for s in donors}
    dv={};nv={};dp={};npred={}
    for s in donors:dv[s],dp[s]=baseline_sites(model,donors[s]);nv[s],npred[s]=baseline_sites(model,neutral[s])
    rows={};losses={};columns=list(range(4))+list(range(16,20))
    for c in f['contexts']:
        key,site,source,time,layer=(c[k] for k in ('key','site','source','time','layer'));h=av[site];erasers=bank['erasers'][key];scale=initial['scales'][site];path=f'backbone.cells.{layer}'
        locald=run_post_edit(model,a['x'],path,time,lambda raw,start:dv[source][site][start:start+len(raw)],keep_traces=False)
        localz=run_post_edit(model,a['x'],path,time,lambda raw,start:nv[source][site][start:start+len(raw)],keep_traces=False)
        operations={'no_op':h,'shrink_fixed':apply_eraser(h,erasers['shrink_fixed']),'author_default':apply_eraser(h,erasers['author_default']),
            'rank_random':apply_eraser(h,erasers['rank_random']),'norm_random':norm_matched(h,erasers['shrink_fixed'],438044),
            'shrink_transport':transport(h,dv[source][site],erasers['shrink_fixed']),'donor_v':dv[source][site],'oracle_zero_v':nv[source][site]}
        baseline={k:predict(p,h/scale) for k,p in old['baseline_probes'][site].items()};rkey={}
        for name,edited in operations.items():
            r=locald if name=='donor_v' else localz if name=='oracle_zero_v' else run_post_edit(model,a['x'],path,time,lambda raw,start:edited[start:start+len(raw)],keep_traces=False)
            lo=native_losses(r['prediction'],a);lh=native_losses(r['prediction'],donors[source]);transfer=name in ('shrink_transport','donor_v')
            natural=dp[source] if transfer else npred[source];local=locald if transfer else localz;target=donors[source] if transfer else neutral[source]
            row={'native_risks':{s:{o:float(v.mean()) for o,v in z.items()} for s,z in lo.items()},'native_hybrid_label_risks':{s:{o:float(v.mean()) for o,v in z.items()} for s,z in lh.items()},
                'state_displacement_MSE':float(np.mean((r['site_after']-h)**2)),'operation_application_max_abs':float(np.max(np.abs(r['site_after']-edited))),
                'frozen_validation_probes':{k:probe_report(predict(p,r['site_after']/scale),a,donors[source],source,time,baseline[k]) for k,p in old['baseline_probes'][site].items()},
                'operator_and_scope':decomposition(r['prediction'][:,columns],natural[:,columns],local['prediction'][:,columns],target['latents'][:,[0,4]].reshape(len(h),8))}
            if name=='shrink_fixed':row['refitted_after_erasure_probes']={k:probe_report(predict(p,r['site_after']/scale),a,donors[source],source,time,baseline[k]) for k,p in bank['refit_probes'][key].items()}
            if name=='no_op':row['native_noop_max_abs']=float(np.max(np.abs(r['prediction']-ap)))
            if name=='norm_random':row['norm_match_max_abs']=float(np.max(np.abs(np.linalg.norm(edited-h,axis=1)-np.linalg.norm(operations['shrink_fixed']-h,axis=1))))
            rkey[name]=row
            for s,z in lo.items():
                for o,v in z.items():losses[key+'::'+name+'::'+s+'::'+o]=v
        rows[key]=rkey;print('evaluated stability context',key,flush=True)
    np.savez_compressed(out/'scene_losses.npz',**losses)
    assert parameter_hash(model)==before and tree_hash(bank)==f['bank_sha256']
    record={'module':'L4.3 adaptive stability diagnosis','evaluation_pairs':1024,'contexts':14,'native_intervention_conditions':sum(len(v) for v in rows.values()),'results':rows,
        'saved_loss_arrays':len(losses),'core_parameters_unchanged':True,'audit_test_accessed':False,'new_SNN_training':0,'formal_benchmark_completed':0,
        'scope':'Fresh development scenes after adaptation; exploratory stability comparison, not a formal confirmation family or independent model replication. Same old one-update checkpoint.'}
    write_json(out/'report.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('fit','evaluate'));p.add_argument('--output',required=True);p.add_argument('--source-root');p.add_argument('--fit-directory');a=p.parse_args()
    if a.mode=='fit':r=fit(a.output,a.source_root)
    else:r=evaluate(a.output,a.fit_directory)
    print('completed',a.mode)
