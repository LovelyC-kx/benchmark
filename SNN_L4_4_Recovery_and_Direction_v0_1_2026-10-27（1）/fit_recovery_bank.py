"""Freeze all recovery contexts, reference permissions and independent probes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from bridge44 import (ROOT,PARENT,RUN_DIRECTORY,TIMES,SOURCES,load_model,fixed_batch,labels23,fit_mlp,
    choose_ridge,fit_covariance_regularized_eraser,save_tree,load_tree,tree_hash,write_json,
    parameter_hash,independent_ids,original_contract,recovery_contexts,site_name,context_name)
from recovery_runtime44 import scheduled_run,trace_site

METHODS=('full_self_delta','target_self_delta','target_reverse_delta','wrong_F_delta',
    'right_F_other_context_delta','unmatched_donor_delta','wrong_time_delta',
    'norm_random_0','norm_random_1','norm_random_2')
POOLS={'development':{'n':1024,'recipient':9200000,'donor':9210000},
       'confirmation':{'n':2048,'recipient':9220000,'donor':9230000}}

def fit(output,steps=400):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    model,cfg,meta=load_model(RUN_DIRECTORY);before=parameter_hash(model)
    old=original_contract();oldstable=load_tree(PARENT/'evidence/stability_fitted/frozen_bank')
    batches={name:fixed_batch(cfg.data,n,split,first,0) for name,n,split,first in [
        ('fit',2048,'geometry_fit',9120000),('probe',1024,'probe_train',9160000),('select',256,'probe_val',9170000)]}
    independent_ids(*(v['sample_ids'] for v in batches.values()),*[np.arange(p[k],p[k]+p['n']) for p in POOLS.values() for k in ('recipient','donor')])
    runs={name:scheduled_run(model,b['x']) for name,b in batches.items()}
    states={name:{site_name(layer,t):trace_site(r,layer,t) for layer in (0,1) for t in TIMES} for name,r in runs.items()}
    scales={s:max(float(np.sqrt(np.mean(h*h))),1e-8) for s,h in states['fit'].items()}
    bank={'damage_erasers':{},'patch_erasers':{},'validation_probes':{}};selection={};ridge_solves=0;mlp_fits=0;updates=0
    for c in old['contexts']:
        bank['damage_erasers'][c['key']]=oldstable['erasers'][c['key']]['shrink_fixed']
    contexts=recovery_contexts();unique={c['patch_key']:(c['source'],c['layer'],c['patch_time']) for c in contexts}
    new_operator_fits=0
    for key,(source,layer,t) in unique.items():
        if key in bank['damage_erasers']:e=bank['damage_erasers'][key]
        else:
            e=fit_covariance_regularized_eraser(states['fit'][site_name(layer,t)],batches['fit']['latents'][:,SOURCES.index(source)],.01);new_operator_fits+=1
        bank['patch_erasers'][key]=e
    yp,ys=labels23(batches['probe']),labels23(batches['select'])
    for site in scales:
        x,vx=states['probe'][site]/scales[site],states['select'][site]/scales[site]
        ridge,r=choose_ridge(x,yp,vx,ys);mlp,m=fit_mlp(x,yp,vx,ys,'budget_mlp',44044,steps=steps)
        bank['validation_probes'][site]={'ridge':ridge,'mlp':mlp};selection[site]={'ridge':r,'mlp':m}
        ridge_solves+=3;mlp_fits+=m['optimization_fits'];updates+=m['optimizer_updates'];print('frozen independent recovery probes',site,flush=True)
    save_tree(out/'frozen_bank',bank)
    # 38 contexts x 10 methods x 2 queries x 2 bounded metrics x 3 patch contrasts,
    # plus 14 damage contexts x 2 queries x 2 bounded metrics.
    family=len(contexts)*len(METHODS)*2*2*3+len(old['contexts'])*2*2
    record={'module':'L4.4','checkpoint_sha256':hashlib.sha256((RUN_DIRECTORY/'best.pt').read_bytes()).hexdigest(),
        'original_optimizer_updates':meta['optimizer_updates'],'bank_sha256':tree_hash(bank),'parent_stable_bank_sha256':tree_hash(oldstable),
        'damage_contexts':old['contexts'],'recovery_contexts':contexts,'methods':METHODS,'evaluation_pools':POOLS,
        'fit_pool':[2048,'geometry_fit',9120000],'independent_validation_fit_pool':[1024,'probe_train',9160000],
        'independent_validation_selection_pool':[256,'probe_val',9170000],'scales':scales,'selection':selection,
        'inherited_frozen_damage_erasers':14,'patch_erasers':len(unique),'new_regularized_operator_fits':new_operator_fits,
        'selected_validation_probes':24,'ridge_candidate_solves':ridge_solves,'MLP_optimization_fits':mlp_fits,'MLP_optimizer_updates':updates,
        'damage_ratio_floor_MSE':.01,'prediction_fidelity_floor':1e-8,'capped_MSE_scale':4.,
        'bounded_comparison_family_per_pool':family,'family_delta':.05,
        'four_groups':'Clean, damaged, clean plus delta, damaged plus exactly the same external delta. Delta is computed from the unpatched damaged reference before either patch run.',
        'patch_permissions':'All clean/donor trajectory patches are external diagnostic information. Wrong-time control may use the next time step and is explicitly future-oracle. No autonomous online-repair claim.',
        'supplementary_protocol':'Natural vs clean-event replay for 12 tau12/p13-or14 contexts and target/reverse/random0; two-owner factorial full/target patches; binary edge recovery; complete and scoped snapshot controls.',
        'core_parameters_unchanged':parameter_hash(model)==before,'choices_frozen_before_evaluation_generation':True,
        'new_SNN_training':0,'formal_benchmark_completed':0,'audit_test_accessed':False,
        'scope':'Two disjoint scene cohorts of the same one-update old model; confirmation is a scene replication, not independently trained architecture confirmation.'}
    assert record['core_parameters_unchanged'];write_json(out/'freeze_contract.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=400);a=p.parse_args()
    r=fit(a.output,a.steps);print('Frozen recovery contexts',len(r['recovery_contexts']),'family',r['bounded_comparison_family_per_pool'])
