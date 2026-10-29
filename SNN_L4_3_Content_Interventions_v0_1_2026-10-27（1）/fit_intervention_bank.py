"""Freeze erasers, donor-response maps and independent validation probes."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from bridge43 import (ROOT,RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,labels23,baseline_sites,
    fit_head,linear_bundle,fit_mlp,predict,save_tree,write_json,tree_hash,parameter_hash,independent_ids)
from intervention_math43 import fit_eraser,apply_eraser,random_eraser

ALPHAS=(.0001,.001,.01)
CONTEXTS=[{'source':source,'time':t,'layer':layer,'site':f'l{layer}_t{t}','key':f'{source}_l{layer}_t{t}'}
    for source,times in [('H1',(3,12,15)),('H4',(12,15)),('C',(12,15))] for t in times for layer in (0,1)]

def choose_ridge(x,y,vx,vy):
    heads={str(a):fit_head(x,y,a) for a in ALPHAS}
    selected=min(heads,key=lambda a:(float(np.mean((predict(linear_bundle(heads[a]),vx)-vy)**2)),float(a)))
    return linear_bundle(heads[selected]),{'alpha':float(selected),'candidate_count':len(heads)}

def fit(output,steps=400):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    model,cfg,meta=load_model(RUN_DIRECTORY);before=parameter_hash(model)
    if cfg.data.k!=4 or cfg.objective!='regression' or model.run_config.cell!='lif_hard':raise ValueError('Explicit old hard-LIF H1/C adapter')
    batches={name:fixed_batch(cfg.data,n,split,first,0) for name,n,split,first in [
        ('fit',2048,'geometry_fit',9000000),('donor_fit',2048,'geometry_fit',9010000),
        ('probe',1024,'probe_train',9020000),('select',256,'probe_val',9040000)]}
    independent_ids(*(b['sample_ids'] for b in batches.values()),np.arange(9060000,9062048),np.arange(9070000,9072048))
    sites={name:baseline_sites(model,b)[0] for name,b in batches.items()};yp=labels23(batches['probe']);ys=labels23(batches['select'])
    scales={site:max(float(np.sqrt(np.mean(a*a))),1e-8) for site,a in sites['fit'].items()}
    baseline={};refit={};erasers={};delta_maps={};selection={};ridge_solves=0;mlp_fits=0;updates=0
    for site in scales:
        x,vx=sites['probe'][site]/scales[site],sites['select'][site]/scales[site]
        b,r=choose_ridge(x,yp,vx,ys);baseline[site]={'ridge':b};selection['base_'+site]={'ridge':r};ridge_solves+=3
        m,r=fit_mlp(x,yp,vx,ys,'budget_mlp',43042,steps=steps);baseline[site]['mlp']=m;selection['base_'+site]['mlp']=r
        mlp_fits+=r['optimization_fits'];updates+=r['optimizer_updates']
    donors={}
    for source in ('H1','H4','C'):
        donors[source]=baseline_sites(model,hybrid(cfg.data,batches['fit'],batches['donor_fit'],source))[0]
    for c in CONTEXTS:
        key,site,source=c['key'],c['site'],c['source'];j=SOURCES.index(source)
        h=sites['fit'][site];f=batches['fit']['latents'][:,j,:]
        binary=np.where(f[:,0:1]>=0,1.,-1.)
        erasers[key]={name:fit_eraser(h,binary if name=='binary_mp' else f,name) for name in ('leace','orth','binary_mp')}
        for seed in range(3):erasers[key]['random_'+str(seed)]=random_eraser(erasers[key]['leace'],43000+seed)
        dz=batches['donor_fit']['latents'][:,j,:]-f;dh=donors[source][site]-h
        delta_maps[key]=np.linalg.solve(dz.T@dz/len(dz)+.001*np.eye(4),dz.T@dh/len(dz))
        refit[key]={};selection[key]={}
        for name in ('leace','orth','binary_mp'):
            x=apply_eraser(sites['probe'][site],erasers[key][name])/scales[site]
            vx=apply_eraser(sites['select'][site],erasers[key][name])/scales[site]
            b,r=choose_ridge(x,yp,vx,ys);refit[key][name]={'ridge':b};selection[key][name]={'ridge':r};ridge_solves+=3
            if name=='leace':
                m,r=fit_mlp(x,yp,vx,ys,'budget_mlp',43042,steps=steps);refit[key][name]['mlp']=m;selection[key][name]['mlp']=r
                mlp_fits+=r['optimization_fits'];updates+=r['optimizer_updates']
        print('frozen intervention',key,flush=True)
    bank={'erasers':erasers,'baseline_probes':baseline,'refit_probes':refit,'delta_maps':delta_maps}
    save_tree(out/'frozen_bank',bank)
    sha=hashlib.sha256((RUN_DIRECTORY/'best.pt').read_bytes()).hexdigest()
    freeze={'module':'L4.3','checkpoint_sha256':sha,'original_optimizer_updates':meta['optimizer_updates'],'bank_sha256':tree_hash(bank),
        'contexts':CONTEXTS,'scales':scales,'selection':selection,'alphas':ALPHAS,'probe_steps_per_optimization_fit':steps,
        'fit_pool':[2048,'geometry_fit',9000000],'paired_donor_fit_pool':[2048,'geometry_fit',9010000],
        'independent_validation_probe_fit_pool':[1024,'probe_train',9020000],'independent_validation_probe_selection_pool':[256,'probe_val',9040000],
        'evaluation_recipient_pool':[2048,'development_test',9060000],'evaluation_donor_pool':[2048,'development_test',9070000],
        'probe_label_order':'20 continuous source coordinates H1,H2,H3,H4,C; 3 binary first-coordinate signs H1,H4,C',
        'ridge_matrix_solves':ridge_solves,'paired_delta_matrix_solves':len(delta_maps),'MLP_optimization_fits':mlp_fits,'MLP_optimizer_updates':updates,
        'selected_baseline_probes':sum(len(v) for v in baseline.values()),'selected_refit_probes':sum(len(w) for v in refit.values() for w in v.values()),
        'choices_frozen_before_evaluation_generation':True,'core_parameters_unchanged':parameter_hash(model)==before,
        'label_cost':'Operator fit uses 2048 scene factor labels plus 2048 matched donor labels for delta maps. Independent probes use 1024+256 scene labels across all 23 diagnostic targets; no per-dimension sample multiplication.',
        'scope':'Operations and probe families specified before evaluation. Label-informed paired delta maps are diagnostic oracle-access interventions, not free online mechanisms.'}
    assert freeze['core_parameters_unchanged'];write_json(out/'freeze_contract.json',freeze);return freeze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=400);a=p.parse_args();r=fit(a.output,a.steps)
    print('Frozen contexts',len(CONTEXTS),'ridge solves',r['ridge_matrix_solves'],'MLP updates',r['MLP_optimizer_updates'])
