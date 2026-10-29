"""Freeze decoder and independent semantic-probe selection before evaluation generation."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from bridge45 import ROOT,CONFIGS,VIEWS,model_for,config_key,site_paths,parameter_hash,save_tree,tree_hash,dump
from spatial_data45 import batch,labels,input_hash
from spatial_runtime45 import run,state
from spatial_readers45 import select_decoder,aggregate,ridge_fit,ridge_predict

POOLS={'train':(512,45000000),'val':(128,45010000),'probe_train':(256,45020000),'probe_val':(128,45030000),
       'development':(256,45040000),'development_donor':(256,45050000),'confirmation':(512,45060000),'confirmation_donor':(512,45070000)}
METHODS=('owner_self','region_self','region_reverse','wrong_H','wrong_time','random_region')

def fit(output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);start=time.monotonic();bank={};selections={};hashes={};solves=0
    data={k:batch(n,first) for k,(n,first) in POOLS.items() if k in ('train','val','probe_train','probe_val')}
    ranges=[set(range(first,first+n)) for n,first in POOLS.values()]
    assert all(not a&b for i,a in enumerate(ranges) for b in ranges[i+1:])
    for backbone,neuron in CONFIGS:
        key=config_key(backbone,neuron);model=model_for(backbone,neuron);before=parameter_hash(model);paths=site_paths(model)
        runs={k:run(model,v['x'],sites=paths) for k,v in data.items()};item={'decoders':{},'probes':{},'class_probes':{}};sel={}
        for view in VIEWS:
            tr=aggregate(runs['train']['features'],view);va=aggregate(runs['val']['features'],view);item['decoders'][view]={};sel[view]={}
            for kind in ('spatial1','spatial3','gap_dense'):
                m,c=select_decoder(tr,data['train']['y'],va,data['val']['y'],kind)
                item['decoders'][view][kind]=m;sel[view][kind]=c;solves+=len(c)
            target=np.sign(data['train']['latents'][:,:,0]);item['class_probes'][view]=ridge_fit(tr.mean((2,3)),target,.01);solves+=1
        for path in paths:
            item['probes'][path]={}
            for t in (2,3,5):
                x=state(runs['probe_train'],path,t).reshape(len(data['probe_train']['x']),-1)
                xv=state(runs['probe_val'],path,t).reshape(len(data['probe_val']['x']),-1)
                y=labels(data['probe_train']);yv=labels(data['probe_val']);candidates=[]
                for alpha in (.001,.01,.1):
                    p=ridge_fit(x,y,alpha);risk=float(np.mean((ridge_predict(p,xv)-yv)**2));candidates.append((risk,p,alpha));solves+=1
                best=min(candidates,key=lambda v:v[0]);item['probes'][path][str(t)]=best[1]
                sel[path+'::'+str(t)]={'alpha':best[2],'candidate_validation_MSE':[{'alpha':v[2],'MSE':v[0]} for v in candidates]}
        assert parameter_hash(model)==before;bank[key]=item;hashes[key]=before;selections[key]=sel
        print('frozen',key,'elapsed',round(time.monotonic()-start,2),flush=True)
    save_tree(out/'bank',bank)
    contract={'module':'L4.5','generator':'spatial_history45_v1','initialization_seed':45001,'model_configs':[list(v) for v in CONFIGS],
        'core_training_updates':0,'native_random_heads_unchanged':True,'fitted_readouts_are_separate_adapters':True,
        'parameter_hashes':hashes,'bank_sha256':tree_hash(bank),'pools':{k:{'n':n,'first_id':first} for k,(n,first) in POOLS.items()},
        'input_hashes_fit_pools':{k:input_hash(v['x']) for k,v in data.items()},'views':list(VIEWS),'methods':list(METHODS),
        'damage_time':2,'patch_times':[3,5],'selected_decoders':36,'selected_independent_probes':36,'classification_auxiliary_probes':12,
        'ridge_candidate_solves_total':solves,'selection':selections,
        'primary_task':'Four-band synthetic image reconstruction at t5; core is untrained, decoders fitted offline.',
        'intervention_decoder':'spatial3, fixed before both evaluation cohorts; both last/mean access reported',
        'damage':'post-v in H target footprint replaced by neutral-H reference at t2; actual current emit not recomputed',
        'four_groups':'Same frozen external additive payload on clean and damaged backgrounds; original recipient label fixed',
        'bounded_family_per_pool':1728,'family_delta':.05,'MSE_damage_floor':.0001,
        'supplements':'Confirmation first128: source/time map, stale source, alignment, noise coupling, area/frequency accounting. Texture OOD uses 256 new scenes and changed basis.',
        'choices_frozen_before_evaluation':True,'formal_benchmark_completed':0,'audit_test_accessed':False,
        'fit_elapsed_seconds':time.monotonic()-start}
    assert solves==228;dump(out/'freeze_contract.json',contract);return contract

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=fit(a.output);print('Ridge candidate solves',r['ridge_candidate_solves_total'])
