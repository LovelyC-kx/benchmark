"""Frozen paired maps: offset-aware cycles and equal-lag transfer on new scenes."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import torch
from bridge35 import load_model,fixed_batch
from temporal_capture import base_capture,temporal_views
from temporal_alignment import fit_transport,transport,cycle_diagnostics,affine_parts
from temporal_probes import tree_hash
from artifacts35 import save_tree,write_json
ROOT=Path(__file__).resolve().parent

def known_consistency():
    rng=np.random.default_rng(360301);z=rng.normal(size=(1024,2));hold=rng.normal(size=(1024,2))
    def rot(a):return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    result={}
    for label,angles in [('constant_speed',[0.,.3,.6]),('changing_speed',[0.,.3,1.2])]:
        x=[z@rot(a)+[2*a,-a] for a in angles]
        m01=fit_transport(x[0],x[1]);m12=fit_transport(x[1],x[2]);m02=fit_transport(x[0],x[2])
        result[label]={'cycle':cycle_diagnostics(m01,m12,m02,hold),
            'equal_lag_matrix_difference':float(np.linalg.norm(m01['w']-m12['w']))}
    return result

def run(primary,out,run_directory=None,scenes=2048):
    primary,out=Path(primary),Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    lock=json.loads((primary/'freeze_contract.json').read_text())
    directory=Path(run_directory) if run_directory else ROOT/'evidence/source_smoke_run'
    if hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()!=lock['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from primary freeze')
    with np.load(primary/'paired_calibration.npz',allow_pickle=False) as data:
        cal=temporal_views({k:data[k].copy() for k in data})
    T=next(iter(cal.values())).shape[1]
    times=sorted(set(t for t in (2,5,8,12,13,T-1) if t<T))
    families=('center','orthogonal','affine');maps={}
    # All directions needed for cycles and equal-lag (12->13 vs 13->14/14->15).
    pairs=set((c,a) for a,b,c in itertools.combinations(times,3))
    pairs.update((b,a) for a,b,c in itertools.combinations(times,3))
    pairs.update((c,b) for a,b,c in itertools.combinations(times,3))
    pairs.update((t,t+1) for t in range(12,T-1))
    for name,x in cal.items():
        x=np.asarray(x,float)/lock['scales'][name];maps[name]={}
        for a,b in sorted(pairs):maps[name][f'{a}:{b}']={f:fit_transport(x[:,a],x[:,b],f) for f in families}
    save_tree(out/'frozen_maps',maps)
    freeze={'checkpoint_sha256':lock['checkpoint_sha256'],'map_sha256':tree_hash(maps),
            'times':times,'families':families,'paired_calibration':lock['calibration_pool'],
            'evaluation':[scenes,'development_test',7700000],
            'map_direction':'a:b maps observation at a to observation at b',
            'frozen_before_evaluation_generation':True,'labels_used_for_maps':False}
    write_json(out/'freeze_contract.json',freeze)
    model,cfg,_=load_model(directory);batch=fixed_batch(cfg.data,scenes,'development_test',7700000,0)
    ev=temporal_views(base_capture(model,batch['x']));cycles=[];translations=[]
    for name,x in ev.items():
        x=np.asarray(x,float)/lock['scales'][name];ms=maps[name]
        for a,b,c in itertools.combinations(times,3):
            for family in families:
                first,second,direct=ms[f'{c}:{b}'][family],ms[f'{b}:{a}'][family],ms[f'{c}:{a}'][family]
                row=cycle_diagnostics(first,second,direct,x[:,c])
                row.update(view=name,family=family,times=[c,b,a],
                    minimum_cross_rank=min(m['rank_cross_covariance'] for m in (first,second,direct)),
                    dimension=x.shape[-1],all_full_rank=all(m['orthogonal_identifiable_full_rank'] for m in (first,second,direct)))
                cycles.append(row)
        for t in range(13,T-1):
            for family in families:
                fixed,local=ms['12:13'][family],ms[f'{t}:{t+1}'][family]
                fw,fb=affine_parts(fixed);lw,lb=affine_parts(local)
                old=transport(fixed,x[:,t]);new=transport(local,x[:,t]);target=x[:,t+1]
                translations.append({'view':name,'family':family,'lag':1,'reference_pair':[12,13],
                    'evaluation_pair':[t,t+1],'matrix_drift':float(np.linalg.norm(fw-lw,'fro')/np.sqrt(len(fw))),
                    'offset_drift':float(np.linalg.norm(fb-lb)/np.sqrt(len(fb))),
                    'fixed_map_prediction_MSE':float(np.mean((old-target)**2)),
                    'local_map_prediction_MSE':float(np.mean((new-target)**2)),
                    'heldout_excess_prediction_MSE':float(np.mean((old-target)**2-(new-target)**2)),
                    'reference_cross_rank':fixed['rank_cross_covariance'],'local_cross_rank':local['rank_cross_covariance'],
                    'dimension':x.shape[-1]})
    assert tree_hash(maps)==freeze['map_sha256']
    r={'module':'L3.6 offset-aware cycle and time-translation closure','known_controls':known_consistency(),
       'cycles':cycles,'equal_lag_transfers':translations,'independent_evaluation_scenes':scenes,
       'checkpoint_sha256':lock['checkpoint_sha256'],'new_SNN_optimizer_updates':0,
       'interpretation':'Cycle agreement is not time-translation invariance. Matrix differences in rank-deficient unused directions do not identify a physical rotation; report heldout prediction and offsets too. Empty equal-lag set means fewer than 15 steps, not evidence of invariance.',
       'scope':'One frozen checkpoint; descriptive frozen-map diagnostics, no trained-model or causal mechanism confirmation.'}
    write_json(out/'report.json',r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('primary');p.add_argument('--output',required=True);p.add_argument('--run-directory');p.add_argument('--scenes',type=int,default=2048)
    a=p.parse_args();r=run(a.primary,a.output,a.run_directory,a.scenes);print('Cycle rows',len(r['cycles']),'equal-lag rows',len(r['equal_lag_transfers']))
