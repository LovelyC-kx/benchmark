"""Extra-trial observer: pathwise and conditional-mean finite-scale geometry."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from bridge35 import load_model,repeat_panel,fixed_batch
from load_data import projection
from temporal_capture import base_capture
from temporal_theory import threshold_probability_geometry
from artifacts35 import write_json
ROOT=Path(__file__).resolve().parent

def run(run_directory=None,n=8,repeats=128):
    torch.set_num_threads(1)
    run_directory=Path(run_directory) if run_directory is not None else ROOT/'evidence/source_smoke_run'
    model,cfg,_=load_model(run_directory)
    panel=repeat_panel(cfg.data,n,repeats,'geometry_fit',7690000,0,'C','measurement_only')
    schedule=fixed_batch(cfg.data,n,'geometry_fit',7690000,0)['event_times'][0]
    T=panel['x'].shape[2];q=cfg.data.visible_content_dim
    basis=projection(cfg.data);base=panel['x'].reshape(n*repeats,*panel['x'].shape[2:]);rows=[]
    for source,onset in (('H1',int(schedule[0])),('C',int(schedule[4]))):
        for delta in (.01,.1):
            plus={};minus={};flips={tag:[] for tag in ('first','last')}
            for k in range(q):
                pair=[]
                for sign in (1,-1):
                    x=base.copy();x[:,onset,:8]+=sign*delta*cfg.data.sensory_scale*basis[:,k].astype(x.dtype)
                    pair.append(base_capture(model,x))
                for name in ('first_post','last_post'):
                    plus.setdefault(name,[]).append(pair[0][name]);minus.setdefault(name,[]).append(pair[1][name])
                for tag in flips:
                    if tag+'_spike' in pair[0]:flips[tag].append(np.mean(pair[0][tag+'_spike']!=pair[1][tag+'_spike'],axis=(0,2)))
            for name in ('first_post','last_post'):
                p=np.stack(plus[name],1).reshape(n,repeats,q,T,-1).astype(float)
                m=np.stack(minus[name],1).reshape(n,repeats,q,T,-1).astype(float)
                j=(p-m)/(2*delta);mean=j.mean(1)
                mean_metric=np.einsum('nktd,nltd->ntkl',mean,mean)
                path_metric=np.einsum('nrktd,nrltd->ntkl',j,j)/repeats
                gap=path_metric-mean_metric
                assert np.linalg.eigvalsh(gap).min()>-1e-7*max(1.,np.abs(path_metric).max())
                rows.append({'source':source,'view':name,'delta':delta,
                    'metric_of_conditional_mean':mean_metric.mean(0),
                    'mean_of_pathwise_metrics':path_metric.mean(0),
                    'Jensen_gap_min_eigenvalue':float(np.linalg.eigvalsh(gap).min()),
                    'event_crossing_fraction_by_time':np.mean(flips[name.split('_')[0]],axis=0) if flips[name.split('_')[0]] else None})
    return {'module':'L3.6 conditional-mean observer extension','scenes':n,'repeats_per_content':repeats,
            'source_first_id':7690000,'forward_captures':8*q,'finite_scale_groups':rows,
            'visible_coordinates_per_source':q,'steps':T,'checkpoint_sha256':hashlib.sha256((run_directory/'best.pt').read_bytes()).hexdigest(),
            'Gaussian_noisy_threshold_reference':threshold_probability_geometry(0,.5),
            'extra_observation_permission':f'{repeats} repeated noisy trials of each of {n} fixed contents, not {n*repeats} independent contents and not a single-trial deployed observer',
            'scope':'Development diagnostic. Finite-scale mean response and mean squared path response are different; no exact learned-SNN Fisher information estimate or new training.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--run-directory');p.add_argument('--scenes',type=int,default=8);p.add_argument('--repeats',type=int,default=128);a=p.parse_args();dst=Path(a.output)
    if dst.exists():raise FileExistsError(dst)
    write_json(dst,run(a.run_directory,a.scenes,a.repeats));print('Conditional mean/pathwise geometry extension completed')
