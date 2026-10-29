"""Explicitly distinct threshold, measurement-noise and observer-precision panels."""
import argparse,hashlib,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from bridge35 import load_model,fixed_batch,flatten_contents,Cell
from bridge import _render
from temporal_capture import base_capture
from temporal_probes import loss_by_group,score,tree_hash
from artifacts35 import load_tree,write_json
from run_temporal_review import target_bank
ROOT=Path(__file__).resolve().parent

def quantize_observation(x,step,limit=None):
    if step is None:return np.asarray(x,float)
    if step<=0:raise ValueError('Fixed positive calibration-derived quantization step')
    x=np.asarray(x,float)
    if limit is None:return np.round(x/step)*step
    if limit<=0:raise ValueError('Positive fixed calibration range')
    levels=round(2*limit/step)+1
    return np.clip(np.round((x+limit)/step),0,levels-1)*step-limit

def run(primary,out,run_directory=None,scenes=128):
    primary,out=Path(primary),Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    lock=json.loads((primary/'freeze_contract.json').read_text())
    directory=Path(run_directory) if run_directory else ROOT/'evidence/source_smoke_run'
    sha=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()
    if sha!=lock['checkpoint_sha256']:raise ValueError('Checkpoint differs from primary')
    model,cfg,_=load_model(directory);bank=load_tree(primary/'frozen_readers')
    if tree_hash(bank)!=lock['reader_tree_sha256']:raise ValueError('Reader bank changed')
    cells=[m for m in model.modules() if isinstance(m,Cell)];thresholds=[c.threshold for c in cells]
    grid={'threshold_multipliers':[.8,1.,1.2],'measurement_sigma':[0.,cfg.data.sigma,.5],
          'content_delta':[.001,.01,.1,.3],'observer_bits':[None,2,4,8]}
    # Remove duplicate sigma values for models with sigma=0 or .5.
    grid['measurement_sigma']=list(dict.fromkeys(grid['measurement_sigma']))
    write_json(out/'freeze_contract.json',{'checkpoint_sha256':sha,'primary_reader_sha256':tree_hash(bank),
        'grid':grid,'evaluation':[scenes,'development_test',7740000],
        'quantizer':'Uniform finite-level observer quantizer; range +/-4*primary calibration RMS, 2**bits levels, clipping included. No model arithmetic quantization.',
        'scope':'Predeclared descriptive matched-scene robustness panel. Threshold changes are new frozen inference variants, not newly trained models.'})
    rows=[];forward=0
    try:
        for ratio in grid['threshold_multipliers']:
            for c,v in zip(cells,thresholds):c.threshold=v*ratio
            for sigma in grid['measurement_sigma']:
                data=replace(cfg.data,sigma=sigma);batch=fixed_batch(data,scenes,'development_test',7740000,0)
                base=base_capture(model,batch['x']);forward+=1
                y,_,groups=target_bank(flatten_contents(batch,data));T=batch['x'].shape[1];anchor=T-1
                for j in (0,4):
                    g=groups[j];onset=int(batch['event_times'][0,j])
                    for delta in grid['content_delta']:
                        pairs=[];flips=[]
                        for coordinate in range(data.visible_content_dim):
                            branches=[]
                            for sign in (1,-1):
                                z=batch['latents'].copy();z[:,j,coordinate]+=sign*delta
                                x=_render(data,z,batch['measurement_noise'],batch['event_times'],batch['source_present'])[:,:T]
                                branches.append(base_capture(model,x));forward+=1
                            pairs.append(branches)
                        for name in ('first_post','last_post'):
                            tag=name.split('_')[0];f=bank[name]
                            w=f['weights'][:,anchor];bias=f['bias'][:,anchor]-np.einsum('rd,rdo->ro',f['origins'][:,anchor],w)
                            for bits in grid['observer_bits']:
                                step=None if bits is None else 8*lock['scales'][name]/(2**bits-1)
                                limit=None if bits is None else 4*lock['scales'][name]
                                jac=[]
                                for plus,minus in pairs:
                                    jac.append((quantize_observation(plus[name],step,limit)-quantize_observation(minus[name],step,limit))/(2*delta))
                                jac=np.stack(jac,1);x=quantize_observation(base[name][:,anchor],step,limit)/lock['scales'][name]
                                error,mse=loss_by_group(score(w,bias,x),y,g['binary_columns'],g['continuous_columns'])
                                event_key=tag+'_spike'
                                crossing=float(np.mean([np.mean(a[event_key][:,anchor]!=b[event_key][:,anchor]) for a,b in pairs])) if event_key in base else None
                                history_crossing=float(np.mean([np.any(a[event_key]!=b[event_key],axis=(1,2)).mean() for a,b in pairs])) if event_key in base else None
                                upstream_crossing=float(np.mean([np.any(a['first_spike']!=b['first_spike'],axis=(1,2)).mean() for a,b in pairs])) if 'first_spike' in base else None
                                sample_energy=np.sum(jac[:,:,anchor]**2,axis=(1,2))
                                rows.append({'threshold_multiplier':ratio,'measurement_sigma':sigma,'source':g['source'],
                                    'delta':delta,'view':name,'observer_bits_reference':bits,'rounding_step':step,
                                    'deadline':anchor,'onset':onset,'finite_scale_metric_trace':float(np.mean(np.sum(jac[:,:,anchor]**2,axis=(1,2)))),
                                    'event_crossing_fraction':crossing,'own_layer_any_time_scene_crossing_fraction':history_crossing,
                                    'first_layer_any_time_scene_crossing_fraction':upstream_crossing,
                                    'scene_metric_trace_quantiles_50_95_99':np.quantile(sample_energy,[.5,.95,.99]),
                                    'scene_metric_trace_max':float(sample_energy.max()),
                                    'scenes_with_nonzero_metric':int(np.sum(sample_energy>1e-18)),
                                    'observer_clipping_fraction':float(np.mean(np.abs(base[name][:,anchor])>limit)) if limit is not None else 0.,
                                    'fixed_primary_reader_error':float(error.mean()),'fixed_primary_reader_MSE':float(mse.mean()),
                                    'pre_write_jacobian_max_abs':float(np.max(np.abs(jac[:,:,:onset]),initial=0))})
                print('event panel threshold',ratio,'sigma',sigma,flush=True)
    finally:
        for c,v in zip(cells,thresholds):c.threshold=v
    assert hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()==sha and tree_hash(bank)==lock['reader_tree_sha256']
    report={'module':'L3.6 event scale/noise/threshold/observer precision closure','rows':rows,'forward_captures':forward,
            'checkpoint_sha256':sha,'independent_scenes':scenes,'checkpoint_and_threshold_restored':True,'new_SNN_optimizer_updates':0,
            'precision_scope':'Finite-level clipped observation quantization with calibration-only range. The encoded observation needs the declared bits per coordinate plus shared calibration range; internal SNN arithmetic remains floating point.',
            'interpretation':'Same original labelled contents and scene IDs across variants; noise directions shared but measurement sigma and sensory energy normalization may change with config. Threshold variants are diagnostic interventions; fixed-reader risk includes representation/reader mismatch. Deadline-only event agreement does not rule out earlier event crossings. Rare crossings can dominate a small-delta mean; quantiles and whole-path scene crossings are included. No boundary-area estimate or Fisher-information identification.'}
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('primary');p.add_argument('--output',required=True);p.add_argument('--run-directory');p.add_argument('--scenes',type=int,default=128)
    a=p.parse_args();r=run(a.primary,a.output,a.run_directory,a.scenes);print('Scale panel rows',len(r['rows']),'captures',r['forward_captures'])
