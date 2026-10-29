"""Explicit supplementary diagnostics: new delays, source clocks, finite scales."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from bridge35 import load_model,fixed_batch,flatten_contents,repeat_panel
from bridge import _render
from temporal_capture import base_capture,temporal_views
from temporal_alignment import fit_transport,push_reader
from temporal_probes import score,loss_by_group,tree_hash
from temporal_theory import central_metric
from artifacts35 import load_tree,save_tree,write_json
from run_temporal_review import target_bank,FAMILIES as PRIMARY_FAMILIES

FAMILIES=PRIMARY_FAMILIES+('similarity','rotation')

ROOT=Path(__file__).resolve().parent

def run(primary,out,run_directory=None):
    primary,out=Path(primary),Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    lock=json.loads((primary/'freeze_contract.json').read_text());bank=load_tree(primary/'frozen_readers')
    assert tree_hash(bank)==lock['reader_tree_sha256']
    run_directory=Path(run_directory) if run_directory is not None else ROOT/'evidence/source_smoke_run'
    assert hashlib.sha256((run_directory/'best.pt').read_bytes()).hexdigest()==lock['checkpoint_sha256']
    model,cfg,meta=load_model(run_directory);long_cfg=replace(cfg.data,blank_delay=11)
    anchor=max(lock['source_times'])
    if anchor>=24:raise ValueError('Current generator supports at most 24 steps')
    b=fixed_batch(long_cfg,512,'geometry_fit',7650000,0);cal=temporal_views(base_capture(model,b['x']))
    times=(13,15,19,23);prepared={}
    for name,c in cal.items():
        c=np.asarray(c,float)/lock['scales'][name];prepared[name]={}
        for t in times:prepared[name][str(t)]={f:fit_transport(c[:,t],c[:,anchor],f) for f in FAMILIES}
    save_tree(out/'delay_maps',prepared)
    write_json(out/'freeze_contract.json',{'delay_map_sha256':tree_hash(prepared),'primary_reader_sha256':tree_hash(bank),
        'checkpoint_sha256':lock['checkpoint_sha256'],'long_blank_delay':11,
        'calibration':[512,'geometry_fit',7650000],'delay_evaluation':[1024,'development_test',7660000],
        'schedule_evaluation':[1024,'development_test',7640000],'times':times,'blank_delays':[1,3,7,11],
        'alignment_families':FAMILIES,'source_reader_time':anchor,
        'scope':'Supplementary development analysis, not independent SNN/model confirmation; map/reader choices fixed before these new scenes.'})
    eb=fixed_batch(long_cfg,1024,'development_test',7660000,0);ev=temporal_views(base_capture(model,eb['x']));y,_,groups=target_bank(flatten_contents(eb,cfg.data));delay={}
    for name,x in ev.items():
        x=np.asarray(x,float)/lock['scales'][name];f=bank[name];delay[name]={}
        for t in times:
            delay[name][str(t)]={}
            for family,m in prepared[name][str(t)].items():
                w,bias=push_reader(m,f['weights'][:,anchor],f['origins'][:,anchor],f['bias'][:,anchor]);p=score(w,bias,x[:,t]);rows={}
                for g in groups:
                    a,b=loss_by_group(p,y,g['binary_columns'],g['continuous_columns']);rows[g['source']]={'classification_error':float(a.mean()),'MSE':float(b.mean())}
                delay[name][str(t)][family]=rows
    print('completed supplementary blank-delay scans',flush=True)
    schedule=[]
    for s in range(6):
        b=fixed_batch(cfg.data,1024,'development_test',7640000,s);ev=temporal_views(base_capture(model,b['x']));y,_,groups=target_bank(flatten_contents(b,cfg.data))
        for name in [k for k in ('first_post','last_post','first_spike','last_spike','first_analog_output','last_analog_output') if k in ev]:
            x=np.asarray(ev[name],float)/lock['scales'][name];f=bank[name]
            for j,g in enumerate(groups):
                reference_onset=lock['groups'][j]['onset'];actual_onset=int(b['event_times'][0,j])
                for age in (0,1,3):
                    source_time=reference_onset+age;actual_time=actual_onset+age
                    w=f['weights'][:,source_time];bias=f['bias'][:,source_time]-np.einsum('rd,rdo->ro',f['origins'][:,source_time],w)
                    row={'schedule':s,'view':name,'source':g['source'],'age':age,'reference_time':source_time,'actual_time':actual_time,
                         'source_available_at_fixed_calendar_time':source_time>=actual_onset}
                    for label,t in (('calendar',source_time),('event_age',actual_time)):
                        error,mse=loss_by_group(score(w,bias,x[:,t]),y,g['binary_columns'],g['continuous_columns'])
                        row[label]={'error':float(error.mean()),'MSE':float(mse.mean())}
                    schedule.append(row)
    print('completed six paired source-identity schedules',flush=True)
    # A separate conditional noise panel holds all contents/schedule fixed.
    panel=repeat_panel(cfg.data,64,8,'geometry_fit',7670100,0,'C','measurement_only')
    noise_views=base_capture(model,panel['x'].reshape(-1,*panel['x'].shape[2:]));noise_cov={}
    for name in ('first_post','last_post'):
        x=np.asarray(noise_views[name],float).reshape(64,8,noise_views[name].shape[1],-1);res=x-x.mean(1,keepdims=True)
        cov=np.einsum('nrtd,nrte->tde',res,res)/(64*7);noise_cov[name]=cov
    b=fixed_batch(cfg.data,64,'geometry_fit',7670000,0);metrics=[]
    for source,j in (('H1',0),('C',4)):
        for delta in (.001,.01,.1):
            plus={};minus={};event_flips=[]
            for k in range(cfg.data.visible_content_dim):
                pairs=[]
                for sign in (1,-1):
                    z=b['latents'].copy();z[:,j,k]+=sign*delta
                    x=_render(cfg.data,z,b['measurement_noise'],b['event_times'],b['source_present'])[:,:b['x'].shape[1]]
                    pairs.append(base_capture(model,x))
                for name in ('first_post','last_post'):
                    plus.setdefault(name,[]).append(pairs[0][name]);minus.setdefault(name,[]).append(pairs[1][name])
                event_flips.append({tag:np.mean(pairs[0][tag+'_spike']!=pairs[1][tag+'_spike'],axis=(0,2)) for tag in ('first','last') if tag+'_spike' in pairs[0]})
            for name in ('first_post','last_post'):
                p=np.stack(plus[name],1);m=np.stack(minus[name],1);r=central_metric(p,m,delta);J=r['jacobian'];gram=r['euclidean_pullback']
                onset=int(b['event_times'][0,j]);assert np.max(np.abs(J[:,:,:onset]),initial=0)==0
                regularized=[];ridges=[]
                for t,cov in enumerate(noise_cov[name]):
                    ridge=max(float(np.trace(cov)/len(cov))*.01,1e-8);ridges.append(ridge)
                    jt=J[:,:,t];inv=np.linalg.inv(cov+ridge*np.eye(len(cov)))
                    regularized.append(np.einsum('nkd,de,nle->nkl',jt,inv,jt).mean(0))
                tag=name.split('_')[0]
                metrics.append({'source':source,'view':name,'delta':delta,'onset':onset,
                    'mean_Euclidean_metric':gram.mean(0),'mean_metric_eigenvalues':np.linalg.eigvalsh(gram).mean(0),
                    'mean_noise_normalized_sensitivity':np.stack(regularized),'covariance_ridge':ridges,
                    'event_flip_fraction_by_time':np.mean([row[tag] for row in event_flips],axis=0) if tag in event_flips[0] else None,
                    'zero_metric_before_write':True,
                    'scope':'Finite-difference scale-dependent sensitivity. At event changes this is not a classical derivative or exact Fisher information.'})
    report={'module':'L3.6 supplementary diagnostics','blank_delay_scan':delay,'source_clock_scan':schedule,
            'finite_scale_task_metrics':metrics,'finite_difference_forward_captures':12*cfg.data.visible_content_dim,
            'conditional_noise_panel':{'scenes':64,'repeats':8,'fixed':'all contents and event schedule','changed':'measurement noise only'},
            'zero_new_SNN_updates':True,'audit_test_accessed':False,
            'source_optimizer_updates':meta['optimizer_updates'],
            'scope':'One frozen project checkpoint; supplementary development diagnostics. Delay maps see paired unlabelled calibration states, finite differences see latent coordinates as offline diagnostic access. Training quality must be verified separately.'}
    assert tree_hash(bank)==lock['reader_tree_sha256'] and tree_hash(prepared)==json.loads((out/'freeze_contract.json').read_text())['delay_map_sha256']
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('primary');p.add_argument('--output',required=True);p.add_argument('--run-directory');a=p.parse_args();run(a.primary,a.output,a.run_directory)
