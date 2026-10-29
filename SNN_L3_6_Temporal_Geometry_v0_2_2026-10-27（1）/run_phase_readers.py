"""Train/choose/freeze phase-conditioned observers before heldout paired schedules."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from bridge35 import load_model,fixed_batch,flatten_contents
from temporal_capture import base_capture,temporal_views
from phase_readers import clock_views,known_phase_control
from temporal_probes import fit_head,loss_by_group,tree_hash,paired_bound
from temporal_contracts import independent_ids
from artifacts35 import save_tree,write_json
from run_temporal_review import target_bank
ROOT=Path(__file__).resolve().parent
CANDIDATES=('pooled','calendar','source_age','input_phase','state_phase','phase_only')

def collect(model,cfg,n,split,first_id,schedules):
    rows=[]
    for schedule in schedules:
        batch=fixed_batch(cfg.data,n,split,first_id,schedule)
        rows.append({'schedule':schedule,'batch':batch,'views':temporal_views(base_capture(model,batch['x'])),
                     'y':target_bank(flatten_contents(batch,cfg.data))[0]})
    return rows

def features(row,name,scale,j,frequency):
    x=np.asarray(row['views'][name],float)/scale
    return clock_views(x,row['batch']['x'][...,-1],row['batch']['event_times'][:,j],frequency)

def evaluate(rows,name,scale,j,group,heads,frequency):
    losses={k:[] for k in CANDIDATES}
    q=len(group['continuous_columns']);columns=group['continuous_columns']+group['binary_columns']
    for row in rows:
        fs=features(row,name,scale,j,frequency);y=row['y'][:,columns];onset=int(row['batch']['event_times'][0,j])
        T=row['batch']['x'].shape[1]
        for candidate,x in fs.items():
            w,o,b=heads[candidate];p=(x[:,onset:]-o)@w+b
            p=p.reshape(1,-1,p.shape[-1]);target=np.repeat(y,T-onset,axis=0)
            binary,mse=loss_by_group(p,target,list(range(q,len(columns))),list(range(q)))
            losses[candidate].append((binary.reshape(len(y),-1).mean(1),mse.reshape(len(y),-1).mean(1)))
    return {k:(np.mean([v[0] for v in a],axis=0),np.mean([v[1] for v in a],axis=0)) for k,a in losses.items()}

def run(primary,out,run_directory=None,selection_scenes=1024,evaluation_scenes=2048):
    primary,out=Path(primary),Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    lock=json.loads((primary/'freeze_contract.json').read_text())
    directory=Path(run_directory) if run_directory else ROOT/'evidence/source_smoke_run'
    checkpoint_sha=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()
    if checkpoint_sha!=lock['checkpoint_sha256']:raise ValueError('Checkpoint differs from primary')
    model,cfg,_=load_model(directory);frequency=.125
    train=collect(model,cfg,128,'probe_train',7710000,(0,1,2))
    selection=collect(model,cfg,selection_scenes,'probe_val',7720000,(3,))
    independent_ids(np.arange(7710000,7710128),np.arange(7720000,7720000+selection_scenes),np.arange(7730000,7730000+evaluation_scenes))
    _,_,groups=target_bank(flatten_contents(train[0]['batch'],cfg.data))
    views=[v for v in ('first_post','last_post','first_spike','last_spike','core_post','first_analog_output','last_analog_output') if v in train[0]['views']]
    bank={};choices={};budgets={}
    for name in views:
        bank[name]={};choices[name]={};budgets[name]={}
        for j,g in enumerate(groups):
            xs={c:[] for c in CANDIDATES};ys=[];cols=g['continuous_columns']+g['binary_columns']
            for row in train:
                fs=features(row,name,lock['scales'][name],j,frequency)
                onset=int(row['batch']['event_times'][0,j]);T=row['batch']['x'].shape[1]
                ys.append(np.repeat(row['y'][:,cols],T-onset,axis=0))
                for c,x in fs.items():xs[c].append(x[:,onset:].reshape(-1,x.shape[-1]))
            y=np.concatenate(ys);heads={}
            for c in CANDIDATES:
                xx=np.concatenate(xs[c]);heads[c]=fit_head(xx,y)
            bank[name][g['source']]=heads
            dl=evaluate(selection,name,lock['scales'][name],j,g,heads,frequency)
            risks={c:float(v[0].mean()) for c,v in dl.items()}
            choices[name][g['source']]={'chosen':min(CANDIDATES,key=lambda c:(risks[c],CANDIDATES.index(c))), 'selection_errors':risks}
            budgets[name][g['source']]={c:{'feature_dimension':h[0].shape[0],
                'linear_parameters':int(h[0].size+h[2].size),'training_rows':len(y),'unique_labelled_scenes':128,
                'time_rows_and_schedules_not_independent':True} for c,h in heads.items()}
        print('phase reader prepared',name,flush=True)
    save_tree(out/'frozen_readers',bank)
    freeze={'checkpoint_sha256':checkpoint_sha,'reader_sha256':tree_hash(bank),'choices':choices,'budgets':budgets,
            'frequency_cycles_per_step':frequency,'frequency_selection':'fixed in code before all pools',
            'train':[128,'probe_train',7710000,[0,1,2]],'selection':[selection_scenes,'probe_val',7720000,[3]],
            'evaluation':[evaluation_scenes,'development_test',7730000,[4,5]],
            'independent_unit':'content/noise scene; schedule and time risks averaged inside each scene',
            'input_phase_permission':'causal filter of event marker available in sensory input',
            'source_age_permission':'external source-specific write-onset ledger; extra observer side information',
            'state_phase_permission':'causal filter of view population mean; nonlinear history observer, not evidence of an autonomous rhythm',
            'frozen_before_evaluation_generation':True}
    write_json(out/'freeze_contract.json',freeze)
    del train,selection
    held=collect(model,cfg,evaluation_scenes,'development_test',7730000,(4,5))
    results={};arrays={};family_size=len(views)*len(groups)
    for name in views:
        results[name]={}
        for j,g in enumerate(groups):
            source=g['source'];loss=evaluate(held,name,lock['scales'][name],j,g,bank[name][source],frequency)
            chosen=choices[name][source]['chosen'];a,b=loss['pooled'][0],loss[chosen][0]
            results[name][source]={'chosen':chosen,'all_candidate_risks':{c:{'classification_error':float(v[0].mean()),'MSE':float(v[1].mean())} for c,v in loss.items()},
                'paired_bound_selected_vs_pooled':paired_bound(a,b,family_size)}
            arrays[f'{name}__{source}__pooled']=a;arrays[f'{name}__{source}__chosen']=b
    assert tree_hash(bank)==freeze['reader_sha256']
    np.savez_compressed(out/'scene_losses.npz',**arrays)
    r={'module':'L3.6 causal phase-conditioned observer closure','known_phase_control':known_phase_control(),
       'results':results,'family_size':family_size,'evaluation_scenes':evaluation_scenes,
       'positive_simultaneous_lower_bounds':sum(v['paired_bound_selected_vs_pooled']['interval'][0]>0 for rows in results.values() for v in rows.values()),
       'new_SNN_optimizer_updates':0,'audit_test_accessed':False,
       'scope':'One frozen checkpoint, 128 unique training labels per head. Different observation permissions and feature costs are disclosed; this is not training the original Rhythm-SNN or establishing a rhythmic neural mechanism. This confidence family is separate from primary/delay families.'}
    write_json(out/'report.json',r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('primary');p.add_argument('--output',required=True);p.add_argument('--run-directory');p.add_argument('--selection-scenes',type=int,default=1024);p.add_argument('--evaluation-scenes',type=int,default=2048)
    a=p.parse_args();r=run(a.primary,a.output,a.run_directory,a.selection_scenes,a.evaluation_scenes);print('Phase comparisons',r['family_size'],'positive lower bounds',r['positive_simultaneous_lower_bounds'])
