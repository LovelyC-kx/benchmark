"""Actual binary edge: content donor versus exact flip/count matched controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge43 import ROOT,RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,capture_edge,parameter_hash,write_json,save_tree
from temporal_probes import paired_bound
from intervention_math43 import matched_binary_message
from run_state_interventions import native_losses

def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    if not (ROOT/'evidence/fitted/freeze_contract.json').exists():raise ValueError('Freeze the declared protocol before evaluation')
    model,cfg,meta=load_model(RUN_DIRECTORY);before=parameter_hash(model)
    a=fixed_batch(cfg.data,2048,'development_test',9060000,0);b=fixed_batch(cfg.data,2048,'development_test',9070000,0)
    ca=capture_edge(model,a['x']);pred0=ca['native_prediction'];messages=ca['sender_output'];rows={};scene={};prefix={};identities={}
    noop=capture_edge(model,a['x'],messages,range(a['x'].shape[1]))
    identities['no_op_prediction_max_abs']=float(np.max(np.abs(noop['native_prediction']-pred0)));del noop
    for source in ('H1','H4','C'):
        donor=hybrid(cfg.data,a,b,source);cd=capture_edge(model,donor['x']);m=cd['sender_output'];onset=int(a['event_times'][0,SOURCES.index(source)])
        assert np.array_equal(cd['sender_output'][:,:onset],messages[:,:onset])
        for kind,window in [('write_only',[onset]),('complete_suffix',list(range(onset,a['x'].shape[1])))]:
            key=source+'__'+kind;controls={}
            for seed in range(3):controls['count_flip_matched_'+str(seed)]=matched_binary_message(messages,m,436701+seed)
            conditions={'content_donor':m,**controls};rows[key]={}
            for name,replacement in conditions.items():
                r=capture_edge(model,a['x'],replacement,window)
                pred=r['native_prediction'];original=native_losses(pred,a);changed=native_losses(pred,donor)
                row={'original_label_risks':{s:{o:float(v.mean()) for o,v in z.items()} for s,z in original.items()},
                    'hybrid_label_risks':{s:{o:float(v.mean()) for o,v in z.items()} for s,z in changed.items()},
                    'natural_donor_prediction_MSE':float(np.mean((pred-cd['native_prediction'])**2)),
                    'sender_state_change_max_abs':float(np.max(np.abs(r['sender_post']-ca['sender_post']))),
                    'actual_edge_arrival_max_abs':float(np.max(np.abs(r['sender_output']-r['receiver_affine_input'])))}
                if name!='content_donor':
                    row['per_step_spike_count_mismatch']=int(np.sum(replacement.sum(-1)!=m.sum(-1)))
                    row['per_step_flip_count_mismatch']=int(np.sum((replacement!=messages).sum(-1)!=(m!=messages).sum(-1)))
                if name=='content_donor' and kind=='complete_suffix':
                    row['receiver_donor_replay_max_abs']=float(np.max(np.abs(r['receiver_post']-cd['receiver_post'])))
                    row['native_donor_replay_max_abs']=float(np.max(np.abs(pred-cd['native_prediction'])))
                rows[key][name]=row;prefix[key+'::'+name]=pred[:32]
                for s,z in changed.items():
                    for obj,value in z.items():scene[key+'::'+name+'::'+s+'::'+obj]=value
                del r
            print('evaluated edge context',key,flush=True)
        del cd
    family=len(rows)*2;comparisons={}
    for key in rows:
        for source in ('H1','C'):
            donor=scene[key+'::content_donor::'+source+'::derived_classification']
            control=np.mean([scene[key+'::count_flip_matched_'+str(seed)+'::'+source+'::derived_classification'] for seed in range(3)],axis=0)
            comparisons[key+'::'+source]=paired_bound(control,donor,family)
    np.savez_compressed(out/'scene_losses.npz',**scene);save_tree(out/'prediction_prefix',prefix)
    assert parameter_hash(model)==before
    report={'module':'L4.3 content-specific binary edge','evaluation_pairs':2048,'contexts':len(rows),'conditions':sum(len(v) for v in rows.values()),
        'results':rows,'identities':identities,'bounded_classification_comparison_family':family,'classification_donor_gain_vs_count_flip_matched':comparisons,
        'core_parameters_unchanged':True,'new_SNN_training':0,'formal_benchmark_completed':0,'audit_test_accessed':False,
        'scope':'Actual two-layer feedforward MLP edge. Same pair rows as state analysis; not an independent replication. Each random control matches flips and total emitted count at every scene/time, so this tests unit/path pattern beyond these counts, not an isolated millisecond timing advantage.'}
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=run(a.output);print('Actual edge conditions',r['conditions'])
