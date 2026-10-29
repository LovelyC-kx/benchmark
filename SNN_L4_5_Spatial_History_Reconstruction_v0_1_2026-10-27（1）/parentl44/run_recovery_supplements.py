"""Predetermined small confirmation-cohort subsets for event and joint-owner checks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge44 import (ROOT,RUN_DIRECTORY,load_model,fixed_batch,hybrid,load_tree,tree_hash,write_json,save_tree,SOURCES)
from recovery_runtime44 import scheduled_run,trace_site,damage_edit,additive_edit
from recovery_math44 import output_decomposition,four_losses,mobius_outputs
from run_recovery_main import build_patches,losses

def run(output,n=256):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    f=json.loads((ROOT/'evidence/fitted/freeze_contract.json').read_text());bank=load_tree(ROOT/'evidence/fitted/frozen_bank');assert tree_hash(bank)==f['bank_sha256']
    p=f['evaluation_pools']['confirmation'];model,cfg,_=load_model(RUN_DIRECTORY)
    a=fixed_batch(cfg.data,n,'development_test',p['recipient'],0);b=fixed_batch(cfg.data,n,'development_test',p['donor'],0)
    clean=scheduled_run(model,a['x']);br=scheduled_run(model,b['x']);fr={s:scheduled_run(model,hybrid(cfg.data,a,b,s)['x']) for s in ('H1','H4','C')}
    ar={s:scheduled_run(model,hybrid(cfg.data,b,a,s)['x']) for s in fr};damaged={}
    events={path:clean['traces'][path+'/emit_event'] for path in ('backbone.cells.0','backbone.cells.1')}
    clean_fixed=scheduled_run(model,a['x'],fixed_events=events);assert np.array_equal(clean_fixed['prediction'],clean['prediction'])
    columns=list(range(4))+list(range(16,20));target=a['latents'][:,[0,4]].reshape(n,8);event_results={};prefix={}
    chosen=[c for c in f['recovery_contexts'] if c['time']==12 and c['patch_time'] in (13,14)]
    for c in chosen:
        if c['key'] not in damaged:damaged[c['key']]=scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']])])
        d=damaged[c['key']];patches,_=build_patches(c,clean,d,br,fr[c['source']],ar[c['source']],bank['patch_erasers'][c['patch_key']])
        fixed_damage=scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']])],fixed_events=events)
        rows={}
        for name in ('target_self_delta','target_reverse_delta','norm_random_0'):
            patch=additive_edit(c['layer'],c['patch_time'],patches[name]);regimes={}
            for regime,base,dd,policy in [('natural',clean,d,None),('fixed_clean_events',clean_fixed,fixed_damage,events)]:
                pr=scheduled_run(model,a['x'],[patch],fixed_events=policy)
                rr=scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']]),patch],fixed_events=policy)
                metrics=output_decomposition(base['prediction'][:,columns],dd['prediction'][:,columns],pr['prediction'][:,columns],rr['prediction'][:,columns],target)
                if policy is not None:
                    metrics['maximum_forced_emit_policy_gap']=max(float(np.max(np.abs(rr['traces'][path+'/emit_event']-event))) for path,event in events.items())
                regimes[regime]=metrics;prefix[c['key44']+'::'+name+'::'+regime]=rr['prediction'][:16]
            rows[name]=regimes
        event_results[c['key44']]=rows;print('event replay recovery',c['key44'],flush=True)
    # Both layers damaged together; single and joint patches use the same saved delta.
    unique={(c['source'],c['time']):c for c in f['damage_contexts']};joint_results={};joint_arrays={}
    for (source,time),template in unique.items():
        contexts=[next(c for c in f['damage_contexts'] if c['source']==source and c['time']==time and c['layer']==layer) for layer in (0,1)]
        de=[damage_edit(c,bank['damage_erasers'][c['key']]) for c in contexts]
        de[1].label='damage1';d=scheduled_run(model,a['x'],de)
        times=(4,14,15) if time==3 else (13,14,15) if time==12 else (15,)
        for pt in times:
            for mode in ('full','target'):
                deltas=[]
                for layer in (0,1):
                    delta=trace_site(clean,layer,pt)-trace_site(d,layer,pt)
                    if mode=='target':delta=delta@(np.eye(delta.shape[1])-bank['patch_erasers'][f'{source}_l{layer}_t{pt}']['matrix'])
                    deltas.append(delta)
                outputs={};risk_rows={}
                for subset in range(4):
                    ops=de+[additive_edit(layer,pt,deltas[layer],label='patch'+str(layer)) for layer in (0,1) if subset&(1<<layer)]
                    r=d if subset==0 else scheduled_run(model,a['x'],ops);outputs[subset]=r['prediction'][:,columns]
                    risk_rows[str(subset)]={s:{k:float(v.mean()) for k,v in z.items()} for s,z in losses(r['prediction'],a).items()}
                    joint_arrays[f'{source}_t{time}_p{pt}_{mode}::{subset}']=r['prediction'][:16]
                row={'output_factorial':mobius_outputs(outputs),'native_risks':risk_rows,
                    'full_joint_clean_prediction_max_abs':float(np.max(np.abs(outputs[3]-clean['prediction'][:,columns]))),
                    'conditional_effect_A_without_B_MSE':float(np.mean((outputs[1]-outputs[0])**2)),
                    'conditional_effect_A_with_B_MSE':float(np.mean((outputs[3]-outputs[2])**2))}
                if mode=='full':assert row['full_joint_clean_prediction_max_abs']==0
                joint_results[f'{source}_t{time}_p{pt}_{mode}']=row
    save_tree(out/'prediction_prefix',prefix);save_tree(out/'joint_prediction_prefix',joint_arrays)
    record={'module':'L4.4 event-policy and joint-owner recovery','confirmation_subset_scene_pairs':n,
        'event_contexts':len(chosen),'event_four_group_cells':len(chosen)*3*2,'event_results':event_results,
        'joint_contexts':len(joint_results),'joint_subset_conditions':len(joint_results)*4,'joint_results':joint_results,
        'full_joint_exact_restoration_is_operational_ceiling':True,
        'scope':'First256 of the main confirmation cohort, not another independent sample. Same patch vectors across natural and fixed-event policies. Fixed policy externally replays all emit/reset/aux branches; conditional diagnostic, not natural online dynamics.'}
    write_json(out/'report.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=run(a.output)
    print('Event four-group cells',r['event_four_group_cells'],'joint conditions',r['joint_subset_conditions'])
