"""Legal binary recovery with explicit assignment operators and matched budgets."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge44 import ROOT,RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,capture_edge,load_tree,matched_binary_message,write_json,save_tree
from recovery_math44 import four_losses,output_decomposition
from run_recovery_main import losses

def assign_operator(clean,damaged,repaired):
    mask=repaired!=damaged;patch_only=clean.copy();patch_only[mask]=repaired[mask]
    return patch_only,mask

def run(output,n=256):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    f=json.loads((ROOT/'evidence/fitted/freeze_contract.json').read_text());bank=load_tree(ROOT/'evidence/fitted/frozen_bank');p=f['evaluation_pools']['confirmation']
    model,cfg,_=load_model(RUN_DIRECTORY);a=fixed_batch(cfg.data,n,'development_test',p['recipient'],0);b=fixed_batch(cfg.data,n,'development_test',p['donor'],0)
    clean=capture_edge(model,a['x']);m=clean['sender_output'];cl=losses(clean['native_prediction'],a);rows={};prefix={};scene={};columns=list(range(4))+list(range(16,20))
    for source in ('H1','H4','C'):
        j=SOURCES.index(source);onset=int(a['event_times'][0,j]);neutral=hybrid(cfg.data,a,b,source,zero=True);neutral_run=capture_edge(model,neutral['x'])
        dm=m.copy();dm[:,onset:]=neutral_run['sender_output'][:,onset:];damaged=capture_edge(model,a['x'],dm,range(16));dl=losses(damaged['native_prediction'],a)
        assert np.array_equal(damaged['native_prediction'],neutral_run['native_prediction'])
        # Candidate pulse units are derived without recovery outcomes, from a frozen membrane operator.
        e=bank['patch_erasers'][source+'_l0_t12'];q=np.eye(32)-e['matrix'];importance=np.linalg.norm(q,axis=0)
        chosen=np.argsort(-importance,kind='stable')[:8];random_units=np.random.default_rng(440808).choice(32,8,replace=False)
        for start in (onset,onset+1,14):
            time_mask=np.arange(16)>=start;full=dm.copy();full[:,time_mask]=m[:,time_mask]
            one=dm.copy();one[:,start]=m[:,start];geometry=dm.copy();random=dm.copy()
            for t in np.flatnonzero(time_mask):geometry[:,t,chosen]=m[:,t,chosen];random[:,t,random_units]=m[:,t,random_units]
            methods={'clean_full_suffix':full,'clean_one_step':one,'membrane_geometry_mask':geometry,'random_unit_mask':random}
            for seed in range(3):methods['count_flip_matched_'+str(seed)]=matched_binary_message(dm,geometry,440900+seed)
            cases={}
            for name,repaired in methods.items():
                assert np.isin(repaired,[0,1]).all();po,mask=assign_operator(m,dm,repaired)
                pr=capture_edge(model,a['x'],po,range(16));rr=capture_edge(model,a['x'],repaired,range(16));pl=losses(pr['native_prediction'],a);rl=losses(rr['native_prediction'],a)
                row={'four_group_risks':{s:{metric:four_losses(cl[s][metric],dl[s][metric],pl[s][metric],rl[s][metric],damage_floor=.01 if metric=='MSE' else None) for metric in cl[s]} for s in cl},
                    'output_analysis':output_decomposition(clean['native_prediction'][:,columns],damaged['native_prediction'][:,columns],pr['native_prediction'][:,columns],rr['native_prediction'][:,columns],a['latents'][:,[0,4]].reshape(n,8)),
                    'sender_state_change_max_abs':float(np.max(np.abs(rr['sender_post']-clean['sender_post']))),
                    'actual_edge_arrival_max_abs':float(np.max(np.abs(rr['sender_output']-rr['receiver_affine_input']))),
                    'restored_bit_assignments':int(mask.sum()),'binary_legal':True}
                if name.startswith('count_flip_matched'):
                    row['per_scene_step_flip_count_mismatches']=int(np.sum((repaired!=dm).sum(-1)!=(geometry!=dm).sum(-1)))
                    row['per_scene_step_spike_count_mismatches']=int(np.sum(repaired.sum(-1)!=geometry.sum(-1)))
                if start==onset and name=='clean_full_suffix':
                    row['exact_full_restore_prediction_max_abs']=float(np.max(np.abs(rr['native_prediction']-clean['native_prediction'])))
                    assert row['exact_full_restore_prediction_max_abs']==0
                cases[name]=row;key=f'{source}_p{start}::{name}';prefix[key]=rr['native_prediction'][:16]
                for s,z in rl.items():
                    for metric,value in z.items():scene[key+'::'+s+'::'+metric]=value
            rows[f'{source}_p{start}']={'source':source,'damage_window':list(range(onset,16)),'restore_start':start,'geometry_units':chosen.tolist(),'random_units':random_units.tolist(),'methods':cases}
    save_tree(out/'prediction_prefix',prefix);np.savez_compressed(out/'repair_scene_losses.npz',**scene)
    report={'module':'L4.4 actual binary-edge recovery','confirmation_subset_scene_pairs':n,'contexts':len(rows),'four_group_cells':sum(len(v['methods']) for v in rows.values()),
        'results':rows,'saved_repair_loss_arrays':len(scene),
        'scope':'First256 of main confirmation scenes. Same fixed bit-assignment operator in clean and damaged groups; this is not an equal additive real-valued delta. Pulse-unit candidate mask comes from the frozen membrane operator, not test rescue selection. Flip/count budgets match on damaged recipient; no claim of equal budgets on clean recipients or hardware energy.'}
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=run(a.output);print('Binary four-group cells',r['four_group_cells'])
