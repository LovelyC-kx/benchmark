"""Restore sender membrane, then reblock its actual future communication branch."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge44 import ROOT,RUN_DIRECTORY,load_model,fixed_batch,load_tree,write_json,save_tree
from recovery_runtime44 import scheduled_run,trace_site,damage_edit,additive_edit
from edge_reblock44 import run_with_edge_policy

def run(output,n=256):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);f=json.loads((ROOT/'evidence/fitted/freeze_contract.json').read_text());bank=load_tree(ROOT/'evidence/fitted/frozen_bank')
    model,cfg,_=load_model(RUN_DIRECTORY);a=fixed_batch(cfg.data,n,'development_test',f['evaluation_pools']['confirmation']['recipient'],0)
    clean=scheduled_run(model,a['x']);rows={};prefix={}
    for c in f['recovery_contexts']:
        if c['layer']!=0 or c['time']!=12 or c['patch_time'] not in (13,14):continue
        de=damage_edit(c,bank['damage_erasers'][c['key']]);damaged=scheduled_run(model,a['x'],[de]);messages=damaged['traces']['backbone.cells.0/emit_event']
        base=trace_site(clean,0,c['patch_time'])-trace_site(damaged,0,c['patch_time'])
        for mode in ('full','target'):
            delta=base if mode=='full' else base@(np.eye(32)-bank['patch_erasers'][c['patch_key']]['matrix'])
            edits=[de,additive_edit(0,c['patch_time'],delta)];repair=scheduled_run(model,a['x'],edits)
            blocked=run_with_edge_policy(model,a['x'],edits,messages,list(range(c['patch_time']+1,16)))
            offwindow=run_with_edge_policy(model,a['x'],edits,messages,list(range(c['patch_time']+1)))
            row={'reblocked_unpatched_damage_prediction_max_abs':float(np.max(np.abs(blocked['prediction']-damaged['prediction']))),
                'offwindow_repaired_prediction_max_abs':float(np.max(np.abs(offwindow['prediction']-repair['prediction']))),
                'sender_state_reblock_change_max_abs':float(np.max(np.abs(blocked['traces']['backbone.cells.0/post']-repair['traces']['backbone.cells.0/post']))),
                'actual_edge_arrival_max_abs':blocked['edge_arrival_max_abs'],
                'unblocked_repair_vs_damage_prediction_MSE':float(np.mean((repair['prediction']-damaged['prediction'])**2)),
                'reblocked_vs_repaired_prediction_MSE':float(np.mean((blocked['prediction']-repair['prediction'])**2))}
            assert row['reblocked_unpatched_damage_prediction_max_abs']==row['offwindow_repaired_prediction_max_abs']==row['sender_state_reblock_change_max_abs']==0
            key=c['key44']+'::'+mode;rows[key]=row;prefix[key]={'repair':repair['prediction'][:16],'blocked':blocked['prediction'][:16]}
    save_tree(out/'prediction_prefix',prefix)
    record={'module':'L4.4 actual communication reblocking','confirmation_subset_scene_pairs':n,'cases':len(rows),'results':rows,
        'scope':'First256 confirmation scenes. Restored sender v retained; only future returned-output messages replaced by unpatched damaged messages. Off-window reblocking is a temporal negative control. Exact equalities are specific to this feedforward graph; they do not establish uniqueness in other backbones.'}
    write_json(out/'report.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=run(a.output);print('Path reblocking cases',r['cases'])
