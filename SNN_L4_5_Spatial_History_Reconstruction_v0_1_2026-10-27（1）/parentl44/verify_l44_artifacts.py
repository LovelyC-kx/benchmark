"""Recompute every four-cell risk/interval and replay all saved main prefixes."""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
import torch
from bridge44 import ROOT,PARENT,RUN_DIRECTORY,load_model,fixed_batch,hybrid,load_tree,tree_hash,write_json,parameter_hash
from recovery_math44 import four_losses,bounded_interval
from recovery_runtime44 import scheduled_run,trace_site,damage_edit,additive_edit
from run_recovery_main import build_patches,validation

def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

def verify(replay=True):
    torch.set_num_threads(1);python_count=0;json_count=0
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text());python_count+=1
    for p in ROOT.rglob('*.json'):read(p);json_count+=1
    provenance=read(ROOT/'references/project_provenance.json')
    for name,sha in provenance['inherited_files'].items():assert hashlib.sha256((PARENT/name).read_bytes()).hexdigest()==sha
    f=read(ROOT/'evidence/fitted/freeze_contract.json');bank=load_tree(ROOT/'evidence/fitted/frozen_bank');assert tree_hash(bank)==f['bank_sha256']
    assert tree_hash(load_tree(PARENT/'evidence/stability_fitted/frozen_bank'))==f['parent_stable_bank_sha256']
    assert hashlib.sha256((RUN_DIRECTORY/'best.pt').read_bytes()).hexdigest()==f['checkpoint_sha256']
    summaries={};loss_count=0;four_cells=0;prefix_count=0;payload_gaps=[];prediction_gaps=[];site_gaps=[];semantic_gaps=[]
    for pool,settings in f['evaluation_pools'].items():
        directory=ROOT/'evidence'/pool;r=read(directory/'report.json');data=np.load(directory/'scene_losses.npz');expected=load_tree(directory/'replay_prefix')
        assert r['four_group_cells']==380 and len(r['bounded_comparisons'])==4616 and len(data.files)==4650
        for name in data.files:
            a=data[name];assert a.shape==(settings['n'],) and np.isfinite(a).all()
        for c in f['damage_contexts']:
            for q in ('H1','C'):
                for metric in ('capped_MSE4','derived_sign'):
                    value=data[c['key']+'::damage::'+q+'::'+metric]-data['clean::'+q+'::'+metric]
                    assert bounded_interval(value,4616)==r['bounded_comparisons'][c['key']+'::damage::'+q+'::'+metric]
        for c in f['recovery_contexts']:
            key=c['key44'];context=r['results'][key];assert 'damaged_state_validation' in context
            for method,row in context['methods'].items():
                for q in ('H1','C'):
                    for metric in ('MSE','capped_MSE4','derived_sign'):
                        values=(data['clean::'+q+'::'+metric],data[c['key']+'::damage::'+q+'::'+metric],
                            data[key+'::'+method+'::patch_only::'+q+'::'+metric],data[key+'::'+method+'::repair::'+q+'::'+metric])
                        assert four_losses(*values,damage_floor=.01 if metric=='MSE' else None)==row['four_group_risks'][q][metric]
                        if metric!='MSE':
                            for effect,v,half in [('rescue',values[1]-values[3],1.),('patch_only_gain',values[0]-values[2],1.),('risk_interaction',values[1]-values[3]-values[0]+values[2],2.)]:
                                assert bounded_interval(v,4616,half_width=half)==r['bounded_comparisons'][key+'::'+method+'::'+q+'::'+metric+'::'+effect]
                oa=row['output_analysis'];assert oa['risk_identity_absolute_residual']<1e-12*(1+abs(oa['risk_interaction']))
                for name in ('immediate_algebraic_restore_prediction_max_abs','terminal_readout_overwrite_prediction_max_abs'):
                    if name in row:assert row[name]==0
                four_cells+=1
        loss_count+=len(data.files)
        summaries[pool]={'independent_scene_pairs':settings['n'],'positive_bounded_lower_bounds':sum(v['interval'][0]>0 for v in r['bounded_comparisons'].values()),
            'MSE_recovery_ratios_defined':sum(row['four_group_risks'][q]['MSE']['ratio_defined'] for v in r['results'].values() for row in v['methods'].values() for q in ('H1','C'))}
        if replay:
            model,cfg,_=load_model(RUN_DIRECTORY);sha=parameter_hash(model);a=fixed_batch(cfg.data,128,'development_test',settings['recipient'],0);b=fixed_batch(cfg.data,128,'development_test',settings['donor'],0)
            clean=scheduled_run(model,a['x']);br=scheduled_run(model,b['x']);donors={s:hybrid(cfg.data,a,b,s) for s in ('H1','H4','C')}
            fr={s:scheduled_run(model,d['x']) for s,d in donors.items()};ar={s:scheduled_run(model,hybrid(cfg.data,b,a,s)['x']) for s in donors}
            damaged={c['key']:scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']])]) for c in f['damage_contexts']}
            for c in f['recovery_contexts']:
                d=damaged[c['key']];site=c['patch_site'];h=trace_site(clean,c['layer'],c['patch_time'])
                patches,_=build_patches(c,clean,d,br,fr[c['source']],ar[c['source']],bank['patch_erasers'][c['patch_key']])
                for name in f['methods']:
                    delta=patches[name];e=expected[c['key44']+'::'+name];payload_gaps.append(float(np.max(np.abs(delta[:16]-e['delta_prefix']))))
                    patch=additive_edit(c['layer'],c['patch_time'],delta);p=scheduled_run(model,a['x'],[patch],keep_traces=False)
                    rr=scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']]),patch],keep_traces=False)
                    prediction_gaps.extend([float(np.max(np.abs(p['prediction'][:16]-e['patch_only_prediction_prefix']))),float(np.max(np.abs(rr['prediction'][:16]-e['repair_prediction_prefix'])))])
                    site_gaps.append(float(np.max(np.abs(rr['edits']['patch']['after'][:16]-e['repair_site_prefix']))));prefix_count+=1
            assert parameter_hash(model)==sha
    if replay:assert max(payload_gaps+prediction_gaps+site_gaps)==0
    sup=read(ROOT/'evidence/supplements/report.json');assert sup['event_four_group_cells']==72 and sup['joint_subset_conditions']==120
    fixed=[v['fixed_clean_events'] for c in sup['event_results'].values() for v in c.values()]
    assert max(v['maximum_forced_emit_policy_gap'] for v in fixed)==0
    assert max(v['output_interaction_max_abs'] for v in fixed)<2e-6
    for key,row in sup['joint_results'].items():
        assert row['output_factorial']['reconstruction_max_abs']<1e-12
        if key.endswith('_full'):assert row['full_joint_clean_prediction_max_abs']==0
    binary=read(ROOT/'evidence/binary_recovery/report.json');assert binary['four_group_cells']==63
    bl=np.load(ROOT/'evidence/binary_recovery/repair_scene_losses.npz');assert len(bl.files)==378
    for key,context in binary['results'].items():
        for method,row in context['methods'].items():
            assert row['binary_legal'] and row['sender_state_change_max_abs']==row['actual_edge_arrival_max_abs']==0
            if method.startswith('count_flip_matched'):assert row['per_scene_step_flip_count_mismatches']==row['per_scene_step_spike_count_mismatches']==0
            if 'exact_full_restore_prediction_max_abs' in row:assert row['exact_full_restore_prediction_max_abs']==0
            for q,metrics in row['four_group_risks'].items():
                for metric,values in metrics.items():assert abs(bl[key+'::'+method+'::'+q+'::'+metric].mean()-values['R11'])<1e-12
    path=read(ROOT/'evidence/path_reblocking/report.json');assert path['cases']==12
    for row in path['results'].values():assert row['reblocked_unpatched_damage_prediction_max_abs']==row['offwindow_repaired_prediction_max_abs']==row['sender_state_reblock_change_max_abs']==row['actual_edge_arrival_max_abs']==0
    grid=read(ROOT/'evidence/scoped_snapshot_grid.json');assert grid['configurations']==240 and grid['scoped_conditions']==1920
    for row in grid['results']:
        assert row['scopes']['system']['max_abs']==0
        if row['readout']=='last':assert row['scopes']['core']['max_abs']==0
    geometry=read(ROOT/'evidence/event_geometry/report.json');assert geometry['dose_conditions']==72
    return {'module':'L4.4 artifact verification','python_AST_files':python_count,'strict_JSON_files':json_count,'unchanged_parent_files_verified':len(provenance['inherited_files']),
        'main_four_group_cells_recomputed':four_cells,'main_scene_loss_arrays_recomputed':loss_count,'binary_repair_scene_loss_arrays_recomputed':len(bl.files),
        'bounded_intervals_recomputed':2*f['bounded_comparison_family_per_pool'],'main_payload_and_prediction_prefix_cases_replayed':prefix_count,
        'prefix_scenes_per_case':16,'original_replay_batch_size':128,'prefix_payload_max_abs':max(payload_gaps,default=0),
        'prefix_prediction_max_abs':max(prediction_gaps,default=0),'prefix_state_max_abs':max(site_gaps,default=0),
        'cohort_summaries':summaries,'fixed_event_output_interaction_max_abs':max(v['output_interaction_max_abs'] for v in fixed),
        'snapshot_binary_joint_and_reblocking_contracts_verified':True,
        'scope':'Main model predictions, patch construction, risk contrasts and intervals reproduced. Supplementary exact contracts and saved loss means checked; supplements are shared confirmation subsets, not independent replications. Formal benchmark remains unrun.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--skip-replay',action='store_true');a=p.parse_args();r=verify(not a.skip_replay);write_json(a.output,r);print(json.dumps(r,indent=2))
