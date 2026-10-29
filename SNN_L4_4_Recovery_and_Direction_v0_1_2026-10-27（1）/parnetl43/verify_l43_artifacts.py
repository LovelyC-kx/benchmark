"""Read-only contracts plus replay of every saved initial state/edge prefix."""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
import torch
from bridge43 import (ROOT,RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,baseline_sites,load_tree,tree_hash,capture_edge,write_json)
from intervention_math43 import apply_eraser,transport,norm_matched,matched_binary_message
from runtime_interventions43 import run_post_edit
from temporal_probes import paired_bound

def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

def means_check(directory,report,field):
    data=np.load(directory/'scene_losses.npz');count=0
    for key,a in report['results'].items():
        for operation,row in a.items():
            for source,values in row[field].items():
                for objective,value in values.items():
                    loss=data[key+'::'+operation+'::'+source+'::'+objective]
                    assert np.isfinite(loss).all() and len(loss)==report['evaluation_pairs']
                    assert abs(float(loss.mean())-value)<1e-10
                    if objective=='derived_classification':assert loss.min()>=0 and loss.max()<=1
                    count+=1
    assert count==len(data.files);return count

def verify(replay=True):
    torch.set_num_threads(1);parsed=0;strict=0
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text());parsed+=1
    for p in ROOT.rglob('*.json'):read(p);strict+=1
    provenance=read(ROOT/'references/project_provenance.json')
    for name,sha in provenance['inherited_files'].items():assert hashlib.sha256((ROOT/'parent_l42'/name).read_bytes()).hexdigest()==sha
    f=read(ROOT/'evidence/fitted/freeze_contract.json');bank=load_tree(ROOT/'evidence/fitted/frozen_bank');assert tree_hash(bank)==f['bank_sha256']
    sf=read(ROOT/'evidence/stability_fitted/freeze_contract.json');sb=load_tree(ROOT/'evidence/stability_fitted/frozen_bank');assert tree_hash(sb)==sf['bank_sha256']
    assert hashlib.sha256((RUN_DIRECTORY/'best.pt').read_bytes()).hexdigest()==f['checkpoint_sha256']
    state=read(ROOT/'evidence/state_interventions/report.json');edge=read(ROOT/'evidence/edge_interventions/report.json');stable=read(ROOT/'evidence/stability_followup/report.json')
    assert sum(len(v) for v in state['results'].values())==238
    assert sum(len(v) for v in edge['results'].values())==24
    assert sum(len(v) for v in stable['results'].values())==112
    total=means_check(ROOT/'evidence/state_interventions',state,'native_risks')+means_check(ROOT/'evidence/edge_interventions',edge,'hybrid_label_risks')+means_check(ROOT/'evidence/stability_followup',stable,'native_risks')
    assert total==1496
    decomposition_records=[z['operator_and_scope'] for report in (state,stable) for x in report['results'].values() for z in x.values()]
    relative_residuals=[]
    for d in decomposition_records:
        scale=1+sum(abs(d[k]) for k in ('operator_fidelity_MSE','scope_fidelity_MSE','interaction','natural_fidelity_MSE'))
        relative_residuals.append(d['identity_max_abs']/scale)
        assert d['identity_max_abs']<=1e-12*scale
        assert abs(d['risk_difference'])<=d['risk_difference_absolute_upper_bound']+1e-12*(1+abs(d['risk_difference']))
    assert all(v['native_noop_max_abs']==0 for v in state['no_op_identities'].values())
    assert state['phase_control']['after_final_step_zero_v_prediction_change_max_abs']==0
    assert state['phase_control']['before_G_zero_v_prediction_change_max_abs']>0
    assert all(v['max_abs']==0 for v in state['whole_snapshot_donor_controls'].values())
    assert edge['identities']['no_op_prediction_max_abs']==0
    dynamics=[v for x in state['results'].values() for z in x.values() for v in z['dynamics'].values()]
    assert all(v['flips_inside_strict_margin_bound']==0 for v in dynamics)
    assert max(v['max_excess_after_roundoff_allowance'] for v in dynamics)==0
    for key,conditions in edge['results'].items():
        for op,row in conditions.items():
            assert row['sender_state_change_max_abs']==row['actual_edge_arrival_max_abs']==0
            if op!='content_donor':assert row['per_step_spike_count_mismatch']==row['per_step_flip_count_mismatch']==0
            elif key.endswith('complete_suffix'):assert row['receiver_donor_replay_max_abs']==row['native_donor_replay_max_abs']==0
    for x in stable['results'].values():assert x['no_op']['native_noop_max_abs']==0
    ds=np.load(ROOT/'evidence/state_interventions/scene_losses.npz');de=np.load(ROOT/'evidence/edge_interventions/scene_losses.npz')
    for key in f['contexts']:
        for q in ('H1','C'):
            for control in ('random','normmatched'):
                base=key['key']+'::';a=ds[base+'leace::'+q+'::derived_classification'];b=np.mean([ds[base+control+'_'+str(i)+'::'+q+'::derived_classification'] for i in range(3)],axis=0)
                assert paired_bound(a,b,56)==state['classification_excess_damage_vs_control'][key['key']+'::'+q+'::'+control]
    for key in edge['results']:
        for q in ('H1','C'):
            a=de[key+'::content_donor::'+q+'::derived_classification'];b=np.mean([de[key+'::count_flip_matched_'+str(i)+'::'+q+'::derived_classification'] for i in range(3)],axis=0)
            assert paired_bound(b,a,12)==edge['classification_donor_gain_vs_count_flip_matched'][key+'::'+q]
    grid=read(ROOT/'evidence/interface_grid.json');assert grid['case_count']==240 and all(r['noop_max_abs']==0 and r['parameters_unchanged'] for r in grid['cases'])
    state_gaps=[];site_gaps=[];edge_gaps=[]
    if replay:
        model,cfg,_=load_model(RUN_DIRECTORY);a=fixed_batch(cfg.data,128,'development_test',9060000,0);b=fixed_batch(cfg.data,128,'development_test',9070000,0)
        av,_=baseline_sites(model,a);bv,_=baseline_sites(model,b);donors={s:hybrid(cfg.data,a,b,s) for s in ('H1','H4','C')}
        dv={s:baseline_sites(model,d)[0] for s,d in donors.items()};nv={s:baseline_sites(model,hybrid(cfg.data,a,b,s,zero=True))[0] for s in donors}
        expected=load_tree(ROOT/'evidence/state_interventions/replay_prefix')
        for c in f['contexts']:
            key,site,source,time,layer=(c[k] for k in ('key','site','source','time','layer'));h=av[site];er=bank['erasers'][key];j=SOURCES.index(source)
            operations={'no_op':h,'zero_v':np.zeros_like(h),'leace':apply_eraser(h,er['leace']),'orth':apply_eraser(h,er['orth']),'binary_mp':apply_eraser(h,er['binary_mp']),
                'donor_v':dv[source][site],'oracle_zero_v':nv[source][site],'transport_leace_matched':transport(h,dv[source][site],er['leace']),
                'transport_leace_unmatched':transport(h,bv[site],er['leace']),'transport_orth_unmatched':transport(h,bv[site],er['orth']),
                'paired_delta_oracle':h+(b['latents'][:,j]-a['latents'][:,j])@bank['delta_maps'][key]}
            for seed in range(3):operations['random_'+str(seed)]=apply_eraser(h,er['random_'+str(seed)]);operations['normmatched_'+str(seed)]=norm_matched(h,er['leace'],seed+4360000)
            for name,edited in operations.items():
                r=run_post_edit(model,a['x'],f'backbone.cells.{layer}',time,lambda raw,start:edited[start:start+len(raw)],keep_traces=False)
                saved=expected[key+'::'+name];state_gaps.append(float(np.max(np.abs(r['prediction'][:32]-saved['native_prediction_prefix']))));site_gaps.append(float(np.max(np.abs(r['site_after'][:32]-saved['site_after_prefix']))))
        ep=load_tree(ROOT/'evidence/edge_interventions/prediction_prefix');ca=capture_edge(model,a['x']);msg=ca['sender_output']
        for source,donor in donors.items():
            cd=capture_edge(model,donor['x']);target=cd['sender_output'];onset=int(a['event_times'][0,SOURCES.index(source)])
            for kind,window in [('write_only',[onset]),('complete_suffix',list(range(onset,16)))]:
                conditions={'content_donor':target,**{'count_flip_matched_'+str(seed):matched_binary_message(msg,target,436701+seed) for seed in range(3)}}
                for name,replacement in conditions.items():
                    r=capture_edge(model,a['x'],replacement,window);edge_gaps.append(float(np.max(np.abs(r['native_prediction'][:32]-ep[source+'__'+kind+'::'+name]))))
        assert max(state_gaps+site_gaps+edge_gaps)==0
    summary={'module':'L4.3 artifact verification','strict_json_files':strict,'python_AST_files':parsed,'inherited_files_verified':len(provenance['inherited_files']),
        'loss_arrays_recomputed':total,'initial_state_prefix_conditions_replayed':len(state_gaps),'edge_prefix_conditions_replayed':len(edge_gaps),
        'operator_scope_and_risk_bound_records_verified':len(decomposition_records),'max_scaled_identity_residual':max(relative_residuals),
        'prefix_native_prediction_max_abs':max(state_gaps+edge_gaps,default=0),'prefix_state_max_abs':max(site_gaps,default=0),
        'max_raw_real_arithmetic_envelope_excess':max(v['max_recurrence_bound_violation'] for v in dynamics),
        'max_envelope_excess_after_declared_FP32_allowance':max(v['max_excess_after_roundoff_allowance'] for v in dynamics),
        'no_op_snapshot_edge_and_binary_budgets_verified':True,'frozen_banks_and_checkpoint_verified':True,'state_classification_family_positive_lower_count':sum(v['interval'][0]>0 for v in state['classification_excess_damage_vs_control'].values()),
        'edge_classification_family_positive_lower_count':sum(v['interval'][0]>0 for v in edge['classification_donor_gain_vs_count_flip_matched'].values()),
        'scope':'Recomputed every saved scene-loss mean and bounded comparison; replayed first32 of all initial conditions using original batch128. Adaptive follow-up means/hash verified; its full native reruns are the evaluation log, not a second independent validation sample.'}
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--skip-replay',action='store_true');a=p.parse_args()
    r=verify(not a.skip_replay);write_json(a.output,r);print(json.dumps(r,indent=2))
