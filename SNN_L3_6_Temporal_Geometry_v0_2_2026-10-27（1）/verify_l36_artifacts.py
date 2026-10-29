"""Read-only evidence consistency and saved-reader prefix replay."""
import argparse,ast,json,hashlib,re
from pathlib import Path
import numpy as np
import torch
from bridge35 import load_model,fixed_batch,flatten_contents
from temporal_capture import base_capture,temporal_views
from temporal_probes import tree_hash,paired_bound
from run_temporal_review import target_bank,candidate_losses,map_digest
from artifacts35 import load_tree,write_json
ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())
def verify():
    torch.set_num_threads(1)
    py=list(ROOT.rglob('*.py'))
    for path in py:ast.parse(path.read_text(),filename=str(path))
    js=list(ROOT.rglob('*.json'))
    def invalid(x):raise ValueError(x)
    for path in js:json.loads(path.read_text(),parse_constant=invalid)
    primary=ROOT/'evidence/frozen_temporal_review';r=read(primary/'report.json');f=read(primary/'freeze_contract.json');bank=load_tree(primary/'frozen_readers')
    checkpoint_sha=hashlib.sha256((ROOT/'evidence/source_smoke_run/best.pt').read_bytes()).hexdigest()
    assert checkpoint_sha==f['checkpoint_sha256']=='f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e'
    assert tree_hash(bank)==f['reader_tree_sha256'] and r['all_map_hashes_replayed_exactly']
    with np.load(primary/'paired_calibration.npz',allow_pickle=False) as data:cal=temporal_views({k:data[k].copy() for k in data})
    digest,map_count,ranks=map_digest(cal,f['scales']);assert digest==f['alignment_full_bank_sha256'] and map_count==9216
    cells=0
    for name,families in r['risk_matrices'].items():
        for family,groups in families.items():
            for source,objects in groups.items():
                for objective,array in objects.items():
                    a=np.asarray(array);assert a.shape==(16,16) and np.isfinite(a).all() and a.min()>=0
                    if objective=='classification':assert a.max()<=1
                    if family!='raw':np.testing.assert_allclose(np.diag(a),np.diag(families['raw'][source][objective]),atol=1e-10)
                    cells+=a.size
    assert cells==92160
    arrays=0
    with np.load(primary/'selected_scene_losses.npz',allow_pickle=False) as saved:
        for name,groups in r['selected_reader_results'].items():
            for source,row in groups.items():
                a=saved[name+'__'+source+'__baseline'];b=saved[name+'__'+source+'__selected']
                assert a.shape==b.shape==(8192,)
                assert abs(a.mean()-row['baseline_error'])<1e-12 and abs(b.mean()-row['chosen_error'])<1e-12
                assert paired_bound(a,b,45)==row['paired_classification_bound'];arrays+=2
        model,cfg,_=load_model(ROOT/'evidence/source_smoke_run');batch=fixed_batch(cfg.data,128,'development_test',7630000,0)
        views=temporal_views(base_capture(model,batch['x']));y,_,groups=target_bank(flatten_contents(batch,cfg.data));maximum=0.
        for j,g in enumerate(groups):g['available_times']=f['groups'][j]['available_times']
        for name,x in views.items():
            scale=f['scales'][name];loss=candidate_losses(bank[name],np.asarray(cal[name],float)/scale,np.asarray(x,float)/scale,y,groups,15)
            for source,entries in loss.items():
                chosen=f['selection'][name][source]['chosen']
                for label,key in (('baseline','anchor_raw'),('selected',chosen)):
                    gap=float(np.max(np.abs(entries[key][0]-saved[name+'__'+source+'__'+label][:128])));maximum=max(maximum,gap)
        assert maximum==0
    ext=read(ROOT/'evidence/temporal_extensions/report.json');assert len(ext['source_clock_scan'])==360 and len(ext['finite_scale_task_metrics'])==12
    assert all(v['zero_metric_before_write'] for v in ext['finite_scale_task_metrics'])
    certificates=read(ROOT/'evidence/delay_reader_certificates.json');assert len(certificates['comparisons'])==900 and certificates['mean_risk_replay_max_abs']==0
    audit=read(ROOT/'evidence/selected_author_function_checks.json');assert audit['source_count']==8 and all(x['blob_bytes_match'] for x in audit['selected_sources'].values())
    assert audit['superposition_four_terms_max_abs']<1e-12 and audit['MNE_source_time_target_time_axes']['max_abs']==0
    mean_geometry=read(ROOT/'evidence/conditional_mean_geometry.json')
    assert mean_geometry['scenes']==8 and mean_geometry['repeats_per_content']==128 and len(mean_geometry['finite_scale_groups'])==8
    log=(ROOT/'evidence/test_log.txt').read_text();match=re.search(r'Ran (\d+) tests',log)
    assert match and int(match.group(1))>=64 and log.rstrip().endswith('OK') and 'skipped=' not in log
    tests=int(match.group(1))
    closure=ROOT/'evidence/temporal_consistency';cycle=read(closure/'report.json');cl=read(closure/'freeze_contract.json')
    assert tree_hash(load_tree(closure/'frozen_maps'))==cl['map_sha256']
    assert len(cycle['cycles'])==540 and len(cycle['equal_lag_transfers'])==54
    assert all(v['includes_centering_offsets'] for v in cycle['cycles'])
    phase=read(ROOT/'evidence/phase_readers/report.json');pl=read(ROOT/'evidence/phase_readers/freeze_contract.json')
    assert tree_hash(load_tree(ROOT/'evidence/phase_readers/frozen_readers'))==pl['reader_sha256']
    assert phase['family_size']==25 and phase['positive_simultaneous_lower_bounds']==0
    with np.load(ROOT/'evidence/phase_readers/scene_losses.npz',allow_pickle=False) as loss:
        for name,rows in phase['results'].items():
            for source,row in rows.items():
                a,b=loss[f'{name}__{source}__pooled'],loss[f'{name}__{source}__chosen']
                assert a.shape==b.shape==(phase['evaluation_scenes'],)
                assert paired_bound(a,b,phase['family_size'])==row['paired_bound_selected_vs_pooled']
    events=read(ROOT/'evidence/event_scale_scan/report.json');assert len(events['rows'])==576
    assert events['forward_captures']==585 and events['checkpoint_and_threshold_restored']
    assert all(v['pre_write_jacobian_max_abs']==0 for v in events['rows'])
    assert all('first_layer_any_time_scene_crossing_fraction' in v and 'scenes_with_nonzero_metric' in v and 0<=v['observer_clipping_fraction']<=1 for v in events['rows'])
    interfaces=read(ROOT/'evidence/model_interfaces.json');assert len(interfaces['cases'])==120
    assert all(v['prediction_max_abs']==0 and not v['trained'] for v in interfaces['cases'])
    new_cert=read(ROOT/'evidence/delay_reader_certificates_v02.json')
    assert new_cert['checkpoint_sha256']==checkpoint_sha and new_cert['comparisons']==certificates['comparisons']
    integration=read(ROOT/'evidence/integrated_pipeline_receipt.json');custom=read(ROOT/'evidence/custom_pipeline_receipt.json')
    assert integration['status']==custom['status']=='complete' and integration['stage_count']==8 and custom['stage_count']==9
    assert all(v['exit_code']==0 for r in (integration,custom) for v in r['stages'])
    assert custom['primary_times']==20 and custom['primary_views']==7 and custom['source_coordinates']==1 and custom['delay_certified_comparisons']==700
    assert custom['checkpoint_sha256']!=checkpoint_sha and custom['source_optimizer_updates']==0
    assert r['new_SNN_optimizer_updates']==0 and r['formal_benchmark_completed']==0 and not r['audit_test_accessed']
    return {'module':'L3.6','version':'0.2','unit_tests_passed':tests,'skipped':0,'python_AST_files':len(py),'strict_JSON_files_before_verification_write':len(js),
        'offset_aware_cycle_rows':540,'equal_lag_transfer_rows':54,'phase_comparisons':25,'phase_positive_lower_bounds':0,
        'event_scale_noise_threshold_precision_rows':576,'untrained_model_interface_cases':120,
        'integrated_reproduction_stages_passed':8,'custom_untrained_integration_stages_passed':9,
        'risk_matrix_cells_checked':cells,'selected_scene_arrays_checked':arrays,'maps_reconstructed_from_frozen_calibration':map_count,
        'map_hash_match':True,'saved_reader_replay_scenes':128,'saved_reader_replay_arrays':90,'saved_reader_replay_max_abs':maximum,
        'checkpoint_sha256':checkpoint_sha,'checkpoint_unchanged':True,'selected_author_blobs':8,
        'new_SNN_training_runs':0,'formal_benchmark_completed':0,'formal_benchmark_planned':435,'audit_test_accessed':False,
        'scope':'Saved-artifact verification and exact replay of the same first scenes; no refitting of readers and no independent statistical confirmation.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();r=verify()
    if a.output:
        path=Path(a.output)
        if path.exists():raise FileExistsError(path)
        write_json(path,r)
    print(json.dumps(r,indent=2))
