"""Read-only frozen-head replay and protocol/evidence checks."""
import argparse,ast,hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from bridge35 import load_model,fixed_batch
from artifacts35 import load_tree,write_json
from temporal_probes import tree_hash,paired_bound
from run_access_review import collect,targets,losses
ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())
def verify():
    torch.set_num_threads(1)
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    for p in ROOT.rglob('*.json'):json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    root=ROOT/'evidence/frozen_access_review';r=read(root/'report.json');f=read(root/'freeze_contract.json');bank=load_tree(root/'frozen_readers')
    sha=hashlib.sha256((ROOT/'evidence/source_smoke_run/best.pt').read_bytes()).hexdigest()
    assert sha==f['checkpoint_sha256']=='f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e'
    assert tree_hash(bank)==f['reader_sha256'] and r['views']==15 and r['head_matrix_solves']==90 and r['risk_cells']==120
    with np.load(root/'constant_baseline_losses.npz',allow_pickle=False) as saved:
        assert len(saved)==8 and r['constant_baseline_risk_cells']==8
        for p in ('128','512'):
            for source in ('H1','C'):
                for objective in ('classification','MSE'):
                    a=saved[f'p{p}__{source}__{objective}']
                    assert a.shape==(r['evaluation_scenes'],) and abs(a.mean()-r['constant_reader_risks'][p][source][objective])<1e-12
    assert r['native_head_replay_max_abs']<1e-5 and r['parameters_unchanged']
    grid=read(ROOT/'evidence/interface_grid.json');assert len(grid['cases'])==1728 and grid['classification_cases']==1440 and grid['image_reconstruction_cases']==288
    for row in grid['cases']:
        assert not row['trained'] and row['parameters_unchanged']
        for key in ('carrier_to_features_max_abs','manual_GPR_prediction_max_abs','stream_vs_batch_prediction_max_abs'):assert row[key]<2e-5
    folded=read(ROOT/'evidence/folded_trajectory_grid.json');assert folded['case_count']==288 and len(folded['cases'])==288
    assert all(x['max_abs']<2e-5 and x['retained_trajectory_length']==0 and x['parameters_unchanged'] for x in folded['cases'])
    audit=read(ROOT/'evidence/selected_author_checks.json');assert audit['source_count']==5 and all(x['bytes_match'] for x in audit['verified_sources'].values())
    assert audit['TET_loss_lambda0_max_abs']==0 and audit['eval_mask_module_returns_input_unchanged']
    native=read(ROOT/'evidence/native_access_audit.json');assert len(native['cases'])==12 and native['same_original_parameters']
    assert sum(x['actual_trained_configuration'] for x in native['cases'])==1
    controls=read(ROOT/'evidence/known_access_controls.json')
    assert controls['adaptive_weighting_has_its_own_observation_permission']['target_recovered']
    integrated=read(ROOT/'evidence/integrated_pipeline_receipt.json');assert integrated['status']=='complete'
    assert len(integrated['stages'])==3 and all(x['exit_code']==0 for x in integrated['stages'])
    model,cfg,_=load_model(ROOT/'evidence/source_smoke_run');batch=fixed_batch(cfg.data,128,'development_test',8430000,0);views,_=collect(model,batch);y,_,groups=targets(batch,cfg.data.k)
    arrays=0;classification_gap=0.;mse_gap=0.
    with np.load(root/'scene_losses.npz',allow_pickle=False) as saved:
        assert len(saved)==120
        for name,x in views.items():
            x=np.asarray(x,float)/f['scales'][name]
            for p in (128,512):
                for g in groups:
                    for j,obj in enumerate(('classification','MSE')):
                        key=f'{name}__p{p}__{g["source"]}__{obj}';array=saved[key]
                        assert array.shape==(r['evaluation_scenes'],) and np.isfinite(array).all()
                        assert abs(array.mean()-r['risks'][name][str(p)][g['source']][obj])<1e-12
                        alpha=f['selection'][name][str(p)][g['source']][obj];value=losses(bank[name][str(p)][alpha],x,y,g)[j]
                        gap=float(np.max(np.abs(value-array[:128])));arrays+=1
                        if obj=='classification':classification_gap=max(classification_gap,gap)
                        else:mse_gap=max(mse_gap,gap)
        for key,c in r['classification_comparisons'].items():
            pair,p,source=key.rsplit('__',2);a,b=pair.split('_vs_');a=saved[f'{a}__{p}__{source}__classification'];b=saved[f'{b}__{p}__{source}__classification']
            assert paired_bound(a,b,r['comparison_family_size'])==c
    assert classification_gap==0 and mse_gap<1e-8
    log=(ROOT/'evidence/test_log.txt').read_text();m=re.search(r'Ran (\d+) tests',log)
    assert m and int(m.group(1))>=31 and log.rstrip().endswith('OK') and 'skipped=' not in log
    return {'module':'L4.1','unit_tests_passed':int(m.group(1)),'skipped':0,'untrained_interface_cases':1728,'folded_linear_trajectory_cases':288,
        'risk_cells':120,'constant_baseline_risk_cells':8,'saved_scene_arrays':arrays,'frozen_reader_hash_match':True,'native_head_replay_max_abs':r['native_head_replay_max_abs'],
        'replayed_scenes':128,'classification_scene_replay_max_abs':classification_gap,'MSE_scene_replay_max_abs':mse_gap,
        'checkpoint_sha256':sha,'pinned_author_files':5,'formal_training_completed':0,'audit_test_accessed':False,
        'scope':'Same saved development scenes replayed for consistency, not independent scientific confirmation.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    r=verify();write_json(out,r);print(json.dumps(r,indent=2))
