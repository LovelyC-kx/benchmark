"""Verify saved L3.4 v0.2 artifacts without refitting or accessing audit data."""
import ast,hashlib,json,re,sys,platform
from pathlib import Path
import numpy as np
import scipy
BASE=Path(__file__).resolve().parent
EXPECTED='f46effd40083643fc4d298b7390657488dd91b4fcb989c2fd13a1f3b35a8410e'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def strict(path):
    def reject(s):raise ValueError(f'Nonstandard JSON constant {s}')
    value=json.loads(path.read_text(),parse_constant=reject)
    def walk(x):
        if isinstance(x,float) and not np.isfinite(x):raise ValueError(f'Nonfinite JSON: {path}')
        if isinstance(x,dict):
            for v in x.values():walk(v)
        if isinstance(x,list):
            for v in x:walk(v)
    walk(value);return value
def verify(write_status=False):
    py=sorted(p for p in BASE.rglob('*.py') if '__pycache__' not in p.parts)
    for p in py:ast.parse(p.read_text(),filename=str(p))
    json_paths=sorted(BASE.rglob('*.json'))
    for p in json_paths:strict(p)
    checkpoint=BASE/'evidence/source_smoke_run/best.pt'
    assert sha(checkpoint)==EXPECTED
    root=BASE/'evidence/validated_v0_2_thirteen_views'
    summary=strict(root/'summary.json')
    assert summary['semantic_view_count']==13 and summary['formal_benchmark_completed']==0
    assert summary['formal_benchmark_planned']==435 and not summary['audit_test_accessed']
    runner_sha=sha(BASE/'run_focused_review.py')
    verified_weights=verified_losses=verified_plans=0
    stats={}
    for view in summary['views']:
        folder=root/view['folder'];report=strict(folder/'report.json')
        freeze=strict(folder/'selection_freeze.json')
        assert report['selection_freeze_sha256']==sha(folder/'selection_freeze.json')
        assert report['selection_frozen_before_evaluation_generation']
        assert report['evaluation_scene_first_id']==5600000 and report['evaluation_scenes']==8192
        assert not report['audit_test_accessed'] and report['new_SNN_optimizer_updates']==0
        with np.load(folder/'frozen_readers_and_scene_losses.npz',allow_pickle=False) as saved:
            for g in ('F','G','FG'):
                locked=freeze[g];cur=report['target_groups'][g]
                assert locked['runner_sha256']==runner_sha
                indices=saved[g+'__train_indices']
                assert hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==locked['draw_plan_sha256']
                np.testing.assert_array_equal(saved[g+'__theta'],locked['theta'])
                verified_plans+=1
                h=hashlib.sha256()
                for name,curves in sorted(cur['reader_results'].items()):
                    h.update(name.encode())
                    for p in sorted(map(int,curves)):
                        h.update(str(p).encode())
                        prefix=f'{g}__{name}__p{p}'
                        if name=='mlp':
                            states=sorted({int(re.search(r'__model(\d+)__',k)[1]) for k in saved.files if k.startswith(prefix+'__model')})
                            for j in states:
                                for key in ('w1','b1','w2','b2'):
                                    h.update(saved[f'{prefix}__model{j}__{key}'].tobytes());verified_weights+=1
                        else:
                            h.update(saved[prefix+'__weights'].tobytes());verified_weights+=1
                        loss=saved[prefix+'__scene_loss']
                        assert loss.shape==(8192,) and np.isfinite(loss).all()
                        assert loss.min()>=0 and loss.max()<=1
                        assert abs(float(loss.mean())-curves[str(p)]['mean_error'])<1e-12
                        verified_losses+=1
                assert h.hexdigest()==locked['weight_sha256']
                for selector,choices in locked['choices'].items():
                    for p,name in choices.items():
                        assert cur['selected_reader_gains'][selector][p]['selected_candidate']==name
                key=(report['focal'],report['observation'],report['schedule'],g)
                small={k:cur['four_statistics'].get(k) for k in ('PR','c','inv_f','inv_s')}
                if key in stats:assert small==stats[key]
                stats[key]=small
    continuous=BASE/'evidence/validated_v0_2_continuous_native'
    cs=strict(continuous/'summary.json');assert cs['semantic_views']==13
    assert cs['closed_form_regression_maps_fitted']==1248 and not cs['audit_test_accessed']
    continuous_weight_groups=0
    for item in cs['views']:
        folder=continuous/item['view'];r=strict(folder/'report.json')
        lock=strict(folder/'continuous_selection_freeze.json')
        assert r['continuous_evaluation_first_id']==5700000 and r['continuous_evaluation_scenes']==8192
        with np.load(folder/'continuous_scene_losses_and_weights.npz',allow_pickle=False) as arrays:
            for g in ('F','G','FG'):
                h=hashlib.sha256()
                for p in (1,8,32,128):
                    for kind in ('raw','ridge'):h.update(arrays[f'{g}__p{p}__{kind}_weights'].tobytes())
                    curve=r['groups'][g]['continuous_learning_curves'][str(p)]
                    for kind,method in (('raw','raw_Hebbian'),('gain','moment_gain_Hebbian'),('ridge','ridge')):
                        loss=arrays[f'{g}__p{p}__{kind}_scene_mse']
                        assert loss.shape==(8192,) and np.isfinite(loss).all() and loss.min()>=0
                        assert abs(float(loss.mean())-curve[method]['MSE_per_target'])<1e-10
                    assert curve['moment_selected_scalar_gain']==lock[g]['moment_selected_scalar_gain'][str(p)]
                    assert curve['gain_fitting_extra_labelled_scenes']==2048
                assert h.hexdigest()==lock[g]['weight_sha256']
                continuous_weight_groups+=1
    controls=strict(BASE/'evidence/validated_v0_2_higher_order_controls/summary.json')
    assert controls['checkpoint_sha256']==EXPECTED and not controls['audit_test_accessed']
    diagnostics=sum(len(x) for x in controls['SNN_role_diagnostics'].values())
    assert diagnostics==9
    log=(BASE/'evidence/test_log_v0_2.txt').read_text()
    assert re.search(r'Ran 55 tests',log) and log.rstrip().endswith('OK') and 'skipped=' not in log
    result={'date':'2026-10-25','module':'L3.4 v0.2',
        'verification_scope':'Saved-artifact consistency, AST and strict JSON; tests were executed separately and their saved log checked. No model refitting.',
        'unit_tests_passed':55,'unit_test_skips':0,'python_files_AST_checked':len(py),
        'strict_JSON_files_checked_before_status_write':len(json_paths),
        'classification_views':13,'target_group_views':39,'continuous_views':13,
        'classification_scene_loss_arrays_verified':verified_losses,
        'classification_weight_parameter_arrays_verified':verified_weights,
        'classification_training_plans_verified':verified_plans,
        'continuous_frozen_weight_groups_verified':continuous_weight_groups,
        'continuous_closed_form_maps_fitted':1248,
        'higher_order_control_diagnostics':diagnostics,
        'MLP_vectorized_optimizer_steps':7680,'MLP_individual_task_head_fits':1536,
        'MLP_scope':'C final-post, sphere prior, F/G/FG only',
        'checkpoint_sha256':EXPECTED,'checkpoint_unchanged':True,
        'source_original_optimizer_updates':1,'new_SNN_training_runs':0,'new_SNN_optimizer_updates':0,
        'audit_test_accessed':False,'formal_benchmark_completed':0,'formal_benchmark_planned':435,
        'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),
        'validated_PyTorch_version':'2.14.1+cpu','runtime':'CPU; user A40 not used',
        'primary_evidence':'evidence/validated_v0_2_thirteen_views',
        'historical_v0_1_results_are_not_independent_checkpoints':True}
    if write_status:(BASE/'module_status.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
if __name__=='__main__':print(json.dumps(verify(write_status='--write-status' in sys.argv),indent=2,allow_nan=False))

