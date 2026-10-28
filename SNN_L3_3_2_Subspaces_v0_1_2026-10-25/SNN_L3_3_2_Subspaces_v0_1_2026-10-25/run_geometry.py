"""L3.3.2: independent coding geometry and declared-readout consequences.

One checkpoint is analyzed; no SNN training or audit access. All task losses
retain every originally requested coordinate. Context-specific axes are fitted
and selected without using development evaluation outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from bridge import (load_model, collect, coordinate_roles, flatten_contents,
    fit_joint, representation_score, role_risk_budget, repeated_signal_and_noise,
    repeated_prediction_risk, projected_variance)
from subspace_geometry import (ROLES, basis, complement, projector, overlap,
    fit_axes, independent_replica, covariance_capture, null_overlaps, random_basis,
    fixed_rank_bootstrap_overlap, variance_matched_basis)
from readout_geometry import (covariance, choose_covariance_metric, noise_metric_geometry,
    mapping_diagnostics, affine_restrict, fit_projected_probe, predict_raw,
    metrics, paired_error_interval)
from sampling import SCHEDULES, fixed_batch, repeat_panel


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [serial(item) for item in value]
    return value


def head_for_observation(model, columns, focal, observation):
    selected = [row for row in columns if row['role'] == 'focal']
    head_rows = [row['source_id']*4+row['coordinate'] for row in selected]
    w = model.core.head.weight.detach().cpu().numpy()[head_rows].astype(float)
    b = model.core.head.bias.detach().cpu().numpy()[head_rows].astype(float)
    if observation == 'joint_post': w = np.concatenate((np.zeros_like(w), w), axis=1)
    return {'w': w, 'b': b, 'observation': observation, 'focal': focal}


def collect_repeat(model, panel, observation):
    x = panel['x']; g, repeats = x.shape[:2]
    features = collect(model, x.reshape(g*repeats, *x.shape[2:]))[observation]
    return features.astype(float).reshape(g, repeats, -1)


def projection_spaces(q_f, axes, q_replica, natural_cov, null_readers, pool, seed):
    dim, rank = q_f.shape
    spaces = {'full': np.eye(dim), 'focal_encoding': q_f,
              'focal_encoding_replica': q_replica['focal'],
              'focal_complement': complement(q_f),
              'cotarget_encoding': axes['cotarget']['q'],
              'unrequested_encoding': axes['unrequested']['q']}
    result = {'target_dimension': rank, 'random_reader_count': 0,
              'variance_match_scope': 'Selection covariance only; no labels or test losses used to choose the nearest random projection.'}
    if rank == 0:
        result['status'] = 'no_nonzero_selected_focal_subspace'; return spaces, result
    rng = np.random.default_rng(seed)
    for i in range(null_readers): spaces['haar_random_'+str(i).zfill(2)] = random_basis(dim, rank, rng)
    target_power = covariance_capture(q_f, natural_cov)['captured_power']
    candidates = []
    for _ in range(pool):
        q = random_basis(dim, rank, rng)
        power = covariance_capture(q, natural_cov)['captured_power']
        candidates.append((abs(power-target_power), power, q))
    difference, power, q = min(candidates, key=lambda item: item[0])
    relative_gap = difference/max(abs(target_power), 1e-30)
    spaces['variance_nearest_random'] = q
    constructed = variance_matched_basis(natural_cov, rank, target_power, rng)
    spaces['variance_constructed_control'] = constructed
    constructed_power = covariance_capture(constructed, natural_cov)['captured_power']
    result.update(status='computed', random_reader_count=null_readers,
                  variance_candidate_pool=pool, target_retained_power=target_power,
                  nearest_random_power=power, relative_power_gap=relative_gap,
                  matched_within_10_percent=relative_gap <= .1,
                  constructed_control_power=constructed_power,
                  constructed_absolute_power_gap=abs(constructed_power-target_power),
                  constructed_control_method='Random rotations within low/high covariance eigenspaces with a scalar variance-matching mixture; complement if needed.',
                  warning='A failed nearest-Haar match remains unmatched. The separate constructed control matches rank and total retained variance but is not a uniform conditional null and does not match the full signal/noise spectrum.')
    return spaces, result


def analyze_context(model, run, metadata, context, focal, observation, smoke):
    cfg = run.data
    if run.objective != 'regression' or cfg.extra_writes != 3:
        raise ValueError('This initial analysis requires regression with every source present')
    nfit, nselect, neval = (128, 64, 128) if smoke else (1024, 512, 1024)
    scenes, select_scenes, repeats = (32, 16, 4) if smoke else (256, 128, 8)
    bootstrap_draws, null_draws = (32, 64) if smoke else (200, 1024)
    null_readers, random_pool = (4, 64) if smoke else (16, 512)
    specs = [('axis_fit', 'geometry_fit', 2000000, nfit),
             ('axis_selection', 'geometry_fit', 2100000, nselect),
             ('axis_replica', 'geometry_fit', 2200000, nfit),
             ('linear_remainder_fit', 'geometry_fit', 2300000, nfit),
             ('probe_fit', 'probe_train', 2600000, nfit),
             ('probe_selection', 'probe_val', 2700000, nselect),
             ('evaluation', 'development_test', 2800000, neval)]
    batches = {name: fixed_batch(cfg, count, split, first_id, context)
               for name, split, first_id, count in specs}
    features = {name: collect(model, batch['x']) for name, batch in batches.items()}
    r = {name: values[observation].astype(float) for name, values in features.items()}
    z = {name: flatten_contents(batch, cfg) for name, batch in batches.items()}
    columns = coordinate_roles(cfg, focal, batches['evaluation']['source_present'])
    roles = np.array([row['role'] for row in columns])
    fit, axes = fit_axes(r['axis_fit'], z['axis_fit'], roles, r['axis_selection'], z['axis_selection'])
    replica, q_replica, stability = independent_replica(fit, axes, r['axis_replica'], z['axis_replica'], roles)
    mean = fit['rmean']; dim = len(mean)
    scale = float(np.sqrt(np.trace(covariance(r['axis_fit']))/dim))
    scale_degenerate = scale == 0
    if scale_degenerate: scale = 1.
    natural_cov = covariance(r['axis_selection'])
    panel_fit = repeat_panel(cfg, scenes, repeats, 'geometry_fit', 2400000, context, focal)
    panel_select = repeat_panel(cfg, select_scenes, repeats, 'geometry_fit', 2500000, context, focal)
    panel_eval = repeat_panel(cfg, scenes, repeats, 'development_test', 2900000, context, focal)
    rf = collect_repeat(model, panel_fit, observation)
    rs = collect_repeat(model, panel_select, observation)
    re = collect_repeat(model, panel_eval, observation)
    covariance_fit = repeated_signal_and_noise(rf)
    covariance_select = repeated_signal_and_noise(rs)
    covariance_eval = repeated_signal_and_noise(re)
    noise = choose_covariance_metric(covariance_fit['within_covariance'], covariance_select['within_covariance'])
    independent_overlaps, same_fit_overlaps, power_capture, nulls = {}, {}, {}, {}
    for other in ('cotarget', 'unrequested'):
        independent_overlaps[other] = {'focal_a_vs_other_b': overlap(axes['focal']['q'], q_replica[other]),
                                      'focal_b_vs_other_a': overlap(q_replica['focal'], axes[other]['q'])}
        same_fit_overlaps[other] = overlap(axes['focal']['q'], axes[other]['q'])
        a_other_b = replica['a'][:, roles == other]
        power_capture[other] = covariance_capture(axes['focal']['q'], a_other_b @ a_other_b.T)
        nulls[other] = null_overlaps(axes['focal']['q'], q_replica[other].shape[1], natural_cov,
                                     draws=null_draws, seed=1481+(other == 'unrequested'))
    boot = fixed_rank_bootstrap_overlap(r['axis_fit'], z['axis_fit'], r['axis_replica'],
        z['axis_replica'], roles, axes['focal']['q'].shape[1], q_replica['cotarget'].shape[1],
        draws=bootstrap_draws, seed=1483)
    selected_af = projector(axes['focal']['q']) @ axes['focal']['a']
    selected_ag_b = projector(q_replica['cotarget']) @ replica['a'][:, roles == 'cotarget']
    whitened = noise_metric_geometry(selected_af, selected_ag_b, noise['regularized_covariance']) if noise['status'] == 'computed' else {'status': noise['status']}
    full_focal_map = fit['a'][:, roles == 'focal']
    linear_remainder = r['linear_remainder_fit']-mean-(z['linear_remainder_fit']-fit['zmean'])[:, roles == 'focal'] @ full_focal_map.T
    linear_remainder_cov = covariance(linear_remainder)
    spaces, random_metadata = projection_spaces(axes['focal']['q'], axes, q_replica, natural_cov,
                                                 null_readers, random_pool, seed=1484)
    native = head_for_observation(model, columns, focal, observation)
    head_rows = [row['source_id']*4+row['coordinate'] for row in columns if row['role'] == 'focal']
    native_prediction = predict_raw(native, r['evaluation'])
    export_prediction = features['evaluation']['prediction'][:, head_rows]
    native_error = float(np.max(np.abs(native_prediction-export_prediction)))
    if native_error > 1e-6: raise AssertionError('Observation-to-original-head embedding does not match original predictions')
    learning_sizes = (32, 128) if smoke else (64, 256, 1024)
    probe_fits, probe_reports, projected_arrays = {}, {}, {}
    for group in ('focal', 'cotarget'):
        target_columns = np.flatnonzero(roles == group)
        if not len(target_columns): continue
        yt = z['probe_fit'][:, target_columns]; yv = z['probe_selection'][:, target_columns]
        ye = z['evaluation'][:, target_columns]
        group_rows = {}
        for name, q in spaces.items():
            rows = []
            for count in learning_sizes:
                probe = fit_projected_probe(r['probe_fit'][:count], yt[:count], r['probe_selection'], yv, q, mean, scale)
                prediction = predict_raw(probe, r['evaluation'])
                rows.append({'train_count': count, 'selected_lambda': probe['selected_lambda'],
                    'alpha': probe['alpha'], 'observed_dimension': q.shape[1],
                    'readout_parameter_count': probe['readout_parameter_count'],
                    'metrics': metrics(prediction, ye)})
                probe_fits[group, name, count] = probe
                projected_arrays[group+'__'+name+'__n'+str(count)] = prediction
                baseline = projected_arrays[group+'__full__n'+str(count)]
                rows[-1]['paired_change_from_same_n_full'] = paired_error_interval(baseline, prediction, ye,
                    draws=128 if smoke else 1000, seed=1485)
            group_rows[name] = rows
        probe_reports[group] = group_rows
    full_probe = probe_fits['focal', 'full', learning_sizes[-1]]
    readers = {'original_task_head': native, 'natural_full_ridge_probe': full_probe}
    fixed_reader_restrictions, mappings, risk_budgets, conditional_risk = {}, {}, {}, {}
    target = z['evaluation'][:, roles == 'focal']
    for reader_name, reader in readers.items():
        w, b = reader['w'], reader['b']
        mappings[reader_name] = mapping_diagnostics(w, full_focal_map,
            fit['a'][:, roles == 'cotarget'], fit['a'][:, roles == 'unrequested'], covariance_eval['within_covariance'])
        risk_budgets[reader_name] = role_risk_budget(r['evaluation'], z['evaluation'], fit['a'], fit['zmean'], mean, roles, w, b)
        base_prediction = predict_raw(reader, r['evaluation'])
        restricted_rows = {}
        for name, q in spaces.items():
            p = predict_raw(reader, affine_restrict(r['evaluation'], mean, q))
            restricted_rows[name] = {'observed_dimension': q.shape[1], 'metrics': metrics(p, target),
                                    'paired_change': paired_error_interval(base_prediction, p, target,
                                        draws=128 if smoke else 1000, seed=1486)}
        fixed_reader_restrictions[reader_name] = restricted_rows
        prediction = re @ w.T+b
        conditional_risk[reader_name] = repeated_prediction_risk(prediction, panel_eval['targets'])
        conditional_risk[reader_name]['projected_within_covariance_mse'] = projected_variance(w, covariance_eval['within_covariance'])
    # The focal-restricted refitted probe has its own fixed raw-space map.
    restricted_probe = probe_fits['focal', 'focal_encoding', learning_sizes[-1]]
    conditional_risk['refitted_focal_encoding_probe'] = repeated_prediction_risk(re @ restricted_probe['w'].T+restricted_probe['b'], panel_eval['targets'])
    encoding_prediction = mean+(z['evaluation']-fit['zmean']) @ fit['a'].T
    role_metadata = {role: {'selected_rank': axes[role]['selection']['selected_rank'],
        'numeric_rank': axes[role]['selection']['numeric_rank'], 'rank_selection': axes[role]['selection'],
        'singular_values': axes[role]['singular_values'],
        'condition_on_nonzero_modes': axes[role]['condition_on_nonzero_modes'],
        'full_column_rank': axes[role]['full_column_rank'],
        'selected_boundary_relative_singular_gap': axes[role]['selected_boundary_relative_singular_gap'],
        'selected_cut_near_degenerate_block': axes[role]['selected_cut_near_degenerate_block']} for role in ROLES}
    result = {'module': 'L3.3.2', 'version': '0.1', 'date': '2026-10-25',
        'analysis_kind': 'pipeline_smoke' if smoke else 'development_analysis',
        'source_checkpoint_id': metadata['effective_id'], 'source_run_kind': metadata['run_kind'],
        'focal': focal, 'observation': observation, 'observation_dimension': dim,
        'schedule_index': context, 'source_event_times': SCHEDULES[context].tolist(),
        'time_index': cfg.deadline_steps-1, 'coordinate_roles': columns,
        'sampling_contract': [{'purpose': name, 'split': split, 'first_id': first_id, 'independent_scene_count': count}
                             for name, split, first_id, count in specs],
        'repeat_contract': {'fit': [2400000, scenes, repeats], 'selection': [2500000, select_scenes, repeats],
                            'evaluation': [2900000, scenes, repeats], 'conditioning': panel_eval['conditioning'],
                            'scene_repeats_not_counted_as_independent_scenes': True},
        'common_full_observation_scale': scale, 'observation_scale_was_degenerate': scale_degenerate,
        'axis_fit': role_metadata, 'independent_replica_stability': stability,
        'independent_fit_subspace_overlaps': independent_overlaps, 'same_fit_subspace_overlaps_secondary': same_fit_overlaps,
        'independent_other_encoding_power_in_focal_subspace': power_capture,
        'null_reference_distributions': nulls, 'fixed_selected_rank_bootstrap_focal_cotarget': boot,
        'noise_metric_selection': {key: value for key, value in noise.items() if key not in ('regularized_covariance','metric','inverse_sqrt')},
        'conditional_covariance_metric_geometry': whitened,
        'conditional_covariance_trace': float(np.trace(covariance_eval['within_covariance'])),
        'linear_focal_remainder_covariance_trace': float(np.trace(linear_remainder_cov)),
        'different_covariance_objects_not_interchangeable': True,
        'random_projection_controls': random_metadata,
        'linear_encoding_holdout': representation_score(r['evaluation'], encoding_prediction, mean),
        'original_head_embedding_max_error': native_error,
        'readout_mapping_diagnostics': mappings, 'exact_heldout_role_risk': risk_budgets,
        'fixed_reader_observation_restrictions': fixed_reader_restrictions,
        'refitted_probe_learning_curves': probe_reports, 'conditional_prediction_risk': conditional_risk,
        'audit_test_accessed': False, 'new_snn_training_runs': 0, 'new_optimizer_updates': 0,
        'interpretation': 'Independent-fit descriptive geometry, observation restrictions and held-out readout diagnostics. No anatomical partition, nonlinear information-absence proof, SNN Bayes bound, causal mediation or architectural ranking.'}
    arrays = {'encoding_a': fit['a'], 'encoding_replica_a': replica['a'], 'encoding_mean': mean,
              'encoding_zmean': fit['zmean'], 'evaluation_r': r['evaluation'], 'evaluation_z': z['evaluation'],
              'evaluation_scene_ids': batches['evaluation']['sample_ids'],
              'conditional_within_covariance_fit': covariance_fit['within_covariance'],
              'conditional_within_covariance_evaluation': covariance_eval['within_covariance'],
              'linear_focal_remainder_covariance': linear_remainder_cov,
              'focal_repeat_responses': re, 'focal_repeat_targets': panel_eval['targets']}
    arrays.update({'space__'+name: q for name, q in spaces.items()})
    arrays.update({'prediction__'+name: value for name, value in projected_arrays.items()})
    arrays.update({'reader__'+name+'__'+field: reader[field] for name, reader in readers.items() for field in ('w', 'b')})
    if noise['status'] == 'computed':
        arrays['regularized_conditional_covariance'] = noise['regularized_covariance']
        arrays['conditional_metric'] = noise['metric']
    return result, arrays


def run(root, output, schedules=(0,), focal='C', observation='last_post', smoke=False, device='cpu'):
    import platform
    import torch
    root, output = Path(root), Path(output)
    if output.exists(): raise FileExistsError('Choose a new derived analysis output directory')
    if observation not in ('last_post', 'joint_post'): raise ValueError('Initial original-head-compatible observations are last_post and joint_post')
    model, run_config, metadata = load_model(root, device)
    output.mkdir(parents=True)
    contexts, arrays = {}, {}
    for context in schedules:
        report, arr = analyze_context(model, run_config, metadata, context, focal, observation, smoke)
        folder = output/('schedule_'+str(context)); folder.mkdir()
        (folder/'report.json').write_text(json.dumps(serial(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        np.savez_compressed(folder/'geometry_arrays.npz', **arr)
        contexts[context], arrays[context] = report, arr
    cross = []
    for left in schedules:
        for right in schedules:
            if left >= right: continue
            # All schedules share latent/noise scene draws and model identity.
            if not np.array_equal(arrays[left]['evaluation_z'], arrays[right]['evaluation_z']):
                raise AssertionError('Cross-schedule contents are not paired')
            for role in ROLES:
                cross.append({'left_schedule': left, 'right_schedule': right, 'role': role,
                              'overlap': overlap(arrays[left]['space__'+role+'_encoding'], arrays[right]['space__'+role+'_encoding'])})
    result = {'module': 'L3.3.2', 'version': '0.1', 'source_checkpoint_id': metadata['effective_id'],
        'checkpoint_sha256': hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
        'source_run_kind': metadata['run_kind'], 'source_original_optimizer_updates': metadata['optimizer_updates'],
        'source_config': metadata['config'],
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': torch.__version__, 'device': str(next(model.parameters()).device),
                    'torch_cpu_threads': torch.get_num_threads()},
        'focal': focal, 'observation': observation, 'schedule_indices': list(schedules),
        'schedules_are_views_of_one_checkpoint': True, 'paired_context_subspace_descriptions': cross,
        'selected_ranks_by_schedule': {str(i): {role: contexts[i]['axis_fit'][role]['selected_rank'] for role in ROLES} for i in schedules},
        'audit_test_accessed': False, 'new_snn_training_runs': 0, 'new_optimizer_updates': 0,
        'formal_benchmark_results': False}
    (output/'summary.json').write_text(json.dumps(serial(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--schedule-index', type=int, choices=range(6))
    group.add_argument('--all-schedules', action='store_true')
    parser.add_argument('--focal', choices=('C', 'H1'), default='C')
    parser.add_argument('--observation', choices=('last_post', 'joint_post'), default='last_post')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    schedules = tuple(range(6)) if args.all_schedules else (0 if args.schedule_index is None else args.schedule_index,)
    print(json.dumps(run(args.run_directory, args.output, schedules, args.focal,
                         args.observation, args.smoke, args.device), indent=2))
