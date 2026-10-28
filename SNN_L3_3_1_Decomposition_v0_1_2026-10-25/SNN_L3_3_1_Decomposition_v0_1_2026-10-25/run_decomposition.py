"""Run L3.3.1 on an existing L3.2 checkpoint; no model training or audit access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from inherited import SOURCES, generate, load_model, collect
from roles import coordinate_roles, flatten_contents
from decomposition import fit_joint, apply_joint, covariance_accounting, representation_score, encoding_diagnostics
from repeat_panels import MODES, repeat_panel, within_scene_covariance


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serial(v) for v in x]
    return x


def run_analysis(root, output, focal='C', observation='last_post', device='cpu', smoke=False):
    root, output = Path(root), Path(output)
    if output.exists(): raise FileExistsError('Derived analysis output exists; choose a new path')
    model, run, metadata = load_model(root, device)
    if run.objective != 'regression':
        raise ValueError('Initial roles refer to requested continuous coordinates; classification needs a task-direction/orthogonal-content contract first')
    loaded = {name: np.load(root/(name+'.npz')) for name in ('geometry_fit', 'development_test')}
    observations = ('last_post', 'joint_post', 'last_pre', 'last_output')
    if observation not in observations: raise ValueError('Known observation required')
    matrices, contents, coordinates = {}, {}, None
    for split, arrays in loaded.items():
        count = len(arrays[observation])
        if smoke: count = min(count, 128 if split == 'geometry_fit' else 32)
        batch = generate(run.data, count, split)
        cols = coordinate_roles(run.data, focal, batch['source_present'])
        truth = flatten_contents(batch, run.data)
        if not np.array_equal(truth, arrays['targets'][:count]):
            raise ValueError('Export row/latent identity does not match the pinned generator')
        if not np.array_equal(batch['source_present'], arrays['source_present'][:count]):
            raise ValueError('Source-presence identity mismatch')
        coordinates = cols
        matrices[split] = arrays[observation][:count].astype(float)
        contents[split] = truth
    roles = [row['role'] for row in coordinates]
    # Verify regenerated observations before combining old exports and fresh panels.
    check_n = min(16, len(contents['development_test']))
    check_batch = generate(run.data, check_n, 'development_test')
    regenerated = collect(model, check_batch['x'][:, :run.data.deadline_steps])[observation]
    exported = matrices['development_test'][:check_n]
    export_match_error = float(np.max(np.abs(regenerated-exported)))
    if not np.allclose(regenerated, exported, atol=1e-5, rtol=1e-4):
        raise ValueError('Regenerated features differ from the recorded export; check runtime, generator and checkpoint identity')
    for arrays in loaded.values(): arrays.close()
    source_ids = [row['source_id'] for row in coordinates]
    n = len(contents['geometry_fit'])
    nfit = max(2, (2*n//3//6)*6)
    if n-nfit < 2: raise ValueError('Insufficient geometry rows for disjoint fit/selection')
    z, r = contents['geometry_fit'], matrices['geometry_fit']
    fit = fit_joint(r[:nfit], z[:nfit], roles)
    ez, er = contents['development_test'], matrices['development_test']
    components = apply_joint(fit, er, ez)
    reconstructed = fit['rmean']+sum(components.values())
    if not np.allclose(reconstructed, er, atol=1e-10): raise AssertionError('Pointwise reconstruction')
    cov = covariance_accounting(er, components)
    linear_prediction = er-components['unexplained_by_joint_linear_model']
    remainder = sum(components[name] for name in ('cotarget', 'unrequested', 'unexplained_by_joint_linear_model'))
    focal_remainder_cov = np.cov(remainder, rowvar=False)
    diagnostics = encoding_diagnostics((z[:nfit], r[:nfit]), (z[nfit:], r[nfit:]), (ez, er), source_ids)
    output.mkdir(parents=True)
    arrays_to_save = {'encoding_a': fit['a'], 'encoding_zmean': fit['zmean'], 'encoding_rmean': fit['rmean'],
                      'evaluation_representation': er, 'evaluation_contents': ez,
                      'linear_focal_remainder_covariance': focal_remainder_cov}
    arrays_to_save.update({'component__'+key: value for key, value in components.items()})
    arrays_to_save.update({'covariance__'+key: value for key, value in cov['component_covariances'].items()})
    arrays_to_save.update({'cross__'+key: value for key, value in cov['cross_covariances'].items()})
    panels = []
    scenes, repeats = (12, 4) if smoke else (256, 8)
    for mode in MODES:
        panel = repeat_panel(run.data, scenes, repeats, focal=focal, mode=mode)
        x = panel['x'][:, :, :run.data.deadline_steps]
        features = collect(model, x.reshape(scenes*repeats, *x.shape[2:]))[observation]
        responses = features.reshape(scenes, repeats, -1)
        estimate = within_scene_covariance(responses)
        if mode == 'identical_input' and float(np.max(np.abs(responses-responses[:, :1]))) > 1e-6:
            raise AssertionError('Identical-input deterministic replay failed')
        arrays_to_save['repeat_covariance__'+mode] = estimate['covariance']
        panels.append({'mode': mode, 'conditioned_on': panel['conditioned_on'],
            'schedule_fixed_within_scene': True, 'scene_count': scenes, 'repeat_count': repeats,
            'average_within_scene_trace': estimate['trace'],
            'meaning': {'identical_input':'deterministic replay control',
                        'measurement_only':'measurement-induced variability with all content and schedule fixed',
                        'focal_fixed_full':'nonfocal-content plus measurement variability with focal requested coordinates and schedule fixed'}[mode],
            'not_automatically_harmful_interference': True,
            'scene_mean_is_not_noiseless': mode != 'identical_input'})
    np.savez_compressed(output/'decomposition_arrays.npz', **arrays_to_save)
    result = {'module': 'L3.3.1', 'version': '0.1', 'date': '2026-10-25',
        'analysis_kind': 'pipeline_smoke' if smoke else 'development_analysis',
        'source_run_id': metadata['effective_id'], 'source_run_kind': metadata['run_kind'],
        'source_checkpoint_sha256': hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
        'regenerated_export_check': {'rows': check_n, 'max_abs_error': export_match_error, 'atol': 1e-5, 'rtol': 1e-4},
        'cell': run.cell, 'objective': run.objective, 'observation': observation, 'focal': focal,
        'time_index': run.data.deadline_steps-1, 'data_config': metadata['config']['data'],
        'coordinate_roles': coordinates, 'fit_rows': nfit, 'selection_rows': n-nfit,
        'evaluation_rows': len(er), 'fit_split': 'geometry_fit first rows',
        'selection_split': 'geometry_fit disjoint remaining rows', 'evaluation_split': 'development_test',
        'design_rank': fit['design_rank'], 'design_columns': len(roles),
        'coefficients_identifiable': fit['coefficients_identifiable'],
        'joint_linear_representation_score': representation_score(er, linear_prediction, fit['rmean']),
        'component_trace': cov['component_trace'], 'cross_trace': cov['cross_trace'],
        'max_pointwise_identity_error': float(np.max(np.abs(reconstructed-er))),
        'max_covariance_identity_error': cov['max_covariance_identity_error'],
        'linear_focal_remainder_trace': float(np.trace(focal_remainder_cov)),
        'focal_remainder_scope': 'after subtracting only the fitted linear focal component; can contain nonlinear focal signal',
        'encoding_family_diagnostics': diagnostics, 'repeated_panels': panels,
        'old_geometry_noise_fields_not_called': True, 'audit_test_accessed': False,
        'new_optimizer_updates': 0,
        'interpretation': 'Representation encoding/decomposition diagnostics on a fixed checkpoint. No subspace-harm claim, causal mediation, full functional ANOVA, or new benchmark-performance result.'}
    (output/'report.json').write_text(json.dumps(serial(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--focal', choices=SOURCES, default='C')
    parser.add_argument('--observation', choices=['last_post', 'joint_post', 'last_pre', 'last_output'], default='last_post')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    result = run_analysis(args.run_directory, args.output, args.focal, args.observation, args.device, args.smoke)
    print(json.dumps({key: result[key] for key in ('module', 'source_run_kind', 'fit_rows',
        'selection_rows', 'evaluation_rows', 'max_pointwise_identity_error', 'max_covariance_identity_error',
        'audit_test_accessed', 'new_optimizer_updates')}, indent=2))
