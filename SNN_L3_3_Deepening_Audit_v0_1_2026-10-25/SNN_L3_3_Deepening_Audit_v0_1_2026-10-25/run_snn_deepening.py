"""Development-only extensions on a pinned existing SNN checkpoint.

No SNN training, no audit access. Regression/full-presence/terminal-last-post
contract only. Fitting diagnostic encoders/probes is explicitly recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent/'parent_l331'))
from inherited import load_model, collect, generate, _render
from roles import coordinate_roles, flatten_contents
from decomposition import fit_joint
from repeat_panels import repeat_panel
from linear_reference import fit_ridge, predict_ridge, risk_accounting
from deep_diagnostics import (role_risk_budget, pickfreeze_statistics,
    repeated_signal_and_noise, repeated_prediction_risk, projected_variance)


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [serial(item) for item in value]
    return value


def fixed_schedule_batch(cfg, n, split, first_id):
    """Draw fresh contents/noise, then condition explicitly on one legal order.

    Equal sample_id modulo 6 does NOT fix the parent's order: each six-row block
    has its own shuffled permutation. We explicitly render the declared order.
    """
    batch = generate(cfg, n, split, sample_ids=np.arange(first_id, first_id+n))
    coordinate_roles(cfg, 'C', batch['source_present'])
    batch['event_times'] = np.tile([2, 5, 8, 11, 12], (n, 1))
    batch['x'] = _render(cfg, batch['latents'], batch['measurement_noise'],
                         batch['event_times'], batch['source_present'])
    batch['meta'] = dict(batch['meta'], analysis_condition='fixed legal order H1,H2,H3,H4,C')
    return batch


def pickfreeze_inputs(cfg, batch, columns):
    """Role groups and measurement noise are explicit independent factors."""
    z_a, noise_a = batch['latents'], batch['measurement_noise']
    z_b, noise_b = np.empty_like(z_a), np.empty_like(noise_a)
    for i, scene_id in enumerate(batch['sample_ids']):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.data_seed, int(scene_id), 992741]))
        z_b[i] = rng.normal(size=(5, 4))
        noise_b[i] = cfg.sigma*rng.normal(size=(5, 8))
    def render(z, noise):
        return _render(cfg, z, noise, batch['event_times'], batch['source_present'])[:, :cfg.deadline_steps]
    values = {'a': render(z_a, noise_a), 'b': render(z_b, noise_b)}
    metadata = {}
    for role in ('focal', 'cotarget', 'unrequested', 'measurement'):
        z, noise = z_b.copy(), noise_b.copy()
        if role == 'measurement':
            noise = noise_a.copy()
        else:
            for column in columns:
                if column['role'] == role:
                    j, k = column['source_id'], column['coordinate']
                    z[:, j, k] = z_a[:, j, k]
        values[role] = render(z, noise)
        metadata[role] = {'latents': z, 'measurement_noise': noise}
    return values, metadata


def selected_probe(train_r, train_y, val_r, val_y):
    mean, scale = train_r.mean(0), train_r.std(0)
    scale[scale < 1e-8] = 1.
    candidates = []
    for alpha in (0., .01, 1., 100., 10000.):
        fit = fit_ridge((train_r-mean)/scale, train_y, alpha)
        loss = np.mean((predict_ridge(fit, (val_r-mean)/scale)-val_y)**2)
        candidates.append((float(loss), alpha, fit))
    loss, alpha, fit = min(candidates, key=lambda row: (row[0], row[1]))
    coefficient = fit['coef']/scale[:, None]
    return {'w': coefficient.T, 'b': fit['bias']-mean @ coefficient,
            'alpha': alpha, 'validation_mse': loss}


def natural_and_frozen_events(model, x, forced=None):
    """Two-layer MLP diagnostic, all emit/reset/aux branches locked together.

    This is a declared event-path control, not a naturally perturbed SNN run.
    """
    import torch
    with torch.no_grad():
        dev = next(model.parameters()).device
        xb = torch.as_tensor(x, device=dev)
        encoded = model.encode(xb)
        model.core.reset(capture=True)
        cells = model.core.backbone.cells
        if forced is not None and len(forced) != len(cells):
            raise ValueError('One event trajectory per cell required')
        for t in range(encoded.shape[1]):
            if forced is not None:
                for j, cell in enumerate(cells):
                    event = torch.as_tensor(forced[j][:, t], device=dev)
                    cell.force_events(emit=event, reset=event, aux=event)
            model.core.backbone.step(encoded[:, t])
        r = cells[-1].last['post']
        prediction = model.core.head(r)
        events = [torch.stack(cell.trace['natural_event'], 1).cpu().numpy() for cell in cells]
        margins = [torch.stack(cell.trace['margin'], 1).cpu().numpy() for cell in cells]
        return {'r': r.cpu().numpy(), 'prediction': prediction.cpu().numpy(),
                'events': events, 'margins': margins}


def event_path_diagnostics(model, cfg, batch):
    import torch
    cells = model.core.backbone.cells
    if model.run_config.cell not in ('lif_hard', 'lif_soft', 'threshold', 'analog_hard_ref'):
        return {'status': 'not implemented for this cell; no surrogate replacement'}
    x = batch['x'][:, :cfg.deadline_steps]
    natural = natural_and_frozen_events(model, x)
    replay = natural_and_frozen_events(model, x, forced=natural['events'])
    replay_error = max(float(np.max(np.abs(natural[key]-replay[key]))) for key in ('r', 'prediction'))
    ordinary = collect(model, x)
    ordinary_error = max(float(np.max(np.abs(natural['r']-ordinary['last_post']))),
                         float(np.max(np.abs(natural['prediction']-ordinary['prediction']))))
    if max(replay_error, ordinary_error) > 1e-6:
        raise AssertionError('Manual loop / no-op event replay differs from the original forward')
    dev = next(model.parameters()).device
    xt = torch.tensor(x, device=dev, requires_grad=True)
    model.zero_grad(set_to_none=True)
    target_index = 4*4  # first C coordinate, current regression contract
    output = model(xt)
    output[:, target_index].sum().backward()
    surrogate_input_gradient = xt.grad.detach().cpu().numpy()
    model.zero_grad(set_to_none=True)
    direction_rng = np.random.default_rng(918)
    directions = direction_rng.normal(size=(len(x), cfg.visible_content_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rows = []
    for source_id, source in ((4, 'C'), (1, 'H2')):
        for epsilon in (1e-3, .03, .3):
            responses, frozen = [], []
            for sign in (-1, 1):
                z = batch['latents'].copy()
                z[:, source_id, :cfg.visible_content_dim] += sign*epsilon*directions
                changed = _render(cfg, z, batch['measurement_noise'], batch['event_times'],
                                  batch['source_present'])[:, :cfg.deadline_steps]
                responses.append(natural_and_frozen_events(model, changed))
                frozen.append(natural_and_frozen_events(model, changed, forced=natural['events']))
                if sign == -1: xminus = changed
                else: xplus = changed
            finite = (responses[1]['prediction'][:, target_index]-responses[0]['prediction'][:, target_index])/(2*epsilon)
            sg = np.sum(surrogate_input_gradient*(xplus-xminus)/(2*epsilon), axis=(1, 2))
            flips = [np.mean(left != right) for left, right in zip(responses[0]['events'], responses[1]['events'])]
            rows.append({'changed_source': source, 'latent_direction_l2': 1., 'epsilon': epsilon,
                'natural_centered_finite_response_rms_first_C_coordinate': float(np.sqrt(np.mean(finite**2))),
                'surrogate_backward_directional_rms_first_C_coordinate': float(np.sqrt(np.mean(sg**2))),
                'event_flip_fraction_by_layer': [float(value) for value in flips],
                'natural_last_post_difference_rms': float(np.sqrt(np.mean((responses[1]['r']-responses[0]['r'])**2))),
                'frozen_event_last_post_difference_rms': float(np.sqrt(np.mean((frozen[1]['r']-frozen[0]['r'])**2))),
                'warning': 'Finite perturbation response and surrogate training gradient have different meanings; frozen events are an artificial path control.'})
    return {'same_input_event_replay_max_error': replay_error,
            'manual_vs_original_forward_max_error': ordinary_error,
            'small_margin_fraction_abs_below_0_03_by_layer': [float(np.mean(np.abs(v) < .03)) for v in natural['margins']],
            'rows': rows, 'new_optimizer_updates': 0}


def zero_external_input_diagnostic(model, cfg):
    """A blank external input does not force zero core current: layers have bias."""
    import torch
    with torch.no_grad():
        dev = next(model.parameters()).device
        x = torch.zeros((1, cfg.deadline_steps, 14), device=dev)
        encoded = model.encode(x)
        model(x, capture=True)
        traces = model.core.traces()
        rows = {}
        for layer, trace in traces.items():
            row = {'base_current_rms': float(trace['base_current'].square().mean().sqrt().cpu()),
                   'post_rms': float(trace['post'].square().mean().sqrt().cpu())}
            if 'natural_event' in trace:
                row['event_count_over_one_sequence'] = int(trace['natural_event'].sum().cpu())
            rows[layer] = row
        return {'external_input_max_abs': float(x.abs().max().cpu()),
                'masked_encoder_output_max_abs': float(encoded.abs().max().cpu()),
                'steps': cfg.deadline_steps, 'layers': rows,
                'scope': 'Zero external input from zero state, with existing learned biases. Not a trained retention result or a post-write perturbation.'}


def run(root, output, smoke=False, device='cpu'):
    root, output = Path(root), Path(output)
    if output.exists(): raise FileExistsError('Choose a new derived analysis directory')
    model, run_cfg, meta = load_model(root, device)
    cfg = run_cfg.data
    if run_cfg.objective != 'regression' or cfg.extra_writes != 3:
        raise ValueError('Regression and full-presence contract required')
    nfit, nval, neval, scenes, repeats = (64, 32, 64, 16, 4) if smoke else (512, 256, 512, 256, 8)
    specs = [('encoding_fit', 'geometry_fit', 600000, nfit),
             ('probe_fit', 'probe_train', 660000, nfit),
             ('probe_selection', 'probe_val', 720000, nval),
             ('evaluation', 'development_test', 780000, neval),
             ('pickfreeze', 'development_test', 840000, neval)]
    batches = {name: fixed_schedule_batch(cfg, n, split, first_id) for name, split, first_id, n in specs}
    features = {name: collect(model, b['x'][:, :cfg.deadline_steps]) for name, b in batches.items()}
    cols = coordinate_roles(cfg, 'C', batches['evaluation']['source_present'])
    roles = [col['role'] for col in cols]
    target_columns = np.flatnonzero(np.array(roles) == 'focal')
    contents = {name: flatten_contents(b, cfg) for name, b in batches.items()}
    encoding = fit_joint(features['encoding_fit']['last_post'], contents['encoding_fit'], roles)
    probe = selected_probe(features['probe_fit']['last_post'], contents['probe_fit'][:, target_columns],
                           features['probe_selection']['last_post'], contents['probe_selection'][:, target_columns])
    head_indices = np.arange(16, 16+cfg.k)
    head = model.core.head
    readers = {'original_task_head': {'w': head.weight.detach().cpu().numpy()[head_indices].astype(float),
                                     'b': head.bias.detach().cpu().numpy()[head_indices].astype(float)},
               'natural_ridge_probe': probe}
    budgets, parent_differences = {}, {}
    for name, reader in readers.items():
        budgets[name] = role_risk_budget(features['evaluation']['last_post'], contents['evaluation'],
            encoding['a'], encoding['zmean'], encoding['rmean'], roles, reader['w'], reader['b'])
        parent = risk_accounting(features['evaluation']['last_post'], contents['evaluation'],
                                 reader['w'], reader['b'], encoding, target_columns)
        difference = budgets[name]['direct_mse']-parent['direct_mse']
        if abs(difference) > 1e-10: raise AssertionError('Parent risk identity mismatch')
        parent_differences[name] = float(difference)
    cubes, _ = pickfreeze_inputs(cfg, batches['pickfreeze'], cols)
    cube_r = {name: collect(model, values)['last_post'] for name, values in cubes.items()}
    pickfreeze = {}
    for role in ('focal', 'cotarget', 'unrequested', 'measurement'):
        pickfreeze[role] = {'representation': pickfreeze_statistics(cube_r['a'], cube_r['b'], cube_r[role])}
        for reader_name, reader in readers.items():
            pickfreeze[role][reader_name] = pickfreeze_statistics(cube_r['a'], cube_r['b'], cube_r[role], reader['w'])
    panel = repeat_panel(cfg, scenes, repeats, focal='C', mode='focal_fixed_full', first_scene_id=960000, panel_seed=7991)
    x = panel['x'][:, :, :cfg.deadline_steps]
    r = collect(model, x.reshape(scenes*repeats, *x.shape[2:]))['last_post'].reshape(scenes, repeats, -1)
    target = panel['latents'][:, 0, 4, :cfg.k]
    if not np.array_equal(panel['latents'][:, :, 4, :cfg.k], np.broadcast_to(target[:, None], (scenes, repeats, cfg.k))):
        raise AssertionError('Focal target changed inside a repeat group')
    repeated = repeated_signal_and_noise(r)
    repeated_risks = {}
    for name, reader in readers.items():
        pred = r @ reader['w'].T+reader['b']
        result = repeated_prediction_risk(pred, target)
        result['projected_conditional_covariance_mse'] = projected_variance(reader['w'], repeated['within_covariance'])
        if abs(result['projected_conditional_covariance_mse']-result['unbiased_conditional_variability_mse']) > 1e-8:
            raise AssertionError('Readout projection and repeat variability differ')
        repeated_risks[name] = result
    small = fixed_schedule_batch(cfg, min(16, neval), 'development_test', 1020000)
    event = event_path_diagnostics(model, cfg, small)
    blank = zero_external_input_diagnostic(model, cfg)
    result = {'module': 'L3.3 deepening calibration', 'version': '0.1', 'date': '2026-10-25',
        'analysis_kind': 'pipeline_smoke' if smoke else 'development_analysis',
        'source_checkpoint_id': meta['effective_id'], 'source_run_kind': meta['run_kind'],
        'checkpoint_sha256': hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
        'observation': 'terminal last_post', 'focal': 'C', 'coordinate_roles': cols,
        'data_contract': {'fixed_schedule_for_encoding_probe_and_pickfreeze': [2, 5, 8, 11, 12],
            'repeats_condition_on_focal_and_schedule_within_scene': True,
            'ids_and_splits': [{'purpose': name, 'split': split, 'first_scene_id': first_id, 'rows': n}
                               for name, split, first_id, n in specs],
            'repeat_scenes': scenes, 'repeats_per_scene': repeats, 'repeat_first_scene_id': 960000,
            'event_path_first_scene_id': 1020000, 'all_new_draws_independent_of_old_export_rows': True},
        'diagnostic_encoder_fit_rows': nfit, 'diagnostic_probe_fit_rows': nfit,
        'diagnostic_probe_selection_rows': nval, 'evaluation_rows': neval,
        'probe_alpha': probe['alpha'], 'fixed_reader_risk_budget': budgets,
        'parent_risk_direct_mse_difference': parent_differences,
        'pickfreeze_sensitivity': pickfreeze, 'conditional_readout_risk': repeated_risks,
        'repeat_covariance_trace': {name: float(np.trace(value)) for name, value in repeated.items() if isinstance(value, np.ndarray)},
        'event_path_diagnostics': event, 'zero_external_input_diagnostic': blank, 'audit_test_accessed': False,
        'new_snn_training_runs': 0, 'new_optimizer_updates': 0,
        'interpretation': 'One old tiny checkpoint, diagnostic fits and forward/backward checks. No capacity, architectural ranking, causal mediation or trained-SNN scientific conclusion.'}
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(serial(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(output/'diagnostic_arrays.npz',
        encoding_a=encoding['a'], encoding_zmean=encoding['zmean'], encoding_rmean=encoding['rmean'],
        evaluation_r=features['evaluation']['last_post'], evaluation_z=contents['evaluation'],
        **{'reader_'+name+'_'+field: reader[field] for name, reader in readers.items() for field in ('w', 'b')},
        **{'pickfreeze_'+name: value for name, value in cube_r.items()},
        repeat_responses=r, repeat_targets=target,
        **{name: value for name, value in repeated.items() if isinstance(value, np.ndarray)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    result = run(args.run_directory, args.output, args.smoke, args.device)
    print(json.dumps({key: result[key] for key in ('analysis_kind', 'source_checkpoint_id',
        'diagnostic_encoder_fit_rows', 'diagnostic_probe_fit_rows', 'diagnostic_probe_selection_rows',
        'evaluation_rows', 'audit_test_accessed', 'new_snn_training_runs', 'new_optimizer_updates')}, indent=2))
