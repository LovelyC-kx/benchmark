"""Content-resolved operations with probes fitted on natural states only.

The existing v0.2 exporter only saved operation predictions on development
validation inputs. This script closes the feature/label/locked-probe gap using
a separate development evaluation split. It never accesses the final audit.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_features import serial
from experiment_plan import config_from_dict
from load_data import SOURCES, generate, make_source_pair, model_payload, classification_labels
from linear_reference import fit_ridge, predict_ridge, regression_metrics


def fit_natural_probe(train_x, train_y, val_x, val_y):
    """Use natural probe_train/probe_val only; never choose using an operation."""
    raw = np.asarray(train_x, dtype=float)
    mean = raw.mean(0)
    scale = raw.std(0)
    scale[scale < 1e-8] = 1.
    x = (raw-mean)/scale
    vx = (np.asarray(val_x)-mean)/scale
    candidates = []
    for alpha in (1e-4, .01, 1., 100., 10000.):
        fit = fit_ridge(x, train_y, alpha)
        loss = float(np.mean((predict_ridge(fit, vx)-val_y)**2))
        candidates.append((loss, alpha, fit))
    _, _, fit = min(candidates, key=lambda item: (item[0], item[1]))
    return {'mean': mean, 'scale': scale, 'fit': fit}


def apply_locked_probe(probe, features):
    return predict_ridge(probe['fit'], (np.asarray(features)-probe['mean'])/probe['scale'])


def paired_mse_change(natural, operated, target, seed=1203):
    """Within-checkpoint sequence uncertainty; not a training-seed interval."""
    base = np.mean((natural-target)**2, axis=1)
    changed = np.mean((operated-target)**2, axis=1)
    delta = changed-base
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(1000, len(delta)))].mean(1)
    return {'natural_mean_mse': float(base.mean()), 'operated_mean_mse': float(changed.mean()),
            'paired_delta_mse': float(delta.mean()),
            'sequence_bootstrap_95': np.quantile(draws, [.025, .975]).tolist(),
            'interval_scope': 'paired sequences within one checkpoint; not across model training seeds'}


def encoder_availability(model, cfg, counts):
    """Per-source instantaneous encoding; no prefix aggregation or IPC subtraction."""
    import torch
    device = next(model.parameters()).device
    values = {}
    for split, count in counts.items():
        b = generate(cfg, count, split)
        at_events = b['x'][np.arange(count)[:, None], b['event_times']]
        with torch.no_grad():
            encoded = model.encode(torch.as_tensor(at_events, device=device)).cpu().numpy()
        values[split] = (encoded, b['latents'][:, :, :cfg.visible_content_dim], b['source_present'])
    result = []
    for j, source in enumerate(SOURCES):
        parts = {split: (x[present[:, j], j], y[present[:, j], j])
                 for split, (x, y, present) in values.items()}
        available = {split: len(x) for split, (x, _) in parts.items()}
        if min(available.values()) < 2:
            result.append({'source': source, 'status': 'insufficient source-present samples', 'counts': available})
            continue
        x, y = parts['probe_train']; vx, vy = parts['probe_val']; tx, ty = parts['development_test']
        probe = fit_natural_probe(x, y, vx, vy)
        result.append({'source': source, 'counts': available, 'encoder_observation_dim': x.shape[1],
                       'chosen_alpha': probe['fit']['alpha'],
                       'metrics': regression_metrics(apply_locked_probe(probe, tx), ty),
                       'scope': 'instantaneous encoder-output availability at its own source event; not temporal retention or a causal capacity subtraction'})
    return result


def analyze(root, probe_n=None, sample_n=None, device='cpu', write_arrays=True):
    import torch
    from torch_adapter import LoadModel
    from intervention_features import OBSERVATIONS, collect_operation_features
    root = Path(root)
    metadata = json.loads((root/'run.json').read_text())
    run = config_from_dict(metadata['config'])
    sizes = metadata['split_sizes']
    n = sizes['development_test'] if sample_n is None else sample_n
    pn = sizes['probe_train'] if probe_n is None else probe_n
    if not 2 <= n <= sizes['development_test'] or not 2 <= pn <= sizes['probe_train']:
        raise ValueError('Requested counts must fit the recorded independent splits')
    checkpoint = root/'best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['config'] != asdict(run):
        raise ValueError('Checkpoint and run metadata configurations differ')
    torch.manual_seed(run.seed)
    model = LoadModel(run).to(device)
    model.load_state_dict(saved['model_state'])
    batch = generate(run.data, n, 'development_test')
    target_k = run.data.visible_content_dim
    targets = batch['latents'][:, :, :target_k]
    natural = collect_operation_features(model, model_payload(batch)['x'])
    max_noop = max(float(np.max(np.abs(natural['baseline'][key]-natural['noop'][key])))
                   for key in natural['baseline'])
    if max_noop > 1e-6:
        raise AssertionError('No-op feature/head replay failed')
    train = np.load(root/'probe_train.npz')
    val = np.load(root/'probe_val.npz')
    probes = {}
    recoveries = []
    fingerprints = {}
    for obs in OBSERVATIONS:
        for j, source in enumerate(SOURCES):
            sl = slice(j*target_k, (j+1)*target_k)
            probe = fit_natural_probe(train[obs][:pn], train['targets'][:pn, sl],
                                      val[obs], val['targets'][:, sl])
            probes[obs, j] = probe
            fingerprint = hashlib.sha256()
            for arr in (probe['mean'], probe['scale'], probe['fit']['coef'], probe['fit']['bias']):
                fingerprint.update(np.ascontiguousarray(arr).tobytes())
            fingerprints[f'{obs}/{source}'] = fingerprint.hexdigest()
            baseline = apply_locked_probe(probe, natural['baseline'][obs])
            for op, features in natural.items():
                pred = apply_locked_probe(probe, features[obs])
                row = {'observation': obs, 'source': source, 'operation': op,
                       'probe_alpha': probe['fit']['alpha'], 'probe_refitted_after_operation': False,
                       'source_present_fraction': float(batch['source_present'][:, j].mean()),
                       'all_visible_coordinates': regression_metrics(pred, targets[:, j]),
                       'requested_coordinates': regression_metrics(pred[:, :run.data.k], targets[:, j, :run.data.k]),
                       'paired_error_change': paired_mse_change(baseline, pred, targets[:, j])}
                recoveries.append(row)
    head_rows = []
    for op, features in natural.items():
        for j in list(range(run.data.required_history))+[4]:
            if run.objective == 'regression':
                p = features['prediction'].reshape(n, 5, 4)[:, j, :run.data.k]
                base = natural['baseline']['prediction'].reshape(n, 5, 4)[:, j, :run.data.k]
                row = paired_mse_change(base, p, targets[:, j, :run.data.k])
            else:
                truth = classification_labels(run.data, batch)[:, j]
                logits = features['prediction'][:, j]
                row = {'accuracy': float(np.mean((logits >= 0) == truth)),
                       'mean_bce': float(np.mean(np.logaddexp(0, logits)-truth*logits))}
            head_rows.append({'operation': op, 'source': SOURCES[j], 'metrics': row})
    responses = []
    clear_history_error = 0.
    for changed_j, changed_source in enumerate(SOURCES):
        if not batch['source_present'][:, changed_j].any():
            continue
        paired = make_source_pair(run.data, batch, changed_source, pair_seed=17)
        pair_features = collect_operation_features(model, model_payload(paired)['x'])
        if changed_source != 'C':
            for key in natural['clear_core']:
                clear_history_error = max(clear_history_error, float(np.max(np.abs(
                    natural['clear_core'][key]-pair_features['clear_core'][key]))))
        for op in natural:
            for obs in OBSERVATIONS:
                for j, decoded_source in enumerate(SOURCES):
                    probe = probes[obs, j]
                    before = apply_locked_probe(probe, natural[op][obs])
                    after = apply_locked_probe(probe, pair_features[op][obs])
                    responses.append({'changed_source': changed_source, 'decoded_source': decoded_source,
                        'observation': obs, 'operation': op,
                        'changed_source_present_fraction': float(batch['source_present'][:, changed_j].mean()),
                        'decoded_response_mean_square': float(np.mean((after-before)**2)),
                        'paired_target_mean_square_change': float(np.mean((
                            paired['latents'][:, j, :target_k]-targets[:, j])**2)),
                        'scope': 'natural-input sensitivity through one locked diagnostic probe; not semantic interference proof'})
    if clear_history_error > 1e-6:
        raise AssertionError('Full core clear retained a dependence on pre-cut content')
    if write_arrays:
        np.savez_compressed(root/'development_intervention_features_v03.npz',
            **{f'{op}__{key}': value for op, values in natural.items() for key, value in values.items()},
            targets=targets, sample_ids=batch['sample_ids'], source_present=batch['source_present'])
    result = serial({'schema': 'l32_interventions_v03', 'scope': 'development_test',
        'run_kind': metadata['run_kind'], 'run_id': metadata['effective_id'],
        'sample_n': n, 'probe_n': pn, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'probe_fit_split': 'probe_train natural states', 'probe_selection_split': 'probe_val natural states',
        'probe_fingerprints': fingerprints, 'max_noop_feature_or_head_error': max_noop,
        'max_pre_cut_source_effect_after_full_clear': clear_history_error,
        'content_recoveries': recoveries, 'original_head': head_rows, 'source_pair_responses': responses,
        'instantaneous_encoder_availability': encoder_availability(model, run.data,
            {'probe_train': pn, 'probe_val': sizes['probe_val'], 'development_test': n}),
        'interpretation': 'Fixed-checkpoint forward dependence and fixed-probe recoverability. Operation-induced distribution/scale changes remain possible; no formal benchmark claim.',
        'audit_test_accessed': False})
    (root/'content_interventions_development_v03.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_directory', type=Path)
    ap.add_argument('--probe-n', type=int)
    ap.add_argument('--sample-n', type=int)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    r = analyze(args.run_directory, args.probe_n, args.sample_n, args.device)
    print(json.dumps({k: r[k] for k in ('run_kind', 'sample_n', 'probe_n',
        'max_noop_feature_or_head_error', 'max_pre_cut_source_effect_after_full_clear', 'audit_test_accessed')}))
