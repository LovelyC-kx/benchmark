"""Nontrivial integration of the geometry pipeline on a known linear system.

This fits diagnostic linear models, never an SNN. It deliberately includes
correlated nuisance references so encoding-only projection loses useful input.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from bridge import gaussian_readout_reference, repeated_signal_and_noise, repeated_prediction_risk
from subspace_geometry import (fit_axes, independent_replica, overlap, projector,
    fixed_rank_bootstrap_overlap, null_overlaps)
from readout_geometry import (covariance, choose_covariance_metric, noise_metric_geometry,
    fit_projected_probe, predict_raw, metrics)


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [serial(v) for v in x]
    return x


def reference_run():
    rng = np.random.default_rng(831)
    dim, factors = 12, 20
    a = np.zeros((dim, factors))
    a[:4, 16:] = np.eye(4)            # focal contents
    a[:4, :4] = np.eye(4)             # cotarget nuisance at focal readout
    a[4:8, :4] = np.eye(4)            # independent spatial reference for it
    a[8:, 4:16] = rng.normal(size=(4, 12))/np.sqrt(12)
    roles = np.array(['cotarget']*4+['unrequested']*12+['focal']*4)
    def draw(n):
        z = rng.normal(size=(n, factors)); r = z @ a.T+.1*rng.normal(size=(n, dim))
        return r, z
    ra, za = draw(1024); rv, zv = draw(512); rb, zb = draw(1024)
    rp, zp = draw(1024); rpv, zpv = draw(512); re, ze = draw(4096)
    fit, axes = fit_axes(ra, za, roles, rv, zv)
    replica, qb, stability = independent_replica(fit, axes, rb, zb, roles)
    mean = fit['rmean']; scale = np.sqrt(np.trace(covariance(ra))/dim)
    full = fit_projected_probe(rp, zp[:, 16:], rpv, zpv[:, 16:], np.eye(dim), mean, scale)
    reduced = fit_projected_probe(rp, zp[:, 16:], rpv, zpv[:, 16:], axes['focal']['q'], mean, scale)
    oracle = gaussian_readout_reference(a[:, 16:], a[:, :16], .01*np.eye(dim))
    # Fresh conditionally iid repeats; all focal coordinates are fixed in each scene.
    z0 = rng.normal(size=(256, 1, factors))
    zr = rng.normal(size=(256, 8, factors)); zr[:, :, 16:] = z0[:, :, 16:]
    rr = zr @ a.T+.1*rng.normal(size=(256, 8, dim))
    repeat = repeated_signal_and_noise(rr)
    z1 = rng.normal(size=(128, 1, factors))
    zs = rng.normal(size=(128, 8, factors)); zs[:, :, 16:] = z1[:, :, 16:]
    sr = zs @ a.T+.1*rng.normal(size=(128, 8, dim))
    noise = choose_covariance_metric(repeat['within_covariance'], repeated_signal_and_noise(sr)['within_covariance'])
    metric = noise_metric_geometry(a[:, 16:], a[:, :4], noise['regularized_covariance'])
    boot = fixed_rank_bootstrap_overlap(ra, za, rb, zb, roles,
        axes['focal']['q'].shape[1], qb['cotarget'].shape[1], draws=48, seed=832)
    result = {'kind': 'known_linear_system_integration', 'new_snn_training_runs': 0,
        'observation_dim': dim, 'factor_dim': factors, 'fit_rows': 1024, 'selection_rows': 512,
        'evaluation_rows': 4096,
        'selected_ranks': {role: axes[role]['selection']['selected_rank'] for role in axes},
        'replica_stability': stability,
        'focal_a_other_b_overlap': overlap(axes['focal']['q'], qb['cotarget']),
        'focal_cotarget_bootstrap': boot,
        'nulls': null_overlaps(axes['focal']['q'], qb['cotarget'].shape[1], covariance(rv), draws=256),
        'conditional_noise_gamma': noise['gamma'],
        'conditional_noise_metric_overlap': metric['metric_overlap'],
        'population_full_gaussian_mmse': oracle['mse_per_focal_coordinate'],
        'empirical_oracle_mse': metrics(re @ oracle['w'].T, ze[:, 16:])['mean_mse'],
        'full_ridge': metrics(predict_raw(full, re), ze[:, 16:]),
        'encoding_only_ridge': metrics(predict_raw(reduced, re), ze[:, 16:]),
        'population_ideal_encoding_only_mmse': 1-1/2.01,
        'conditional_readout_risk': repeated_prediction_risk(rr @ full['w'].T+full['b'], zr[:, 0, 16:]),
        'scope': 'Synthetic independent-Gaussian linear generator with known factors. These are diagnostic regression fits, not trained-SNN scientific results.'}
    if result['selected_ranks']['focal'] == 0 or result['selected_ranks']['cotarget'] == 0:
        raise AssertionError('Strong planted linear structure was not selected')
    if abs(result['empirical_oracle_mse']-result['population_full_gaussian_mmse']) > .003:
        raise AssertionError('Known population oracle calibration failed')
    if result['encoding_only_ridge']['mean_mse'] < result['full_ridge']['mean_mse']+.3:
        raise AssertionError('Noise-reference projection control failed')
    if boot['valid_draws'] != 48: raise AssertionError('Bootstrap integration did not use every requested draw')
    if abs(result['conditional_readout_risk']['identity_error']) > 1e-12: raise AssertionError('Conditional risk identity')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new reference output')
    result = reference_run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serial(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('kind', 'selected_ranks', 'population_full_gaussian_mmse',
        'empirical_oracle_mse', 'population_ideal_encoding_only_mmse', 'new_snn_training_runs')}, indent=2))
