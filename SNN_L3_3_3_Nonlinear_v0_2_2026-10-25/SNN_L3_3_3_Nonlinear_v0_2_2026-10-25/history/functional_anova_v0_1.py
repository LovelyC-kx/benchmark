"""Product-distribution, grouped functional ANOVA by complete pick-freeze.

Classical Sobol/Hoeffding identities, independently implemented. The complete
lattice resolves interactions BETWEEN declared groups, not within each group.
Finite covariance estimates may be indefinite. No clipping or deletion claims.
"""
import numpy as np


def checked_lattice(values, names, independent_factors=True):
    if independent_factors is not True:
        raise ValueError('This estimator requires a declared product input distribution')
    names = tuple(names)
    if not names or len(names) > 8 or len(set(names)) != len(names):
        raise ValueError('One to eight unique active factor groups required')
    a = np.asarray(values, dtype=float)
    if a.ndim != 3 or a.shape[0] != 2**len(names) or a.shape[1] < 2 or a.shape[2] < 1 or not np.isfinite(a).all():
        raise ValueError('Finite [all 2^groups masks, independent pairs, outputs] required')
    return a, names


def members(mask, names):
    return tuple(name for i, name in enumerate(names) if mask & (1 << i))


def mask_for(group, names):
    if not group or not set(group) <= set(names):
        raise ValueError('A nonempty subset of active factor names required')
    return sum(1 << names.index(name) for name in set(group))


def decompose(values, names, independent_factors=True, population_enumeration=False):
    """H_S=(A_S,B_-S); values[full]=h(A), values[0]=h(B).

    Closed(S)=sym Cov(h(A),h(H_S)) estimates Var(E[h|X_S]).
    Closed(empty)=0 is structural; its empirical independent covariance is
    recorded separately. Mobius inversion gives ALL nonempty group effects.
    Enumeration uses denominator n; iid Monte Carlo uses n-1. Do not bootstrap
    a deliberately exhaustive enumeration as though its rows were iid.
    """
    x, names = checked_lattice(values, names, independent_factors)
    masks, n, dim = x.shape
    centered = x-x.mean(axis=1, keepdims=True)
    ddof = 0 if population_enumeration else 1
    cross = np.einsum('ni,snj->sij', centered[-1], centered)/(n-ddof)
    closed = (cross+cross.swapaxes(-1, -2))/2
    empty_empirical = closed[0].copy()
    closed[0] = 0.
    components = closed.copy()
    for bit in range(len(names)):
        for mask in range(masks):
            if mask & (1 << bit):
                components[mask] -= components[mask ^ (1 << bit)]
    # Complement squared differences are an independently formed estimator of
    # each factor's total effect, not a second exact finite-sample identity.
    jansen = {}
    for bit, name in enumerate(names):
        delta = x[-1]-x[(masks-1) ^ (1 << bit)]
        jansen[name] = delta.T @ delta/(2*n)
    return {'names': names, 'n_pairs': n, 'output_dim': dim,
            'closed': closed, 'components': components,
            'total_covariance': closed[-1],
            'empty_empirical_cross_covariance': empty_empirical,
            'total_effect_difference_covariances': jansen,
            'covariance_closure_max_error': float(np.max(np.abs(components[1:].sum(0)-closed[-1]))),
            'population_enumeration': bool(population_enumeration),
            'estimator': 'complete grouped product-distribution pick-freeze with Mobius inversion',
            'negative_estimates_retained': True}


def power_summary(fit, target_groups=None):
    c = fit['components']; names = fit['names']; dim = fit['output_dim']
    power = np.trace(c, axis1=-2, axis2=-1)/dim
    total = float(np.trace(fit['total_covariance'])/dim)
    rows = [{'mask': mask, 'groups': members(mask, names),
             'group_order': mask.bit_count(), 'power_per_output': float(power[mask]),
             'fraction_of_total_variance': float(power[mask]/total) if total > 0 else None}
            for mask in range(1, len(c))]
    per_factor = {}
    for i, name in enumerate(names):
        involved = [mask for mask in range(1, len(c)) if mask & (1 << i)]
        per_factor[name] = {'main': float(power[1 << i]),
            'total_including_interactions': float(power[involved].sum()),
            'total_by_complement_difference': float(np.trace(fit['total_effect_difference_covariances'][name])/dim)}
    result = {'component_rows': rows, 'total_variance_per_output': total,
              'by_group_order': {str(order): float(sum(row['power_per_output'] for row in rows if row['group_order'] == order)) for order in range(1, len(names)+1)},
              'factor_main_and_total': per_factor,
              'empty_independence_check_cross_trace': float(np.trace(fit['empty_empirical_cross_covariance'])/dim),
              'covariance_closure_max_error': fit['covariance_closure_max_error'],
              'note': 'Total effects overlap; only exclusive component rows or disjoint aggregates may be summed.'}
    if target_groups is not None:
        allowed = mask_for(target_groups, names)
        outside = [mask for mask in range(1, len(c)) if mask & ~allowed]
        pure = [mask for mask in range(1, len(c)) if not mask & ~allowed]
        result['target_contract'] = {'groups': tuple(target_groups), 'target_only_power': float(power[pure].sum()),
            'non_target_conditional_variability_power': float(power[outside].sum()),
            'scope': 'For prediction outputs and a target depending only on these groups, the second aggregate estimates E Var(prediction|target factors).'}
    return result


def error_risk_budget(error_lattice, names, target_groups, population_enumeration=False):
    """ANOVA of ERROR, with a finite-sample bias-power correction.

    iid: unbiased squared mean error = ||mean(e)||²/q - tr Cov(e)/(n*q).
    This plus all covariance components equals direct sample MSE exactly.
    Closing the algebra is not evidence that individual component estimates
    are accurate. The sum is unbiased; components need Monte Carlo intervals.
    """
    fit = decompose(error_lattice, names, population_enumeration=population_enumeration)
    summary = power_summary(fit, target_groups)
    e = np.asarray(error_lattice)[-1]
    mean_square = float(np.mean(e.mean(0)**2))
    correction = 0. if population_enumeration else summary['total_variance_per_output']/len(e)
    bias_power = mean_square-correction
    direct = float(np.mean(e**2))
    accounted = bias_power+sum(row['power_per_output'] for row in summary['component_rows'])
    return fit, dict(summary, direct_mse=direct,
        raw_squared_mean_error=mean_square, finite_mean_bias_correction=correction,
        unbiased_squared_population_bias=bias_power,
        accounted_mse=accounted, mse_identity_error=accounted-direct,
        target_only_error_plus_bias=bias_power+summary['target_contract']['target_only_power'],
        scope='ANOVA of fixed prediction minus correctly recomputed hybrid target; exclusive error effects, not benefits of deleting inputs.')


def paired_bootstrap(values, names, target_groups, draws=200, seed=1137):
    """Resample an ENTIRE A/B/hybrid row together; fixed predictor only.

    Reports pointwise percentile and a same-unit max-deviation simultaneous
    bootstrap band. This is approximate Monte Carlo uncertainty, not training,
    selection or architecture uncertainty. No p-values or clipping are used.
    """
    x, names = checked_lattice(values, names)
    if draws < 20:
        raise ValueError('At least 20 bootstrap draws required')
    allowed = mask_for(target_groups, names)
    _, base = error_risk_budget(x, names, target_groups)
    keys = [f'component_{i}' for i in range(1, len(x))]+['non_target', 'target_only_plus_bias', 'direct_mse']
    def vector(s):
        return np.array([row['power_per_output'] for row in s['component_rows']]+[
            s['target_contract']['non_target_conditional_variability_power'], s['target_only_error_plus_bias'], s['direct_mse']])
    point = vector(base); samples = []
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        ids = rng.integers(x.shape[1], size=x.shape[1])
        _, summary = error_risk_budget(x[:, ids], names, target_groups)
        samples.append(vector(summary))
    samples = np.stack(samples)
    band = float(np.quantile(np.max(np.abs(samples-point), axis=1), .95))
    return {'draws': draws, 'independent_pair_rows': x.shape[1], 'target_mask': allowed,
        'rows': [{'quantity': key, 'estimate': float(point[i]),
                  'pointwise_percentile_95': np.quantile(samples[:, i], [.025, .975]).tolist(),
                  'simultaneous_max_deviation_95': [float(point[i]-band), float(point[i]+band)]}
                 for i, key in enumerate(keys)],
        'max_deviation_band_radius': band,
        'scope': 'Approximate uncertainty for iid pair sampling, conditional on fixed predictor, chosen factors and schedule. Full hybrid rows resampled together.'}


def conditional_risk(predictions, targets):
    p, y = np.asarray(predictions, float), np.asarray(targets, float)
    if p.ndim != 3 or min(p.shape[:2]) < 2 or y.shape != (p.shape[0], p.shape[2]) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Predictions [independent scenes, iid repeats, outputs], fixed targets [scenes, outputs] required')
    scenes, repeats, q = p.shape
    per_scene_within = np.sum((p-p.mean(1, keepdims=True))**2, axis=(1, 2))/((repeats-1)*q)
    per_scene_mean_error = np.mean((p.mean(1)-y)**2, axis=1)-per_scene_within/repeats
    direct = float(np.mean((p-y[:, None])**2))
    return {'direct_mse': direct, 'conditional_mean_error': float(per_scene_mean_error.mean()),
            'conditional_variability': float(per_scene_within.mean()),
            'identity_error': float(per_scene_mean_error.mean()+per_scene_within.mean()-direct),
            'per_scene_conditional_variability': per_scene_within,
            'per_scene_conditional_mean_error': per_scene_mean_error,
            'repeats_not_independent_scenes': True, 'negative_corrected_estimates_retained': True}


def conditional_mean_lattice(panels, names):
    """Estimate component FUNCTIONS using nested conditional Monte Carlo.

    Each mask panel: [scenes, completions, outputs]. Each scene shares the same
    base factors across masks, and holds precisely that mask. Full mask must
    be an identical replay. The result reconstructs h(base) algebraically;
    estimated components are not exactly orthogonal in finite completions.
    """
    if set(panels) != set(range(2**len(names))):
        raise ValueError('One conditional completion panel for every mask required')
    means, within = [], []
    shape = None
    for mask in range(2**len(names)):
        p = np.asarray(panels[mask], float)
        if p.ndim != 3 or min(p.shape[:2]) < 2 or not np.isfinite(p).all():
            raise ValueError('Finite conditional panels with >=2 scenes/completions required')
        if shape is not None and p.shape != shape:
            raise ValueError('Identical panel shapes and common base scenes required')
        shape = p.shape
        means.append(p.mean(1))
        within.append(np.var(p, axis=1, ddof=1)/p.shape[1])
    if not np.allclose(panels[len(panels)-1], np.asarray(panels[len(panels)-1])[:, :1], atol=1e-12, rtol=1e-12):
        raise ValueError('The full-mask panel must replay identical complete factors')
    means = np.stack(means); components = means.copy()
    for bit in range(len(names)):
        for mask in range(len(means)):
            if mask & (1 << bit):
                components[mask] -= components[mask ^ (1 << bit)]
    return {'conditional_means': means, 'component_functions_estimated': components,
            'conditional_mean_mc_variance': np.stack(within),
            'reconstruction_max_error': float(np.max(np.abs(components.sum(0)-means[-1]))),
            'scope': 'Nested Monte Carlo functions at common base scenes, not exact population component values. Variances need completion-noise correction; no component neuron labeling.'}
