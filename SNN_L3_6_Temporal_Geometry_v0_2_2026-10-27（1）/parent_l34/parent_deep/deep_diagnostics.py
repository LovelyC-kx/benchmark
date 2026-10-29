"""Declared-readout risk and known-answer controls for L3.3.

Independent implementation. These identities are classical probability/linear
algebra, not new SNN theorems. No model fitting, clipping or causal attribution.
"""
from itertools import combinations
import numpy as np


def matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite rows required')
    return x


def role_risk_budget(r, z, a, zmean, rmean, roles, w, b):
    """Exact held-out MSE: focal mismatch, cotarget, unused, residual, offset.

    w is Q x D. Its output must predict focal coordinates, in their z-column
    order. All cross second moments are retained, including offset cross terms.
    """
    r, z = matrix(r), matrix(z)
    a, w = np.asarray(a, float), np.asarray(w, float)
    zmean, rmean, b = map(lambda x: np.asarray(x, float), (zmean, rmean, b))
    roles = np.asarray(roles)
    if (len(z) != len(r) or a.shape != (r.shape[1], z.shape[1]) or
            roles.shape != (z.shape[1],) or
            not set(roles) <= {'focal', 'cotarget', 'unrequested'}):
        raise ValueError('Matched representation, factor and role schema required')
    focal = np.flatnonzero(roles == 'focal')
    if (not len(focal) or w.shape != (len(focal), r.shape[1]) or
            zmean.shape != (z.shape[1],) or rmean.shape != (r.shape[1],) or
            b.shape != (len(focal),)):
        raise ValueError('Focal linear-head shape mismatch')
    if not all(np.isfinite(x).all() for x in (a, w, zmean, rmean, b)):
        raise ValueError('Finite parameters required')
    centered = z-zmean
    residual = r-rmean-centered @ a.T
    parts = {'focal_gain_or_mixing_error': centered[:, focal] @
             (w @ a[:, focal]-np.eye(len(focal))).T}
    for role in ('cotarget', 'unrequested'):
        columns = np.flatnonzero(roles == role)
        parts[role+'_injection'] = centered[:, columns] @ (w @ a[:, columns]).T
    parts['linear_unexplained_through_head'] = residual @ w.T
    parts['constant_offset'] = np.broadcast_to(rmean @ w.T+b-zmean[focal], (len(r), len(focal)))
    direct = r @ w.T+b-z[:, focal]
    diagonal = {name: float(np.mean(value**2)) for name, value in parts.items()}
    cross = {left+'__'+right: float(2*np.mean(parts[left]*parts[right]))
             for left, right in combinations(parts, 2)}
    accounted = sum(diagonal.values())+sum(cross.values())
    return {'direct_mse': float(np.mean(direct**2)), 'accounted_mse': accounted,
            'diagonal_second_moments': diagonal, 'cross_second_moments': cross,
            'max_error_reconstruction_difference': float(np.max(np.abs(direct-sum(parts.values())))),
            'mse_identity_difference': float(accounted-np.mean(direct**2)),
            'interpretation': 'Exact algebra for one locked linear readout; components are not causal deletion benefits.'}


def gaussian_readout_reference(a_focal, a_other, noise_covariance):
    """z_F,z_other independent N(0,I); additive independent Gaussian noise.

    Generalizes the parent isotropic-noise control to arbitrary PSD covariance.
    This is also the best linear estimator under the corresponding moments.
    """
    af, ao, noise = map(lambda x: np.asarray(x, float), (a_focal, a_other, noise_covariance))
    if (af.ndim != 2 or ao.ndim != 2 or ao.shape[0] != af.shape[0] or
            not af.shape[1] or noise.shape != (af.shape[0], af.shape[0]) or
            not all(np.isfinite(x).all() for x in (af, ao, noise)) or
            not np.allclose(noise, noise.T, atol=1e-10)):
        raise ValueError('Finite matched linear maps and symmetric noise covariance required')
    if np.linalg.eigvalsh(noise).min() < -1e-10:
        raise ValueError('Population noise covariance must be PSD')
    covariance = af @ af.T+ao @ ao.T+noise
    inverse = np.linalg.pinv(covariance, rcond=1e-12, hermitian=True)
    w = af.T @ inverse
    error_covariance = np.eye(af.shape[1])-af.T @ inverse @ af
    error_covariance = (error_covariance+error_covariance.T)/2
    return {'w': w, 'representation_covariance': covariance,
            'error_covariance': error_covariance,
            'mse_per_focal_coordinate': float(np.trace(error_covariance)/af.shape[1]),
            'rank': int(np.linalg.matrix_rank(covariance))}


def projected_variance(w, covariance):
    w, covariance = np.asarray(w, float), np.asarray(covariance, float)
    if (w.ndim != 2 or not w.shape[0] or covariance.shape != (w.shape[1], w.shape[1])
            or not np.isfinite(w).all() or not np.isfinite(covariance).all()):
        raise ValueError('Matched finite linear readout and covariance required')
    return float(np.trace(w @ covariance @ w.T)/w.shape[0])


def pickfreeze_statistics(a, b, hybrid, w=None):
    """Independent A/B, hybrid=(A_i,B_-i), deterministic function of all factors.

    E cross covariance = main-effect covariance for factor i. Half the squared
    B/hybrid difference estimates its total effect, including interactions.
    Independent stochastic inputs must be explicit factors; the schedule must
    be fixed or estimates separately conditioned. Main/total are NOT harm.
    """
    a, b, h = matrix(a), matrix(b), matrix(hybrid)
    if a.shape != b.shape or a.shape != h.shape:
        raise ValueError('Matched pick-freeze rows required')
    ac, hc = a-a.mean(0), h-h.mean(0)
    main = (ac.T @ hc+hc.T @ ac)/(2*(len(a)-1))
    difference = b-h
    total = difference.T @ difference/(2*len(a))
    total_covariance = np.cov(a, rowvar=False).reshape(a.shape[1], a.shape[1])
    if w is None:
        w = np.eye(a.shape[1])
    return {'main_effect_weighted_variance': projected_variance(w, main),
            'total_effect_weighted_variance': projected_variance(w, total),
            'total_minus_main': projected_variance(w, total-main),
            'reference_weighted_variance': projected_variance(w, total_covariance),
            'finite_negative_main_or_difference_retained': True,
            'scope': 'Functional sensitivity under the independent-factor sampling contract, not deletion benefit or mediation.'}


def repeated_signal_and_noise(responses):
    """Noise-corrected between-condition covariance from independent repeats.

    Both estimators can be indefinite in finite samples and are not inverted.
    This splits repeats within a scene, NOT neurons in the 2026 split-trial paper.
    """
    r = np.asarray(responses, float)
    if r.ndim != 3 or min(r.shape[:2]) < 2 or not np.isfinite(r).all():
        raise ValueError('Finite [scenes,repeats,state] with >=2 scenes/repeats required')
    g, repeats, dim = r.shape
    centered = r-r.mean(1, keepdims=True)
    within = np.einsum('gbi,gbj->ij', centered, centered)/(g*(repeats-1))
    means = r.mean(1)
    raw_between = np.cov(means, rowvar=False).reshape(dim, dim)
    corrected = raw_between-within/repeats
    left = r[:, :repeats//2].mean(1)
    right = r[:, repeats//2:].mean(1)
    left, right = left-left.mean(0), right-right.mean(0)
    cross_replica = (left.T @ right+right.T @ left)/(2*(g-1))
    return {'within_covariance': within, 'raw_between_covariance': raw_between,
            'corrected_between_covariance': corrected, 'cross_replica_covariance': cross_replica,
            'scene_count': g, 'repeat_count': repeats,
            'warning': 'Requires conditionally iid repeats. Negative corrected eigenvalues are kept; no Fisher inverse.'}


def repeated_prediction_risk(predictions, scene_targets):
    """Exact finite-B accounting with unbiased conditional mean-error estimate.

    Target is fixed inside each repeat group. The unbiased squared-mean-error
    estimator can be negative. We never label it a causal noise-removal gain.
    """
    p = np.asarray(predictions, float)
    y = np.asarray(scene_targets, float)
    if (p.ndim != 3 or min(p.shape[:2]) < 2 or y.shape != (p.shape[0], p.shape[2]) or
            not np.isfinite(p).all() or not np.isfinite(y).all()):
        raise ValueError('Matched predictions [G,B,Q] and fixed targets [G,Q] required')
    repeats = p.shape[1]
    mean_risk = float(np.mean((p.mean(1)-y)**2))
    within = float(np.sum((p-p.mean(1, keepdims=True))**2)/(p.shape[0]*(repeats-1)*p.shape[2]))
    corrected_mean_risk = mean_risk-within/repeats
    direct = float(np.mean((p-y[:, None])**2))
    return {'direct_mse': direct, 'raw_scene_mean_mse': mean_risk,
            'unbiased_conditional_mean_error_mse': corrected_mean_risk,
            'unbiased_conditional_variability_mse': within,
            'accounted_mse': corrected_mean_risk+within,
            'identity_error': corrected_mean_risk+within-direct,
            'raw_mean_bias_correction': within/repeats,
            'negative_corrected_estimates_retained': True}


def scalar_lif(currents, beta=.8, threshold=1., kind='hard', forced_events=None):
    """Exact discrete scalar forward path; frozen events are a diagnostic."""
    x = np.asarray(currents, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 <= beta <= 1 or threshold <= 0:
        raise ValueError('Finite current sequence and legal dynamics required')
    if kind not in ('hard', 'soft', 'none'):
        raise ValueError('Reset kind')
    if forced_events is not None:
        forced_events = np.asarray(forced_events)
        if forced_events.shape != x.shape or not np.isin(forced_events, [0, 1]).all():
            raise ValueError('One fixed binary event per time required')
    state, pre, events, post = 0., [], [], []
    for t, current in enumerate(x):
        value = beta*state+current
        event = int(value >= threshold) if forced_events is None else int(forced_events[t])
        state = value*(1-event) if kind == 'hard' else value-threshold*event if kind == 'soft' else value
        pre.append(value); events.append(event); post.append(state)
    return {'pre': np.asarray(pre), 'events': np.asarray(events), 'post': np.asarray(post)}
