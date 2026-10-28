"""Joint encoding, finite second-order controls and explicit residual accounting.

Encoding predicts representation FROM known factors. This differs from a probe
predicting factors FROM the representation. No residual is named intrinsic noise.
"""
from itertools import combinations
import numpy as np
from inherited import fit_linear_encoding, fit_ridge, predict_ridge

ROLE_NAMES = ('focal', 'cotarget', 'unrequested')


def matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite rows in a matrix required')
    return x


def fit_joint(r, z, roles):
    r, z = matrix(r), matrix(z)
    if len(r) != len(z) or z.shape[1] != len(roles) or not set(roles) <= set(ROLE_NAMES):
        raise ValueError('Matched representation/factor rows and valid per-coordinate roles required')
    fit = fit_linear_encoding(r, z)
    rank = int(np.linalg.matrix_rank(z-z.mean(0), tol=1e-10))
    fit.update(roles=list(roles), design_rank=rank,
               coefficients_identifiable=rank == z.shape[1],
               interpretation='joint linear statistical encoding; not a causal or exhaustive nonlinear factorization')
    return fit


def apply_joint(fit, r, z):
    r, z = matrix(r), matrix(z)
    if len(r) != len(z) or z.shape[1] != fit['a'].shape[1] or r.shape[1] != fit['a'].shape[0]:
        raise ValueError('Evaluation shapes differ from fitted schema')
    zc = z-fit['zmean']
    roles = np.asarray(fit['roles'])
    parts = {name: zc[:, roles == name] @ fit['a'][:, roles == name].T for name in ROLE_NAMES}
    prediction = fit['rmean']+sum(parts.values())
    parts['unexplained_by_joint_linear_model'] = r-prediction
    return parts


def covariance_accounting(r, parts):
    """Held-out covariance identity including all finite-sample cross terms."""
    r = matrix(r)
    if any(np.shape(x) != r.shape for x in parts.values()):
        raise ValueError('Every component must have the representation shape')
    centered = {name: np.asarray(x)-np.mean(x, axis=0) for name, x in parts.items()}
    denominator = len(r)-1
    matrices = {name: x.T @ x/denominator for name, x in centered.items()}
    cross = {}
    for a, b in combinations(centered, 2):
        cross[a+'__'+b] = (centered[a].T @ centered[b]+centered[b].T @ centered[a])/denominator
    actual = (r-r.mean(0)).T @ (r-r.mean(0))/denominator
    accounted = sum(matrices.values())+sum(cross.values())
    return {'component_covariances': matrices, 'cross_covariances': cross,
            'total_covariance': actual,
            'component_trace': {key: float(np.trace(value)) for key, value in matrices.items()},
            'cross_trace': {key: float(np.trace(value)) for key, value in cross.items()},
            'max_covariance_identity_error': float(np.max(np.abs(actual-accounted))),
            'note': 'Uncorrelated factors need not occupy orthogonal neural directions. Finite cross terms are retained.'}


def representation_score(r, prediction, fit_mean):
    r, prediction = matrix(r), matrix(prediction)
    if r.shape != prediction.shape:
        raise ValueError('Prediction shapes')
    sse = float(np.square(r-prediction).sum())
    sst = float(np.square(r-r.mean(0)).sum())
    mean_sse = float(np.square(r-fit_mean).sum())
    return {'heldout_representation_r2': 1-sse/sst if sst > 1e-20 else None,
            'mse_per_state_coordinate': sse/r.size,
            'mse_relative_to_fit_mean': sse/mean_sse if mean_sse > 1e-20 else None,
            'scope': 'predicting the representation from contents; not downstream target recovery'}


def hermite_second_order(z, source_ids, level):
    """Independent standard-Gaussian coordinate basis, truncated at degree two.

    linear: z_i. additive2 adds source-local quadratics. interaction2 adds all
    cross-source products. These are finite encoding models, not a full ANOVA.
    """
    z = matrix(z)
    if len(source_ids) != z.shape[1] or level not in ('linear', 'additive2', 'interaction2'):
        raise ValueError('Basis schema/level')
    columns = [z[:, i] for i in range(z.shape[1])]
    terms = [{'coordinates': [i], 'family': 'linear'} for i in range(z.shape[1])]
    if level != 'linear':
        for i in range(z.shape[1]):
            columns.append((z[:, i]**2-1)/np.sqrt(2))
            terms.append({'coordinates': [i, i], 'family': 'source_local_quadratic'})
        for i, j in combinations(range(z.shape[1]), 2):
            same = source_ids[i] == source_ids[j]
            if same or level == 'interaction2':
                columns.append(z[:, i]*z[:, j])
                terms.append({'coordinates': [i, j], 'family': 'source_local_quadratic' if same else 'cross_source_product'})
    return np.stack(columns, 1), terms


def encoding_diagnostics(train, val, evaluation, source_ids, alphas=(0., .01, 1., 100., 10000.)):
    """Same train/val/eval for three encoding families; evaluate once selected.

    A difference between independently selected families is a predictive model
    comparison, not a nonnegative variance share or causal interaction effect.
    """
    output = []
    for level in ('linear', 'additive2', 'interaction2'):
        x, terms = hermite_second_order(train[0], source_ids, level)
        vx, _ = hermite_second_order(val[0], source_ids, level)
        tx, _ = hermite_second_order(evaluation[0], source_ids, level)
        candidates = []
        for alpha in alphas:
            fit = fit_ridge(x, train[1], alpha)
            loss = float(np.mean(np.square(predict_ridge(fit, vx)-val[1])))
            candidates.append((loss, alpha, fit))
        loss, alpha, fit = min(candidates, key=lambda row: (row[0], row[1]))
        output.append({'basis': level, 'feature_count': len(terms), 'selected_alpha': alpha,
                       'validation_mse': loss,
                       'evaluation': representation_score(evaluation[1], predict_ridge(fit, tx), train[1].mean(0)),
                       'not_a_complete_nonlinear_decomposition': True})
    return output
