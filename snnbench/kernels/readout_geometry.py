"""Noise metric, fixed readout effects and constrained-observation probes."""
import numpy as np
from .linear_reference import fit_ridge, predict_ridge

def projected_variance(w, covariance):
    w, covariance = np.asarray(w,float), np.asarray(covariance,float)
    if (w.ndim != 2 or not w.shape[0] or covariance.shape != (w.shape[1],w.shape[1])
        or not np.isfinite(w).all() or not np.isfinite(covariance).all()):
        raise ValueError('Matched finite linear readout and covariance required')
    return float(np.trace(w @ covariance @ w.T)/w.shape[0])
from .subspace_geometry import matrix, basis, overlap, projector


def covariance(x):
    x = matrix(x)
    if len(x) < 2: raise ValueError('Two independent scenes required')
    centered = x-x.mean(0)
    return centered.T @ centered/(len(x)-1)


def choose_covariance_metric(fit_cov, selection_cov, gammas=(0., .01, .1, .5, 1.), floor_ratio=1e-6):
    """Independent selection by logdet(S)+tr(S^-1 C_val), a covariance score.

    The score uses second moments, not a Gaussian claim about the SNN. Shrinkage
    to identity is coordinate dependent; no general affine-invariance claim.
    """
    fit_cov, selection_cov = matrix(fit_cov), matrix(selection_cov)
    if fit_cov.shape[0] != fit_cov.shape[1] or selection_cov.shape != fit_cov.shape:
        raise ValueError('Matched square covariance matrices required')
    if floor_ratio <= 0: raise ValueError('Positive declared covariance floor required')
    for cov in (fit_cov, selection_cov):
        if not np.allclose(cov, cov.T, atol=1e-10): raise ValueError('Symmetric covariance required')
        if np.linalg.eigvalsh(cov).min() < -1e-10: raise ValueError('PSD within covariance required')
    dim = len(fit_cov); scale = float(np.trace(fit_cov)/dim)
    if scale <= 0:
        return {'status': 'undefined_zero_conditional_covariance', 'metric': None,
                'warning': 'No invented finite SNR for a zero covariance estimate.'}
    candidates = []
    for gamma in gammas:
        if not 0 <= gamma <= 1: raise ValueError('Shrinkage gamma in [0,1] required')
        candidate = (1-gamma)*fit_cov+gamma*scale*np.eye(dim)
        eig, vec = np.linalg.eigh((candidate+candidate.T)/2)
        eig = np.maximum(eig, floor_ratio*scale)
        regularized = (vec*eig) @ vec.T
        inverse = (vec*(1/eig)) @ vec.T
        score = float(np.log(eig).sum()+np.trace(inverse @ selection_cov))
        candidates.append((score, float(gamma), regularized, inverse, (vec*(1/np.sqrt(eig))) @ vec.T, eig))
    best = min(candidates, key=lambda row: (row[0], row[1]))
    return {'status': 'computed', 'gamma': best[1], 'selection_score': best[0],
            'regularized_covariance': best[2], 'metric': best[3], 'inverse_sqrt': best[4],
            'condition_number': float(best[5].max()/best[5].min()),
            'unregularized_rank': int(np.linalg.matrix_rank(fit_cov)), 'trace_scale': scale,
            'floor_ratio': floor_ratio,
            'candidate_scores': [{'gamma': row[1], 'score': row[0]} for row in candidates],
            'warning': 'Average-conditional-covariance metric with explicit regularization. Not full Fisher information or an SNN Bayes error formula.'}


def noise_metric_geometry(a_f, a_g, noise_covariance):
    """Population SPD metric control. Input covariance must already be declared."""
    a_f, a_g, cov = matrix(a_f), matrix(a_g), matrix(noise_covariance)
    if a_f.shape[0] != a_g.shape[0] or cov.shape != (a_f.shape[0], a_f.shape[0]):
        raise ValueError('Matched coding maps and covariance required')
    eig, vec = np.linalg.eigh((cov+cov.T)/2)
    if eig.min() <= 0: raise ValueError('SPD covariance required; singular cases need an explicit policy')
    inverse_sqrt = (vec*(1/np.sqrt(eig))) @ vec.T
    white_f, white_g = inverse_sqrt @ a_f, inverse_sqrt @ a_g
    gram = white_f.T @ white_f
    return {'metric_overlap': overlap(basis(white_f), basis(white_g)),
            'linear_signal_metric_eigenvalues': np.linalg.eigvalsh((gram+gram.T)/2)[::-1],
            'gram': gram,
            'meaning': 'Coding geometry relative to this covariance; Gaussian-MMSE interpretation only in a matching linear Gaussian model.'}


def mapping_diagnostics(w, a_f, a_g, a_u, conditional_covariance):
    w, af, ag, au = map(matrix, (w, a_f, a_g, a_u))
    if w.shape[0] != af.shape[1] or w.shape[1] != af.shape[0]:
        raise ValueError('Readout must predict the fixed focal coordinate order')
    q = w.shape[0]
    mismatch = w @ af-np.eye(q)
    return {'w_a_focal': w @ af, 'w_a_cotarget': w @ ag, 'w_a_unrequested': w @ au,
            'focal_gain_mixing_error_unit_latents': float(np.sum(mismatch**2)/q),
            'cotarget_linear_injection_unit_latents': float(np.sum((w @ ag)**2)/q),
            'unrequested_linear_injection_unit_latents': float(np.sum((w @ au)**2)/q),
            'readout_rowspace_vs_focal_encoding': overlap(basis(w.T), basis(af)),
            'conditional_output_variability': projected_variance(w, conditional_covariance),
            'warning': 'Unit-latent linear terms are descriptors of the fitted encoder. Held-out exact risk keeps cross terms and offsets.'}


def affine_restrict(r, mean, q):
    r, q = matrix(r), matrix(q)
    mean = np.asarray(mean, float)
    if mean.shape != (r.shape[1],) or q.shape[0] != r.shape[1]: raise ValueError('Observation restriction shape')
    return mean+(r-mean) @ projector(q)


def fit_projected_probe(train_r, train_y, val_r, val_y, q, mean, scale,
                        lambdas=(0., 1e-5, 1e-3, .1, 1.)):
    """Common full-observation scale for all projections; alpha=N*lambda.

    No per-axis standardization: the Euclidean metric survives orthogonal basis
    rotations. All original target coordinates remain in the evaluation loss.
    """
    tr, ty, vr, vy, q = map(matrix, (train_r, train_y, val_r, val_y, q))
    mean = np.asarray(mean, float)
    if not np.isfinite(scale) or scale <= 0: raise ValueError('Positive common observation scale required')
    if q.shape[0] != tr.shape[1] or mean.shape != (tr.shape[1],): raise ValueError('Projection shape')
    projector(q)
    x = ((tr-mean)/scale) @ q; vx = ((vr-mean)/scale) @ q
    candidates = []
    for lam in lambdas:
        if lam < 0: raise ValueError('Nonnegative lambda')
        fit = fit_ridge(x, ty, len(x)*lam)
        loss = float(np.mean((predict_ridge(fit, vx)-vy)**2))
        candidates.append((loss, float(lam), fit))
    loss, lam, fit = min(candidates, key=lambda row: (row[0], row[1]))
    coefficient = q @ fit['coef']/scale
    return {'w': coefficient.T, 'b': fit['bias']-mean @ coefficient,
            'selected_lambda': lam, 'alpha': len(x)*lam, 'validation_mse': loss,
            'observed_dimension': q.shape[1], 'output_dimension': ty.shape[1],
            'readout_parameter_count': (q.shape[1]+1)*ty.shape[1],
            'train_count': len(tr), 'common_scale': scale,
            'target_coordinate_count_unchanged': True}


def predict_raw(probe, r):
    return matrix(r) @ probe['w'].T+probe['b']


def metrics(prediction, target):
    p, y = matrix(prediction), matrix(target)
    if p.shape != y.shape: raise ValueError('Matching prediction/target shapes required')
    mse = np.mean((p-y)**2, axis=0); var = np.var(y, axis=0)
    per_r2 = [float(1-e/v) if v > 0 else None for e, v in zip(mse, var)]
    return {'mean_mse': float(mse.mean()), 'per_coordinate_mse': mse.tolist(),
            'per_coordinate_r2': per_r2,
            'mean_r2': float(np.mean(per_r2)) if all(v is not None for v in per_r2) else None,
            'negative_r2_retained': True}


def paired_error_interval(baseline_prediction, changed_prediction, target, draws=1000, seed=918):
    base, changed, target = map(matrix, (baseline_prediction, changed_prediction, target))
    if base.shape != changed.shape or base.shape != target.shape: raise ValueError('Paired evaluation shapes')
    delta = np.mean((changed-target)**2-(base-target)**2, axis=1)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(1)
    return {'paired_mean_delta_mse': float(delta.mean()),
            'sequence_percentile_95': np.quantile(means, [.025, .975]).tolist(),
            'draws': draws, 'scope': 'Independent evaluation scenes within a checkpoint; not training-seed uncertainty.'}
