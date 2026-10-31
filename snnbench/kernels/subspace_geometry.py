"""Independent role encoding subspaces, rank selection and explicit nulls.

Angles are descriptive. Zero validated rank is allowed. SVD vectors inside a
degenerate singular block are not individually identifiable.
"""
import numpy as np
from .decomposition import fit_joint

ROLES = ('focal', 'cotarget', 'unrequested')


def matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite matrix required')
    return x


def svd_geometry(a, rtol=1e-10):
    a = matrix(a)
    if not a.shape[0] or rtol < 0:
        raise ValueError('Positive observation dimension and legal tolerance required')
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    threshold = float(singular[0]*rtol) if len(singular) else 0.
    rank = int(np.sum(singular > threshold))
    positive = singular[:rank]
    return {'u': u, 'singular_values': singular, 'vh': vh, 'numeric_rank': rank,
            'relative_tolerance': rtol,
            'condition_on_nonzero_modes': float(positive[0]/positive[-1]) if rank else None,
            'full_column_rank': rank == a.shape[1],
            'smallest_full_column_singular': float(positive[-1]) if rank and rank == a.shape[1] else None}


def basis(a, rank=None):
    result = svd_geometry(a)
    if rank is None: rank = result['numeric_rank']
    if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= result['numeric_rank']:
        raise ValueError('Basis rank must fit the numerical support')
    return result['u'][:, :rank]


def projector(q):
    q = matrix(q)
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-8):
        raise ValueError('An orthonormal basis is required')
    return q @ q.T


def complement(q):
    q = matrix(q)
    projector(q)
    return np.linalg.svd(q.T, full_matrices=True)[2][q.shape[1]:].T


def overlap(q_a, q_b):
    q_a, q_b = matrix(q_a), matrix(q_b)
    if q_a.shape[0] != q_b.shape[0]: raise ValueError('Same coordinate system required')
    projector(q_a); projector(q_b)
    dim, ra, rb = q_a.shape[0], q_a.shape[1], q_b.shape[1]
    common = {'observation_dim': dim, 'rank_a': ra, 'rank_b': rb,
              'forced_intersection_dimension': max(0, ra+rb-dim)}
    if not ra or not rb:
        return {**common, 'status': 'undefined_empty_subspace', 'principal_angles_degrees': [],
                'overlap_a_into_b': None, 'overlap_b_into_a': None,
                'projector_inner_product': None, 'haar_expected_a_into_b': None}
    cosines = np.clip(np.linalg.svd(q_a.T @ q_b, compute_uv=False), 0., 1.)
    shared = float(np.sum(cosines**2))
    return {**common, 'status': 'defined', 'principal_angles_degrees': np.degrees(np.arccos(cosines)).tolist(),
            'projector_inner_product': shared, 'overlap_a_into_b': shared/ra,
            'overlap_b_into_a': shared/rb, 'haar_expected_a_into_b': rb/dim,
            'haar_expected_b_into_a': ra/dim,
            'chance_adjusted_a_into_b': (shared/ra-rb/dim)/(1-rb/dim) if rb < dim else None,
            'warning': 'Descriptive geometry, not information loss or causal interference.'}


def covariance_capture(q, covariance):
    q, covariance = matrix(q), matrix(covariance)
    if covariance.shape != (q.shape[0], q.shape[0]): raise ValueError('Covariance shape')
    projector(q)
    values = np.linalg.eigvalsh((covariance+covariance.T)/2)
    if values.min() < -1e-9*max(1., np.max(np.abs(values))):
        raise ValueError('Power fractions require PSD covariance, not an indefinite corrected estimator')
    power = float(np.trace(q.T @ covariance @ q))
    total = float(np.trace(covariance))
    top = float(np.sort(values)[-q.shape[1]:].sum()) if q.shape[1] else 0.
    return {'captured_power': power, 'total_power': total,
            'fraction_of_total_power': power/total if total > 0 else None,
            'fraction_of_best_same_rank_power': power/top if top > 0 else None}


def select_role_rank(fit, role, validation_r, validation_z, rule='paired_one_se'):
    """Rank is predictive compression, not a true intrinsic dimension estimate.

    All other role terms are fixed from the axis-fit split. Selection uses a
    disjoint validation split. The one-SE rule is a predeclared heuristic.
    """
    r, z = matrix(validation_r), matrix(validation_z)
    roles = np.asarray(fit['roles']); mask = roles == role
    a = fit['a'][:, mask]; spectrum = svd_geometry(a)
    zc = z-fit['zmean']
    other_prediction = fit['rmean']+zc[:, ~mask] @ fit['a'][:, ~mask].T
    remaining = r-other_prediction
    errors = []
    for rank in range(spectrum['numeric_rank']+1):
        u = spectrum['u'][:, :rank]
        truncated = u @ u.T @ a
        error = np.mean((remaining-zc[:, mask] @ truncated.T)**2, axis=1)
        errors.append(error)
    means = np.array([v.mean() for v in errors])
    best = int(np.argmin(means)); selected = best
    comparisons = []
    for rank, err in enumerate(errors):
        delta = err-errors[best]
        se = float(np.std(delta, ddof=1)/np.sqrt(len(delta))) if len(delta) > 1 else 0.
        comparisons.append({'rank': rank, 'validation_mse': float(means[rank]),
                            'delta_to_best': float(delta.mean()), 'paired_standard_error': se})
    if rule == 'paired_one_se':
        selected = next(row['rank'] for row in comparisons if row['delta_to_best'] <= row['paired_standard_error']+1e-14)
    elif rule != 'minimum': raise ValueError('Known rank-selection rule required')
    return {'selected_rank': selected, 'minimum_mse_rank': best, 'rule': rule,
            'numeric_rank': spectrum['numeric_rank'], 'candidates': comparisons,
            'not_a_significance_test_or_biological_dimension': True}


def fit_axes(r, z, roles, validation_r, validation_z):
    fit = fit_joint(r, z, roles)
    if not fit['coefficients_identifiable']:
        raise ValueError('Joint content design is rank deficient; coordinate attribution is not identified')
    result = {}
    for role in ROLES:
        a = fit['a'][:, np.asarray(roles) == role]
        choice = select_role_rank(fit, role, validation_r, validation_z)
        spectrum = svd_geometry(a)
        q = basis(a, choice['selected_rank'])
        k = choice['selected_rank']; singular = spectrum['singular_values']
        gap = float((singular[k-1]-singular[k])/singular[k-1]) if 0 < k < spectrum['numeric_rank'] else None
        result[role] = {'a': a, 'q': q, 'selection': choice,
                        'singular_values': spectrum['singular_values'],
                        'condition_on_nonzero_modes': spectrum['condition_on_nonzero_modes'],
                        'full_column_rank': spectrum['full_column_rank'],
                        'selected_boundary_relative_singular_gap': gap,
                        'selected_cut_near_degenerate_block': gap is not None and gap < 1e-3}
    return fit, result


def independent_replica(fit_a, axes_a, r_b, z_b, roles):
    fit_b = fit_joint(r_b, z_b, roles)
    if not fit_b['coefficients_identifiable']: raise ValueError('Replica design is not identifiable')
    results, q_b = {}, {}
    for role in ROLES:
        a = fit_b['a'][:, np.asarray(roles) == role]
        requested = axes_a[role]['q'].shape[1]
        available = svd_geometry(a)['numeric_rank']
        q_b[role] = basis(a, min(requested, available))
        results[role] = {'fixed_selected_rank_from_a': requested, 'replica_available_rank': available,
                         'same_selected_rank_available': available >= requested,
                         'subspace_stability': overlap(axes_a[role]['q'], q_b[role]),
                         'coefficient_relative_difference': float(np.linalg.norm(a-axes_a[role]['a'])/max(np.linalg.norm(axes_a[role]['a']), 1e-30))}
    return fit_b, q_b, results


def random_basis(dim, rank, rng, covariance=None):
    if not 0 <= rank <= dim: raise ValueError('Random basis rank')
    if rank == 0: return np.empty((dim, 0))
    sample = rng.normal(size=(dim, rank))
    if covariance is not None:
        covariance = matrix(covariance)
        if covariance.shape != (dim, dim): raise ValueError('Null covariance shape')
        eig, vec = np.linalg.eigh((covariance+covariance.T)/2)
        tolerance = max(float(np.max(np.abs(eig)))*1e-10, 0.)
        if eig.min() < -max(tolerance, 1e-12): raise ValueError('Null covariance must be PSD')
        if np.sum(eig > tolerance) < rank: raise ValueError('Covariance support too small for requested null rank')
        sample = (vec*np.sqrt(np.maximum(eig, 0.))) @ vec.T @ sample
    return np.linalg.qr(sample, mode='reduced')[0]


def variance_matched_basis(covariance, rank, desired_power, rng):
    """Construct a same-rank, same-total-variance control from random rotations.

    This is NOT a Haar draw conditional on variance. It mixes disjoint low/high
    eigenspaces with a known mixing coefficient; for rank>D/2 use a complement.
    It matches one scalar moment, not signal/noise spectra or neural working point.
    """
    cov = matrix(covariance); dim = len(cov)
    if cov.shape != (dim, dim) or not 0 <= rank <= dim or not np.isfinite(desired_power):
        raise ValueError('Matched covariance, rank and finite target power required')
    eig, vec = np.linalg.eigh((cov+cov.T)/2)
    tolerance = max(1e-12, float(np.max(np.abs(eig)))*1e-9)
    if eig.min() < -tolerance: raise ValueError('PSD covariance required')
    total = float(np.trace(cov))
    if rank == 0:
        if abs(desired_power) > tolerance: raise ValueError('Rank zero has zero power')
        return np.empty((dim, 0))
    if rank == dim:
        if abs(desired_power-total) > tolerance: raise ValueError('Full rank has full covariance power')
        return np.eye(dim)
    if rank > dim//2:
        return complement(variance_matched_basis(cov, dim-rank, total-desired_power, rng))
    low, high = float(eig[:rank].sum()), float(eig[-rank:].sum())
    if desired_power < low-tolerance or desired_power > high+tolerance:
        raise ValueError('Desired power is outside the feasible same-rank interval')
    if abs(high-low) < tolerance: return random_basis(dim, rank, rng)
    fraction = float(np.clip((desired_power-low)/(high-low), 0., 1.))
    rotate_low = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
    rotate_high = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
    q_low = vec[:, :rank] @ rotate_low; q_high = vec[:, -rank:] @ rotate_high
    return np.sqrt(1-fraction)*q_low+np.sqrt(fraction)*q_high


def null_overlaps(q_a, rank_b, covariance, draws=256, seed=781):
    q_a = matrix(q_a); rng = np.random.default_rng(seed)
    if not q_a.shape[1] or not rank_b:
        return {'status': 'undefined_empty_subspace'}
    result = {}
    for family, cov in (('haar_dimension_matched', None), ('covariance_shaped', covariance)):
        values = []
        try:
            for _ in range(draws):
                q = random_basis(q_a.shape[0], rank_b, rng, cov)
                values.append(overlap(q_a, q)['overlap_a_into_b'])
        except ValueError as exc:
            result[family] = {'status': 'not_available', 'reason': str(exc)}; continue
        result[family] = {'status': 'computed', 'draws': draws,
                          'mean': float(np.mean(values)), 'quantiles_025_50_975': np.quantile(values, [.025, .5, .975]).tolist(),
                          'role': 'Reference distribution under declared sampling, not a universal neural null or an adjusted significance test.'}
    return result


def fixed_rank_bootstrap_overlap(r_a, z_a, r_b, z_b, roles, rank_f, rank_g, draws=200, seed=782):
    """Separate resampling of independent fits; ranks held fixed after selection.

    Does not claim post-selection coverage or training-seed uncertainty.
    """
    if not rank_f or not rank_g:
        return {'status': 'undefined_empty_subspace', 'fixed_rank_conditioned': True}
    rng = np.random.default_rng(seed); values = []; invalid = 0
    roles = np.asarray(roles)
    for _ in range(draws):
        ia = rng.integers(len(r_a), size=len(r_a)); ib = rng.integers(len(r_b), size=len(r_b))
        fa = fit_joint(r_a[ia], z_a[ia], roles)
        fb = fit_joint(r_b[ib], z_b[ib], roles)
        if not fa['coefficients_identifiable'] or not fb['coefficients_identifiable']:
            invalid += 1; continue
        try:
            qf = basis(fa['a'][:, roles == 'focal'], rank_f)
            qg = basis(fb['a'][:, roles == 'cotarget'], rank_g)
        except ValueError:
            invalid += 1; continue
        values.append(overlap(qf, qg)['overlap_a_into_b'])
    return {'status': 'computed' if values else 'no_valid_draws', 'requested_draws': draws,
            'valid_draws': len(values), 'invalid_draws': invalid,
            'percentile_95': np.quantile(values, [.025, .975]).tolist() if values else None,
            'fixed_rank_conditioned': True,
            'scope': 'Within-checkpoint scene resampling with selected ranks fixed; not a post-selection or training-seed confidence guarantee.'}
