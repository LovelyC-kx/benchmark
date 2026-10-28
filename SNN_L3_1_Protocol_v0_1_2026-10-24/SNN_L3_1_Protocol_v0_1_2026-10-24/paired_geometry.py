"""Content response geometry. These descriptors are not mutual information."""
import numpy as np

def _matrix(value):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("A nonempty finite [N,D] matrix required")
    return x

def paired_source_moment(r0, r1):
    """Half the paired second moment. Conditional variance interpretation needs
    independent same-distribution source redraw and all other causes shared."""
    a, b = _matrix(r0), _matrix(r1)
    if a.shape != b.shape:
        raise ValueError("Paired shapes differ")
    delta = a-b
    return delta.T@delta/(2*len(delta))

def linear_target_basis(r_fit, z_fit, relative_tolerance=1e-6):
    """Fit only on geometry_fit. A linear encoding basis, not exhaustive nonlinear
    coding directions. Use protocol units, not per-condition whitening."""
    r, z = _matrix(r_fit), _matrix(z_fit)
    if len(r) != len(z) or not 0 < relative_tolerance < 1:
        raise ValueError("Rows or tolerance invalid")
    coef = np.linalg.lstsq(z-z.mean(0), r-r.mean(0), rcond=relative_tolerance)[0].T
    u, singular, _ = np.linalg.svd(coef, full_matrices=False)
    if len(singular) == 0 or singular[0] <= 1e-12:
        return np.zeros((r.shape[1], 0)), singular
    rank = int((singular > singular[0]*relative_tolerance).sum())
    return u[:, :rank], singular

def aligned_power(moment, basis):
    S, U = np.asarray(moment, dtype=np.float64), np.asarray(basis, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or U.ndim != 2 or len(U) != len(S):
        raise ValueError("Shapes invalid")
    if not np.isfinite(S).all() or not np.isfinite(U).all():
        raise ValueError("Nonfinite geometry")
    if U.shape[1] == 0:
        return None
    if not np.allclose(U.T@U, np.eye(U.shape[1]), atol=1e-8):
        raise ValueError("Basis must be orthonormal")
    return float(np.trace(U.T@S@U))

def paired_geometry_report(target_moment, nuisance_moment, target_basis):
    signal = aligned_power(target_moment, target_basis)
    nuisance = aligned_power(nuisance_moment, target_basis)
    return {
        "target_aligned_target_response": signal,
        "target_aligned_nuisance_response": nuisance,
        "nuisance_to_target_response_ratio": (
            nuisance/signal if signal is not None and signal > 1e-12 else None),
        "target_basis_rank": int(target_basis.shape[1]),
        "scope": "linear encoding directions and paired response; not information capacity",
        "interactions_may_be_included": True,
    }

