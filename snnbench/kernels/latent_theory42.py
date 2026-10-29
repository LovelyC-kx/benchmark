"""Exact reference results with explicit linear/Gaussian and query assumptions."""
import numpy as np

def matrix(a, name='matrix'):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Finite two-dimensional ' + name)
    return a

def covariance(a):
    a = matrix(a, 'covariance')
    if a.shape[0] != a.shape[1] or not np.allclose(a, a.T, atol=1e-10):
        raise ValueError('Symmetric covariance required')
    if np.linalg.eigvalsh(a).min(initial=0) < -1e-9:
        raise ValueError('Positive semidefinite covariance required')
    return (a + a.T) / 2

def gaussian_risk(latent_cov, code, target, observation_noise=None, target_noise=None):
    """Per-coordinate Bayes MSE for jointly Gaussian linear observations.

    No Gaussian assumption: this formula instead describes the best affine
    predictor from the specified second moments, not arbitrary-reader Bayes risk.
    Observation and target noise are independent of each other and latent z.
    """
    s, b, l = covariance(latent_cov), matrix(code), matrix(target)
    if b.shape[1] != len(s) or l.shape[1] != len(s) or not len(l):
        raise ValueError('Compatible code/target and nonempty target')
    n = np.zeros((len(b), len(b))) if observation_noise is None else covariance(observation_noise)
    e = np.zeros((len(l), len(l))) if target_noise is None else covariance(target_noise)
    if n.shape != (len(b), len(b)) or e.shape != (len(l), len(l)):
        raise ValueError('Noise covariance dimensions')
    hh, yh = b @ s @ b.T + n, l @ s @ b.T
    residual = l @ s @ l.T + e - yh @ np.linalg.pinv(hh, hermitian=True) @ yh.T
    risk = float(np.trace(residual) / len(l))
    if risk < -1e-8:
        raise ArithmeticError('Substantially negative conditional variance')
    return max(0., risk)

def fixed_linear_risk(latent_cov, code, target, reader, target_noise=None):
    s, b, l, w = covariance(latent_cov), matrix(code), matrix(target), matrix(reader)
    if b.shape[1] != len(s) or l.shape != (len(w), len(s)) or w.shape[1] != len(b):
        raise ValueError('Compatible frozen linear reader')
    e = np.zeros((len(l), len(l))) if target_noise is None else covariance(target_noise)
    if e.shape != (len(l), len(l)):
        raise ValueError('Target noise dimensions')
    d = l - w @ b
    return float(np.trace(d @ s @ d.T + e) / len(l))

def gaussian_risk_chain(latent_cov, state, access, summary, family_map, target, reader, target_noise=None):
    """Exact telescoping risk for S=A z, O=G S, H=P O, family=A_f H.

    F is the affine family restricted to family_map @ H. The fitted reader
    acts on that same family_map @ H. This control is population, not a
    decomposition estimated from a few empirical probe scores.
    """
    a, g, p, f = map(matrix, (state, access, summary, family_map))
    codes = [a, g @ a, p @ g @ a, f @ p @ g @ a]
    risks = [gaussian_risk(latent_cov, c, target, target_noise=target_noise) for c in codes]
    native = fixed_linear_risk(latent_cov, codes[-1], target, reader, target_noise)
    pieces = dict(state_floor=risks[0], access_gap=risks[1]-risks[0],
                  summary_gap=risks[2]-risks[1], family_gap=risks[3]-risks[2],
                  fitted_gap=native-risks[3])
    if min(pieces.values()) < -1e-8 or abs(sum(pieces.values())-native) > 1e-8:
        raise ArithmeticError('Nested-risk identity failed')
    return {'risks': risks, 'pieces': pieces, 'native_risk': native,
            'assumptions': 'Joint Gaussian linear z; explicitly nested maps; square loss; population moments.'}

def query_readability(code, target, tolerance=1e-9):
    """All continuous z in the linear latent model, not a finite label dataset."""
    b, l = matrix(code), matrix(target)
    if b.shape[1] != l.shape[1]:
        raise ValueError('Common latent dimension')
    w = l @ np.linalg.pinv(b)
    residual = float(np.linalg.norm(l-w@b))
    return {'readable': residual <= tolerance, 'residual': residual,
            'reader': w, 'code_rank': int(np.linalg.matrix_rank(b)),
            'task_rank': int(np.linalg.matrix_rank(l))}

def task_metric(reader, output_metric=None):
    """Exact affine-head output metric; only a local pullback for Jacobians."""
    w = matrix(reader)
    m = np.eye(len(w))/len(w) if output_metric is None else covariance(output_metric)
    if m.shape != (len(w), len(w)):
        raise ValueError('Output metric dimension')
    return w.T @ m @ w

def gaussian_extra_risk_reduction(latent_cov, baseline, extra, target):
    a, b = matrix(baseline), matrix(extra)
    r0 = gaussian_risk(latent_cov, a, target)
    r1 = gaussian_risk(latent_cov, np.concatenate((a,b)), target)
    return {'baseline_risk': r0, 'joint_risk': r1, 'gain': r0-r1,
            'scope': 'Conditional task utility, not native causal use.'}

def participation_ratio(x):
    x = matrix(x); x = x-x.mean(0)
    vals = np.linalg.eigvalsh(x.T@x/len(x)).clip(0)
    return float(vals.sum()**2 / max(float(vals@vals), 1e-30))

def affine_gauge(code, reader, transform):
    b, w, a = map(matrix, (code, reader, transform))
    if a.shape != (len(b),len(b)) or np.linalg.matrix_rank(a) != len(a):
        raise ValueError('Invertible code-coordinate change required')
    return a@b, w@np.linalg.inv(a)

def spectral_ranks(x):
    """Keep singular-value and squared-singular-value entropies distinct."""
    s=np.linalg.svd(matrix(x),compute_uv=False)
    def erank(v):
        if v.sum()==0:return 0.
        p=v[v>0]/v.sum()
        return float(np.exp(-np.sum(p*np.log(p))))
    return {'singular_value_effective_rank':erank(s),'covariance_entropy_rank':erank(s*s)}

def safe_whitener(samples, relative_cutoff=1e-8):
    """Whiten the identified covariance range; never invent null directions."""
    x=matrix(samples);center=x.mean(0);c=x-center;s=c.T@c/len(x)
    val,vec=np.linalg.eigh(s)
    keep=val>max(float(val.max(initial=0))*relative_cutoff,1e-15)
    inverse=np.zeros_like(val);inverse[keep]=1/np.sqrt(val[keep])
    w=(vec*inverse)@vec.T
    return {'center':center,'matrix':w,'rank':int(keep.sum()),'cutoff':relative_cutoff,
        'scope':'Diagnostic coordinate map. Rank-deficient directions remain null; not lossless on unobserved inputs.'}
