"""Analytical references; none of these functions is an SNN capacity theorem."""
import numpy as np

def gaussian_linear_capacity(a, noise_std=0.):
    """r=A z+epsilon; z~N(0,I), epsilon~N(0,sigma^2 I).

    Returns population best-linear (also Gaussian MMSE) per-coordinate R².
    Pseudoinverse is used at sigma=0. Arbitrary nonlinear readouts are outside it.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Finite matrix and nonnegative noise")
    covariance = a @ a.T + noise_std**2*np.eye(a.shape[0])
    w = a.T @ np.linalg.pinv(covariance, rcond=1e-12)
    r2 = np.diag(w @ a)
    sv = np.linalg.svd(a, compute_uv=False)
    spectral = float(np.sum(sv**2/(sv**2+noise_std**2))) if noise_std else float(np.linalg.matrix_rank(a))
    return {"decoder": w, "per_coordinate_r2": r2, "sum_r2": float(r2.sum()),
            "spectral_sum": spectral, "rank": int(np.linalg.matrix_rank(a)),
            "observation_dim": a.shape[0], "target_dim": a.shape[1]}

def linear_age_matrices(transition, input_map, steps):
    """A^(T-1-t)B for zero-initialized linear recursion h[t]=A h[t-1]+B x[t]."""
    a = np.asarray(transition, dtype=float); b = np.asarray(input_map, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape[0] != a.shape[0] or steps < 1:
        raise ValueError("Linear reference shapes")
    return np.stack([np.linalg.matrix_power(a, steps-1-t) @ b for t in range(steps)])

def fit_linear_encoding(r, z):
    """Independent geometry-fit split only. r = mean + z @ A.T + residual."""
    r = np.asarray(r, dtype=float); z = np.asarray(z, dtype=float)
    if r.ndim != 2 or z.ndim != 2 or len(r) != len(z) or len(r) < 2:
        raise ValueError("Encoding data shapes")
    zmean = z.mean(0); rmean = r.mean(0)
    a = np.linalg.lstsq(z-zmean, r-rmean, rcond=1e-10)[0].T
    return {"a": a, "zmean": zmean, "rmean": rmean}

def risk_accounting(r, z, w, b, encoding, target_indices):
    """Exact held-out linear-head MSE identity including every cross term.

    z: N x M all factors; w: Q x D; selected true target is z[:, indices].
    This is algebraic accounting, not causal mediation or a nonlinear decomposition.
    """
    r=np.asarray(r,dtype=float); z=np.asarray(z,dtype=float)
    w=np.asarray(w,dtype=float); b=np.asarray(b,dtype=float)
    idx=np.asarray(target_indices,dtype=int)
    if len(idx)!=len(set(idx.tolist())) or np.any(idx<0) or np.any(idx>=z.shape[1]):
        raise ValueError("Unique valid target indices required")
    other=np.array([i for i in range(z.shape[1]) if i not in set(idx)],dtype=int)
    a=encoding['a']; centered=z-encoding['zmean']
    residual=r-encoding['rmean']-centered @ a.T
    u=centered[:,idx] @ (w @ a[:,idx]-np.eye(len(idx))).T
    v=centered[:,other] @ (w @ a[:,other]).T
    remainder=residual @ w.T+encoding['rmean'] @ w.T+b-encoding['zmean'][idx]
    direct=r @ w.T+b-z[:,idx]
    terms={'target_encoding_error':float(np.mean(u*u)),
           'nontarget_injection':float(np.mean(v*v)),
           'residual_and_offset':float(np.mean(remainder*remainder)),
           'cross_target_nontarget':float(2*np.mean(u*v)),
           'cross_target_residual':float(2*np.mean(u*remainder)),
           'cross_nontarget_residual':float(2*np.mean(v*remainder))}
    return {'terms':terms,'accounted_mse':float(sum(terms.values())),
            'direct_mse':float(np.mean(direct*direct)),
            'max_pointwise_identity_error':float(np.max(np.abs(direct-u-v-remainder)))}

def fit_ridge(x,y,alpha):
    """Centered ridge; alpha multiplies coefficient penalty against summed SSE."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if not np.isfinite(alpha) or alpha<0 or x.ndim!=2 or y.ndim!=2 or len(x)!=len(y):
        raise ValueError("Ridge input")
    xm=x.mean(0);ym=y.mean(0);xc=x-xm;yc=y-ym
    if alpha==0:coef=np.linalg.lstsq(xc,yc,rcond=1e-10)[0]
    else:coef=np.linalg.solve(xc.T @ xc+alpha*np.eye(x.shape[1]),xc.T @ yc)
    return {'coef':coef,'bias':ym-xm @ coef,'alpha':float(alpha),'input_mean':xm}

def predict_ridge(fit,x):return np.asarray(x) @ fit['coef']+fit['bias']

def regression_metrics(pred,y):
    pred=np.asarray(pred,dtype=float);y=np.asarray(y,dtype=float)
    if pred.shape!=y.shape or pred.ndim!=2:raise ValueError("Metric shapes")
    mse=np.mean((pred-y)**2,axis=0);var=np.var(y,axis=0)
    r2=np.divide(mse,var,out=np.full_like(mse,np.nan),where=var>0)
    r2=1-r2
    return {'per_coordinate_mse':mse,'per_coordinate_r2':r2,
            'mean_mse':float(mse.mean()),'sum_r2':float(np.sum(r2)),
            'note':'Negative finite-sample R² retained; not a population capacity estimate.'}
