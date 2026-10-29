"""Task-conditioned Hebbian moments for arbitrary square-integrable codes.

Exact moment identities; Gaussian training-weight forecasts are approximate.
These probability identities are not claimed as a newly invented theorem.
"""
import math
import numpy as np


def teachers(d,count,seed=34001):
    if d<1 or count<1:raise ValueError('Positive task dimension/count')
    rng=np.random.default_rng(np.random.SeedSequence([seed,d]))
    t=rng.normal(size=(d,count));t/=np.linalg.norm(t,axis=0)
    return t


def labels(z,t):
    z,t=np.asarray(z,float),np.asarray(t,float)
    if z.ndim!=2 or t.ndim!=2 or z.shape[1]!=t.shape[0] or not np.isfinite(z).all() or not np.isfinite(t).all():raise ValueError('Task/latent matrix schema')
    return np.where(z@t>=0,1.,-1.)


def classification_error(scores,y,tie_policy='half'):
    scores,y=np.asarray(scores,float),np.asarray(y,float)
    if scores.shape!=y.shape or scores.ndim!=2 or not np.isfinite(scores).all() or not np.isin(y,[-1,1]).all():raise ValueError('Finite scores and binary +/-1 labels')
    margin=scores*y;ties=scores==0
    if tie_policy=='half':loss=(margin<0)+.5*ties
    elif tie_policy=='positive':loss=np.where(scores>=0,1.,-1.)!=y
    elif tie_policy=='legacy_strict':loss=margin<0
    else:raise ValueError('Declared tie policy required')
    return {'mean_error':float(np.mean(loss)),'per_task_error':np.mean(loss,axis=0),
        'tie_fraction':float(ties.mean()),'tie_policy':tie_policy,
        'label_positive_fraction_per_task':np.mean(y>0,axis=0)}


def empirical_task_moments(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2 or not np.isfinite(x).all() or not np.isin(y,[-1,1]).all():raise ValueError('Paired response matrix and +/-1 task matrix')
    return {'M':x.T@x/len(x),'m':x.T@y/len(x),'sample_count':len(x),
        'origin':'Fixed independently chosen response origin, not re-centered using evaluation data.'}


def gaussian_task_mean(Phi,Omega,t):
    phi,om,t=map(lambda x:np.asarray(x,float),(Phi,Omega,t))
    variance=np.einsum('it,ij,jt->t',t,om,t)
    if np.any(variance<=0):raise ValueError('Non-degenerate latent tasks')
    return np.sqrt(2/np.pi)*(phi@t)/np.sqrt(variance)


def exact_weight_moments(M,m,p):
    M,m=np.asarray(M,float),np.asarray(m,float)
    if M.ndim!=2 or M.shape[0]!=M.shape[1] or m.ndim!=2 or m.shape[0]!=len(M) or p<1:raise ValueError('Moment dimensions and p')
    cov=(M[None]-np.einsum('it,jt->tij',m,m))/p
    return {'mean_weight':m,'weight_covariance_by_task':cov,
        'scope':'Exact IID training-weight moments for w=mean(y*x), y in +/-1 and a fixed origin.'}


def normal_cdf(a):
    from scipy.special import ndtr
    return ndtr(np.asarray(a,float))


def conditional_forecast(moment,x_reference,y_reference,p):
    """Uses an independent FORECAST REFERENCE pool, never evaluation outcomes.

    a=y m^T x; v=[x^T M x-(m^T x)^2]/p are exact conditional
    mean/variance under the true moments. A normal approximation to the
    training weights gives the local error forecast. Cantelli bounds are
    population statements; plug-in sample moment bounds are diagnostic only.
    """
    x,y=np.asarray(x_reference,float),np.asarray(y_reference,float)
    m,M=moment['m'],moment['M']
    if x.ndim!=2 or y.shape!=(len(x),m.shape[1]) or x.shape[1]!=len(M):raise ValueError('Independent forecast pool schema')
    mean_score=x@m;mean_margin=y*mean_score
    raw_variance=(np.einsum('ni,ij,nj->n',x,M,x)[:,None]-mean_score**2)/p
    reference_second=np.einsum('ni,ij,nj->n',x,M,x)[:,None]/p
    scale=max(float(np.max(np.abs(reference_second))),float(np.max(mean_score**2/p)),1e-300)
    if np.min(raw_variance)<-1e-9*scale:raise ValueError('Incompatible task second moments; variance cannot be materially negative')
    variance=np.maximum(raw_variance,0)
    normal_error=np.empty_like(mean_margin);positive=variance>0
    normal_error[positive]=normal_cdf(-mean_margin[positive]/np.sqrt(variance[positive]))
    normal_error[~positive]=(mean_margin[~positive]<0)+.5*(mean_margin[~positive]==0)
    lower=np.zeros_like(mean_margin);upper=np.ones_like(mean_margin)
    good=(mean_margin>0)&positive;bad=(mean_margin<0)&positive
    upper[good]=variance[good]/(variance[good]+mean_margin[good]**2)
    lower[bad]=mean_margin[bad]**2/(variance[bad]+mean_margin[bad]**2)
    deterministic=~positive;lower[deterministic]=normal_error[deterministic];upper[deterministic]=normal_error[deterministic]
    return {'samples':int(p),'normal_weight_forecast':float(normal_error.mean()),
        'per_task_normal_forecast':normal_error.mean(0),'moment_Cantelli_lower':float(lower.mean()),
        'moment_Cantelli_upper':float(upper.mean()),
        'zero_conditional_variance_fraction':float(deterministic.mean()),
        'scope':'Normal forecast is approximate. Cantelli bounds require true population moments; with estimated moments they are plug-in diagnostics, not certified coverage.'}


def nonlinear_alignment(x,z,y,t,geometry):
    x,z,y=np.asarray(x,float),np.asarray(z,float),np.asarray(y,float)
    phi,om=geometry['Phi'],geometry['Omega'];a=np.linalg.solve(om,phi.T).T
    m= x.T@y/len(x);yz=z.T@y/len(z);linear=a@yz
    residual=(x-z@a.T).T@y/len(x);gaussian=gaussian_task_mean(phi,om,t)
    return {'empirical_task_mean':m,'Gaussian_equivalent_task_mean':gaussian,
        'finite_fit_residual_label_alignment':residual,
        'finite_moment_identity_max_error':float(np.max(np.abs(m-linear-residual))),
        'mean_squared_task_direction_difference':float(np.mean(np.sum((m-gaussian)**2,axis=0))),
        'mean_squared_residual_label_alignment':float(np.mean(np.sum(residual**2,axis=0))),
        'scope':'Residual alignment may contain estimation error and higher-order task signal. Independent moments/replicas and Gaussian-matched controls distinguish them.'}


def hebbian_curves(train_x,train_y,test_x,test_y,sample_sizes=(1,8,32,128),repeats=16,seed=34003):
    tx,ty,ex,ey=map(lambda x:np.asarray(x,float),(train_x,train_y,test_x,test_y))
    if tx.ndim!=2 or ex.ndim!=2 or ty.shape[0]!=len(tx) or ey.shape[0]!=len(ex) or ty.shape[1]!=ey.shape[1]:raise ValueError('Readout train/evaluation schema')
    if len(tx)<max(sample_sizes) or repeats<2:raise ValueError('Enough independent training examples and repeated splits required')
    rng=np.random.default_rng(seed);curves={p:[] for p in sample_sizes};ties={p:[] for p in sample_sizes};legacy={p:[] for p in sample_sizes}
    for _ in range(repeats):
        idx=rng.permutation(len(tx))[:max(sample_sizes)]
        for p in sample_sizes:
            w=tx[idx[:p]].T@ty[idx[:p]]/p;scores=ex@w
            result=classification_error(scores,ey)
            curves[p].append(result['per_task_error']);ties[p].append(result['tie_fraction'])
            legacy[p].append(classification_error(scores,ey,'legacy_strict')['mean_error'])
    return [{'samples':int(p),'mean_error':float(np.mean(curves[p])),
        'per_task_mean_error':np.mean(curves[p],axis=0),
        'per_readout_split_mean_error':np.mean(curves[p],axis=1),
        'std_across_readout_splits':float(np.std(np.mean(curves[p],axis=1),ddof=1)),
        'tie_fraction':float(np.mean(ties[p])),'legacy_strict_error':float(np.mean(legacy[p])),
        'tie_policy':'half','new_task_label_budget':int(p),
        'scope':'Repeated readout splits conditional on one frozen train pool, not independent trained SNN seeds.'} for p in sample_sizes]
