"""Declared task priors and orthogonalized higher-order task alignment.

The Gaussian latent distribution is REQUIRED for the analytic label residual.
It does not require Gaussian neural responses. Negative cross-fit powers are
kept. No coordinate selection or p-value is based on the final test set.
"""
import numpy as np

KAPPA=np.sqrt(2/np.pi)


def support_tasks(d,count,support,seed=34401):
    if min(d,count,support)<1 or support>d:raise ValueError('Declared legal task support')
    rng=np.random.default_rng(np.random.SeedSequence([seed,d,support]));t=np.zeros((d,count))
    for j in range(count):
        chosen=rng.choice(d,support,replace=False);t[chosen,j]=rng.choice((-1.,1.),support)/np.sqrt(support)
    return t


def gaussian_label_residual(z,theta,Omega=None):
    z,t=map(lambda x:np.asarray(x,float),(z,theta));om=np.eye(z.shape[1]) if Omega is None else np.asarray(Omega,float)
    if z.ndim!=2 or t.ndim!=2 or t.shape[0]!=z.shape[1] or om.shape!=(z.shape[1],z.shape[1]):raise ValueError('Known centered Gaussian latent contract')
    if not all(np.isfinite(x).all() for x in (z,t,om)) or np.linalg.eigvalsh(om)[0]<=0:raise ValueError('Finite latents and positive known covariance')
    variance=np.einsum('it,ij,jt->t',t,om,t)
    if np.any(variance<=0):raise ValueError('Nonzero task directions')
    u=z@t/np.sqrt(variance)
    return np.where(u>=0,1.,-1.)-KAPPA*u


def crossfit_higher_alignment(xa,za,xb,zb,theta,Omega=None,control_encoding=None):
    xa,za,xb,zb,t=map(lambda x:np.asarray(x,float),(xa,za,xb,zb,theta))
    if xa.ndim!=2 or xb.ndim!=2 or xa.shape[1]!=xb.shape[1] or len(xa)!=len(za) or len(xb)!=len(zb):raise ValueError('Two independent moment pools with common coordinates')
    ra,rb=xa.copy(),xb.copy()
    if control_encoding is not None:
        a=np.asarray(control_encoding,float)
        if a.shape!=(xa.shape[1],za.shape[1]):raise ValueError('Independent fixed linear control map')
        ra-=za@a.T;rb-=zb@a.T
    ha=gaussian_label_residual(za,t,Omega);hb=gaussian_label_residual(zb,t,Omega)
    eta_a=ra.T@ha/len(ra);eta_b=rb.T@hb/len(rb)
    cross=np.sum(eta_a*eta_b,axis=0)
    return {'eta_pool_a':eta_a,'eta_pool_b':eta_b,'crossfit_power_per_task':cross,
        'mean_crossfit_power':float(cross.mean()),'single_fit_squared_power_per_task':np.sum(eta_a**2,axis=0),
        'negative_values_retained':True,'known_latent_covariance_used':True,
        'scope':'Conditionally unbiased squared higher-order task alignment for independent pools and a known centered Gaussian latent law. A fixed independent linear control map changes variance, not the population target. This is not mutual information or a certified absence test.'}


def hermite3(z):return (z**3-3*z)/np.sqrt(6)


def hermite_task_mean(theta,linear=.1,cubic=1.):
    t=np.asarray(theta,float)
    if t.ndim!=2 or not np.allclose(np.sum(t*t,axis=0),1.):raise ValueError('Unit task directions in known independent Gaussian coordinates')
    m=KAPPA*(linear*t-cubic*t**3/np.sqrt(6));mg=KAPPA*linear*t;delta=m-mg
    pr=1/np.sum(t**4,axis=0)
    return {'exact_mean':m,'Gaussian_mean':mg,'exact_higher_alignment':delta,
        'task_participation':pr,'projection_on_teacher':np.sum(m*t,axis=0),
        'teacher_projection_identity':KAPPA*(linear-cubic/(np.sqrt(6)*pr)),
        'higher_alignment_squared_norm':np.sum(delta*delta,axis=0)}
