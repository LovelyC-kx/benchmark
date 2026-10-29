"""Finite-p Gaussian reference without a Gaussian training-weight approximation.

Conditional on a test response x, sign(u_train)*(x.T r_train) has the exact
law A*|g| + offset*sign(g) + B*e. The half-normal magnitude, Rademacher sign
and normal e are independent. Integrating e analytically is Rao--Blackwell
Monte Carlo, not a new universal Gaussian-equivalence theorem.
"""
import numpy as np
from scipy.special import ndtr


def _integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def conditional_parameters(Psi,Phi,Omega,theta,x,response_mean=None):
    psi,phi,om,t,x=map(lambda a:np.asarray(a,float),(Psi,Phi,Omega,theta,x))
    if psi.ndim!=2 or psi.shape[0]!=psi.shape[1] or om.ndim!=2 or om.shape[0]!=om.shape[1] or phi.shape!=(len(psi),len(om)) or t.ndim!=2 or t.shape[0]!=len(om) or x.ndim!=2 or x.shape[1]!=len(psi):raise ValueError('Gaussian covariance/teacher/test schema')
    if not all(np.isfinite(a).all() for a in (psi,phi,om,t,x)):raise ValueError('Finite Gaussian parameters')
    if not np.allclose(psi,psi.T) or not np.allclose(om,om.T):raise ValueError('Symmetric covariances')
    if np.linalg.eigvalsh(om)[0]<=0:raise ValueError('Positive latent covariance; no silent pseudoinverse')
    rem=psi-phi@np.linalg.solve(om,phi.T)
    scale=max(np.linalg.norm(psi,2),1e-300)
    if np.linalg.eigvalsh((rem+rem.T)/2)[0]<-1e-9*scale:raise ValueError('Incompatible joint Gaussian covariance')
    var_u=np.einsum('it,ij,jt->t',t,om,t)
    if np.any(var_u<=0):raise ValueError('Nondegenerate task directions')
    A=(x@phi@t)/np.sqrt(var_u)
    variance=np.einsum('ni,ij,nj->n',x,psi,x)[:,None]
    raw=variance-A*A
    local_scale=np.maximum(np.abs(variance)+A*A,1e-300)
    if np.any(raw < -1e-9*local_scale):raise ValueError('Negative conditional variance')
    # Remove only floating cancellation at a truly degenerate covariance.
    B2=np.where(np.abs(raw)<=1e-13*local_scale,0.,np.maximum(raw,0.))
    mean=np.zeros(len(psi)) if response_mean is None else np.asarray(response_mean,float)
    if mean.shape!=(len(psi),) or not np.isfinite(mean).all():raise ValueError('Response-mean vector')
    return A,np.sqrt(B2),x@mean


def finite_gaussian_forecast(Psi,Phi,Omega,theta,x_reference,y_reference,p,
                             response_mean=None,draws=1024,seed=34231,chunk=128):
    p=_integer(p,'p');draws=_integer(draws,'draws')
    A,B,offset=conditional_parameters(Psi,Phi,Omega,theta,x_reference,response_mean)
    y=np.asarray(y_reference,float)
    if y.shape!=A.shape or not np.isin(y,[-1,1]).all():raise ValueError('Paired +/-1 reference labels')
    zero_mean=np.all(offset==0)
    if p==1 and zero_mean:
        denominator=np.sqrt(A*A+B*B)
        rho=np.divide(A,denominator,out=np.zeros_like(A),where=denominator>0)
        error=.5-y*np.arcsin(np.clip(rho,-1,1))/np.pi
        error[denominator==0]=.5
        return {'samples':p,'predicted_error':float(error.mean()),'per_task_error':error.mean(0),
            'integration':'analytic p=1 arcsine identity','inner_draws':0,'inner_MC_standard_error':0.,
            'scope':'Exact conditional Gaussian risk averaged over the supplied independent reference pool. Reference-pool and fitted-parameter uncertainty remain.'}
    rng=np.random.default_rng(np.random.SeedSequence([seed,p]))
    h=np.sum(np.abs(rng.normal(size=(draws,p))),axis=1)/np.sqrt(p)
    s=np.zeros(draws) if zero_mean else (2*rng.binomial(p,.5,size=draws)-p)/np.sqrt(p)
    af,bf,yf=A.ravel(),B.ravel(),y.ravel();of=np.repeat(offset,A.shape[1])
    estimates=np.empty_like(af);draw_mean=np.zeros(draws)
    for first in range(0,len(af),chunk):
        last=min(first+chunk,len(af));mean=af[first:last,None]*h+of[first:last,None]*s
        sd=bf[first:last,None];margin=yf[first:last,None]*mean
        loss=np.empty_like(mean);positive=np.broadcast_to(sd>0,mean.shape)
        ratio=np.divide(-margin,sd,out=np.zeros_like(mean),where=positive)
        loss[positive]=ndtr(ratio[positive]);loss[~positive]=(margin[~positive]<0)+.5*(margin[~positive]==0)
        estimates[first:last]=loss.mean(1);draw_mean+=loss.sum(0)/len(af)
    errors=estimates.reshape(A.shape)
    return {'samples':p,'predicted_error':float(errors.mean()),'per_task_error':errors.mean(0),
        'integration':'exact half-normal/Rademacher conditional law, normal noise integrated analytically',
        'inner_draws':draws,'inner_MC_standard_error':float(draw_mean.std(ddof=1)/np.sqrt(draws)) if draws>1 else None,
        'inner_Hoeffding_95_radius':float(np.sqrt(np.log(40)/(2*draws))),
        'response_mean_included':not zero_mean,
        'scope':'No CLT approximation to the training weights. Numerical integration, supplied-reference sampling and fitted Gaussian parameter uncertainty are distinct.'}
