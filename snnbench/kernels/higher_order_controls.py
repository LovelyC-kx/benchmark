"""Task-specific affine control variates for the orthogonal label residual.

For centered Gaussian z, h=sign(u)-sqrt(2/pi)*u has E[h]=E[z h]=0
and E[(1,z)(1,z)^T h^2]=(1-2/pi)*diag(1,Omega). This gives an
explicit population variance-optimal affine control, estimated independently.
"""
import numpy as np
from .task_structure import gaussian_label_residual,KAPPA


def fit_weighted_affine(x,z,theta,Omega=None):
    x,z,t=map(lambda a:np.asarray(a,float),(x,z,theta));om=np.eye(z.shape[1]) if Omega is None else np.asarray(Omega,float)
    if x.ndim!=2 or len(x)!=len(z):raise ValueError('Paired independent control-fitting pool')
    h=gaussian_label_residual(z,t,om);q=np.c_[np.ones(len(z)),z]
    denominator=np.zeros((q.shape[1],q.shape[1]));denominator[0,0]=1.;denominator[1:,1:]=om;denominator*=1-KAPPA*KAPPA
    numerator=np.einsum('nr,nd,nt->trd',x,q,h*h)/len(x)
    coefficients=np.stack([np.linalg.solve(denominator,n.T).T for n in numerator])
    return coefficients


def contribution_stats(x,z,theta,coefficients,Omega=None):
    x,z,t=map(lambda a:np.asarray(a,float),(x,z,theta));h=gaussian_label_residual(z,t,Omega);q=np.c_[np.ones(len(z)),z]
    c=np.asarray(coefficients,float)
    if c.shape!=(t.shape[1],x.shape[1],q.shape[1]) or len(x)<2:raise ValueError('Task x response x affine-latent coefficient schema')
    mean=[];variance=[]
    for j in range(t.shape[1]):
        value=(x-q@c[j].T)*h[:,j,None]
        mean.append(value.mean(0));variance.append(float(np.var(value,axis=0,ddof=1).sum()))
    return {'eta':np.stack(mean,axis=1),'contribution_variance_per_task':np.array(variance),
        'mean_contribution_variance':float(np.mean(variance)),'independent_scenes':len(x)}


def independent_control_comparison(fit_x,fit_z,validation_x,validation_z,xa,za,xb,zb,theta,Omega=None):
    x,z,t=map(lambda a:np.asarray(a,float),(fit_x,fit_z,theta));q=np.c_[np.ones(len(z)),z]
    ordinary=np.linalg.lstsq(q,x,rcond=None)[0].T
    ordinary=np.repeat(ordinary[None],t.shape[1],axis=0);weighted=fit_weighted_affine(x,z,t,Omega)
    candidates={'none':np.zeros_like(ordinary),'ordinary_linear':ordinary,
        'weighted_half':.5*(ordinary+weighted),'weighted_full':weighted}
    val={k:contribution_stats(validation_x,validation_z,t,c,Omega)['mean_contribution_variance'] for k,c in candidates.items()}
    selected=min(val,key=lambda k:(val[k],k!='ordinary_linear',k));results={}
    for name,c in candidates.items():
        a=contribution_stats(xa,za,t,c,Omega);b=contribution_stats(xb,zb,t,c,Omega)
        results[name]={'crossfit_power_per_task':np.sum(a['eta']*b['eta'],axis=0),
            'mean_crossfit_power':float(np.mean(np.sum(a['eta']*b['eta'],axis=0))),
            'independent_moment_contribution_variance':float((a['mean_contribution_variance']+b['mean_contribution_variance'])/2)}
    base=results['ordinary_linear']['independent_moment_contribution_variance'];after=results[selected]['independent_moment_contribution_variance']
    return {'selected_before_power_pools':selected,'validation_variance':val,'methods':results,
        'independent_variance_reduction_vs_ordinary':1-after/base if base>0 else None,
        'coefficients':candidates,'negative_crossfit_values_retained':True,
        'scope':'All methods estimate the same higher-order alignment. Known centered Gaussian latent law, and fit/selection pools independent of both power pools, are required. Estimated controls are not guaranteed to improve finite-sample variance; negative gains are retained.'}
