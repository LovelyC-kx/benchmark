"""Finite operations and explicit reference theory; no surrogate gradients."""
import numpy as np

def finite_matrix(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite scene x feature matrix')
    return x

def orthogonal_range(x,rtol=1e-10):
    x=finite_matrix(x);u,s,_=np.linalg.svd(x,full_matrices=False)
    keep=s>max(float(s.max(initial=0))*rtol,1e-14)
    return u[:,keep]

def fit_eraser(h,f,method='leace',relative_tolerance=1e-10):
    h,f=finite_matrix(h),finite_matrix(f)
    if len(h)!=len(f) or len(h)<2 or method not in ('leace','orth','binary_mp'):raise ValueError('Paired erasure data and declared method')
    mean=h.mean(0);x=h-mean;z=f-f.mean(0);c=x.T@z/(len(x)-1);d=x.shape[1]
    if method=='binary_mp':
        values=np.unique(f)
        if f.shape[1]!=1 or not np.array_equal(values,[-1.,1.]):raise ValueError('MP target must contain both declared binary classes')
        direction=h[f[:,0]>0].mean(0)-h[f[:,0]<0].mean(0)
        u=orthogonal_range(direction[:,None],relative_tolerance);p=np.eye(d)-u@u.T
    elif method=='orth':
        u=orthogonal_range(c,relative_tolerance);p=np.eye(d)-u@u.T
    else:
        s=x.T@x/(len(x)-1);val,vec=np.linalg.eigh((s+s.T)/2)
        mask=val>max(float(val.max(initial=0))*d*np.finfo(float).eps,1e-15)
        iw=np.zeros_like(val);iw[mask]=1/np.sqrt(val[mask])
        root=(vec*np.sqrt(val.clip(0)))@vec.T;white=(vec*iw)@vec.T
        u=orthogonal_range(white@c,relative_tolerance);p=np.eye(d)-root@u@u.T@white
    return {'method':method,'mean':mean,'matrix':p.T,'removed_rank':u.shape[1],
        'fit_cross_covariance_after_norm':float(np.linalg.norm(p@c)),
        'relative_tolerance':relative_tolerance,
        'guarantee':'Fit cross-covariance removal. Continuous target: affine squared-loss access; binary one-hot theory also requires declared convex loss. Not arbitrary nonlinear erasure.'}

def apply_eraser(h,eraser):
    h=finite_matrix(h)
    return (h-eraser['mean'])@eraser['matrix']+eraser['mean']

def random_eraser(reference,seed):
    d=len(reference['mean']);rank=reference['removed_rank']
    q=np.linalg.qr(np.random.default_rng(seed).normal(size=(d,rank)))[0]
    return dict(reference,method='matched_rank_random',matrix=np.eye(d)-q@q.T)

def norm_matched(h,reference,seed):
    h=finite_matrix(h);delta=apply_eraser(h,reference)-h
    v=np.random.default_rng(seed).normal(size=h.shape)
    v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-30)
    return h+v*np.linalg.norm(delta,axis=1,keepdims=True)

def transport(h,donor,eraser):
    h,donor=finite_matrix(h),finite_matrix(donor)
    if h.shape!=donor.shape:raise ValueError('Matched donor shapes')
    return h+(donor-h)@(np.eye(h.shape[1])-eraser['matrix'])

def matched_binary_message(original,target,seed):
    """Exact per-scene/per-time Hamming and emitted-count matching."""
    a,b=np.asarray(original),np.asarray(target)
    if a.shape!=b.shape or a.ndim!=3 or not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():raise ValueError('Binary scene x time x unit messages')
    out=a.copy();rng=np.random.default_rng(seed)
    for row in range(len(a)):
        for t in range(a.shape[1]):
            plus=int(np.sum((a[row,t]==0)&(b[row,t]==1)));minus=int(np.sum((a[row,t]==1)&(b[row,t]==0)))
            if plus:out[row,t,rng.choice(np.flatnonzero(a[row,t]==0),plus,replace=False)]=1
            if minus:out[row,t,rng.choice(np.flatnonzero(a[row,t]==1),minus,replace=False)]=0
    return out

def decomposition(pred,natural,local,target):
    p,n,l,y=map(finite_matrix,(pred,natural,local,target))
    if len({v.shape for v in (p,n,l,y)})!=1:raise ValueError('Same output and label coordinates')
    operator=p-l;scope=l-n;direct=p-n
    op=np.mean(operator**2);sc=np.mean(scope**2);cross=2*np.mean(operator*scope);fidelity=np.mean(direct**2)
    r0=np.mean((n-y)**2);r1=np.mean((p-y)**2)
    bound=2*np.sqrt(r0*fidelity)+fidelity
    return {'operator_fidelity_MSE':float(op),'scope_fidelity_MSE':float(sc),'interaction':float(cross),
        'natural_fidelity_MSE':float(fidelity),'identity_max_abs':float(abs(op+sc+cross-fidelity)),
        'risk_difference':float(r1-r0),'risk_difference_absolute_upper_bound':float(bound),
        'scope':'Measured finite-output Cauchy bound; not a global state Lipschitz assumption.'}

def finite_event_envelope(a,b,cut,beta=.8,threshold=1.,reset='hard',roundoff_epsilon=0.):
    """Observed two-path recurrence bound AFTER the injected post state."""
    if reset not in ('hard','soft'):raise ValueError('Declared reset rule')
    va,vb=np.asarray(a['post']),np.asarray(b['post']);prea,preb=np.asarray(a['pre']),np.asarray(b['pre'])
    ia,ib=np.asarray(a['effective_current']),np.asarray(b['effective_current'])
    sa,sb=np.asarray(a['natural_event']),np.asarray(b['natural_event'])
    times=slice(cut+1,None);previous=slice(cut,-1)
    event=np.abs(sa[:,times]-sb[:,times]);m=np.maximum(np.abs(prea[:,times]),np.abs(preb[:,times])) if reset=='hard' else threshold
    upper=beta*np.abs(va[:,previous]-vb[:,previous])+np.abs(ia[:,times]-ib[:,times])+m*event
    actual=np.abs(va[:,times]-vb[:,times]);violation=float(np.max(actual-upper,initial=0))
    scale=abs(beta)*(np.abs(va[:,previous])+np.abs(vb[:,previous]))+np.abs(ia[:,times])+np.abs(ib[:,times])+np.abs(prea[:,times])+np.abs(preb[:,times])
    allowance=8*roundoff_epsilon*scale
    ma=prea[:,times]-np.asarray(a['theta'])[:,times];mb=preb[:,times]-np.asarray(b['theta'])[:,times]
    protected=np.abs(ma-mb)<np.abs(mb)
    return {'max_recurrence_bound_violation':violation,'beta_used':float(beta),
        'roundoff_epsilon':float(roundoff_epsilon),'max_declared_roundoff_allowance':float(allowance.max(initial=0)),
        'max_excess_after_roundoff_allowance':float(np.max(actual-upper-allowance,initial=0)),
        'future_event_flip_fraction':float(event.mean()) if event.size else 0.,
        'margin_protected_fraction':float(protected.mean()) if protected.size else 0.,
        'flips_inside_strict_margin_bound':int(np.sum(event[protected])),
        'scope':'Hard/soft fixed-beta reference with observed currents. Injection step excluded; no surrogate-gradient guarantee.'}

def gaussian_erasure_utility_floor(rho):
    if not np.isfinite(rho) or abs(rho)>1:raise ValueError('Correlation in [-1,1]')
    return {'correlation':float(rho),'minimum_MSE_for_G_after_independent_F_erasure':float(rho*rho),
        'attainable_with_U_only':True,'assumptions':'F,U independent N(0,1); G=rho F+sqrt(1-rho^2)U; erased code independent of F.'}
