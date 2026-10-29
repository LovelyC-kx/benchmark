"""Identity, availability and independent-scene contracts for temporal probes."""
import numpy as np

def independent_ids(*groups):
    seen=set()
    for group in groups:
        ids=np.asarray(group)
        if ids.ndim!=1 or len(np.unique(ids))!=len(ids):
            raise ValueError('Unique scene IDs within each split')
        if seen.intersection(ids.tolist()):
            raise ValueError('Scene leakage between estimation/selection/evaluation')
        seen.update(ids.tolist())
    return True

def source_clock(event_times,present,steps):
    e,p=np.asarray(event_times),np.asarray(present,bool)
    if e.ndim!=2 or e.shape!=p.shape or not np.issubdtype(e.dtype,np.integer) or steps<1:
        raise ValueError('Scene x source integer onsets and presence')
    if np.any((e[p]<0)|(e[p]>=steps)):raise ValueError('Present source outside observation horizon')
    age=np.arange(steps)[None,:,None]-e[:,None,:]
    available=(age>=0)&p[:,None,:]
    return {'age':age,'available':available,'onset':e.copy(),
            'interpretation':'post-update at t includes an event at t; negative age is not forgetting'}

def event_age_rows(x,onsets,ages):
    a,e=np.asarray(x),np.asarray(onsets)
    ages=np.asarray(ages)
    if a.ndim!=3 or e.shape!=(len(a),) or not np.issubdtype(e.dtype,np.integer):
        raise ValueError('Scene x time x features, one integer onset per scene')
    if ages.ndim!=1 or not np.issubdtype(ages.dtype,np.integer) or np.any(ages<0):
        raise ValueError('Nonnegative integer event ages')
    times=e[:,None]+ages[None]
    if times.min()<0 or times.max()>=a.shape[1]:raise ValueError('No edge padding or future extrapolation')
    return a[np.arange(len(a))[:,None],times]

def source_at_rank(event_times,present,rank):
    e,p=np.asarray(event_times),np.asarray(present,bool)
    if e.shape!=p.shape or e.ndim!=2 or rank<0 or rank>=e.shape[1]:raise ValueError('Source rank')
    order=np.argsort(np.where(p,e,np.iinfo(np.int64).max),axis=1,kind='stable')
    out=order[:,rank]
    if not p[np.arange(len(p)),out].all():raise ValueError('Requested rank absent in some scenes')
    return out

def prefix_invariant(a,b,stop):
    x,y=np.asarray(a),np.asarray(b)
    if x.shape[0]!=y.shape[0] or x.shape[2:]!=y.shape[2:] or not 0<stop<=min(x.shape[1],y.shape[1]):
        raise ValueError('Compatible causal prefix')
    return bool(np.array_equal(x[:,:stop],y[:,:stop]))
