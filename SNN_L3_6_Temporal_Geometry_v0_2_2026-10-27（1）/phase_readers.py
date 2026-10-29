"""Causal clock features with an explicit, charged observation permission."""
import numpy as np
from temporal_clocks import finite_prefix_phase

def phase_features(x,phase,valid=None,phase_only=False):
    x,p=np.asarray(x,float),np.asarray(phase,float)
    if x.shape[:-1]!=p.shape or x.ndim!=3 or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('Scene x time x features and finite scene x time phase')
    valid=np.ones(p.shape,bool) if valid is None else np.asarray(valid,bool)
    if valid.shape!=p.shape:raise ValueError('Phase validity shape')
    c=np.where(valid,np.cos(p),0)[...,None];s=np.where(valid,np.sin(p),0)[...,None]
    clock=np.concatenate([c,s,valid[...,None]],-1)
    return clock if phase_only else np.concatenate([x,x*c,x*s,clock],-1)

def clock_views(x,input_event,onsets,frequency=.125):
    x=np.asarray(x,float);n,T,_=x.shape
    t=np.broadcast_to(np.arange(T),(n,T));e=np.asarray(onsets)
    if e.shape!=(n,):raise ValueError('Source onset known to this observer')
    input_clock=finite_prefix_phase(input_event,frequency)
    state_clock=finite_prefix_phase(x.mean(-1),frequency)
    return {'pooled':x,
            'calendar':phase_features(x,2*np.pi*frequency*t),
            'source_age':phase_features(x,2*np.pi*frequency*(t-e[:,None])),
            'input_phase':phase_features(x,input_clock['phase'],input_clock['valid']),
            'state_phase':phase_features(x,state_clock['phase'],state_clock['valid']),
            'phase_only':phase_features(x,input_clock['phase'],input_clock['valid'],True)}

def known_phase_control():
    """Phase is observed side information here, never inferred from future data."""
    from temporal_probes import fit_head
    rng=np.random.default_rng(365008);out={}
    for label,n in [('train',512),('test',4096)]:
        z=rng.normal(size=(n,2));p=rng.uniform(-np.pi,np.pi,n);c,s=np.cos(p),np.sin(p)
        x=np.c_[z[:,0]*c-z[:,1]*s,z[:,0]*s+z[:,1]*c]+.01*rng.normal(size=(n,2))
        features=phase_features(x[:,None],p[:,None])[:,0]
        if label=='train':
            raw=fit_head(x,z);adapt=fit_head(features,z)
        else:
            for key,head,f in [('raw',raw,x),('phase_conditioned',adapt,features)]:
                w,o,b=head;out[key+'_MSE']=float(np.mean(((f-o)@w+b-z)**2))
    out['phase_observation']='Exact supplied current phase; known-answer measurement permission. No biological or learned-SNN oscillation inferred.'
    return out
