"""G (carrier/spatial access), P (temporal summary), R (head) are separate."""
import numpy as np
MODES=('last','mean','sum','exponential','trajectory')

def temporal_weights(steps,mode,decay=.8):
    if not isinstance(steps,(int,np.integer)) or steps<1 or mode not in MODES:
        raise ValueError('Positive integer horizon and declared summary')
    if mode=='trajectory':raise ValueError('Trajectory is not a scalar weighted summary')
    if mode=='last':w=np.zeros(steps);w[-1]=1
    elif mode=='mean':w=np.ones(steps)/steps
    elif mode=='sum':w=np.ones(steps)
    else:
        if not 0<=decay<=1:raise ValueError('Causal exponential decay in [0,1]')
        w=decay**np.arange(steps-1,-1,-1,dtype=float);w/=w.sum()
    return w

def aggregate(features,mode,decay=.8):
    a=np.asarray(features)
    if a.ndim<3 or a.shape[1]<1 or not np.isfinite(a).all():raise ValueError('Finite scene x time x feature tensor')
    if mode=='trajectory':return a.reshape(len(a),-1)
    w=temporal_weights(a.shape[1],mode,decay)
    return np.einsum('nt...,t->n...',a,w)

def spatial_access(features,objective):
    a=np.asarray(features)
    if a.ndim<3:raise ValueError('Scene/time/channel coordinates required')
    if objective=='classification' and a.ndim>3:return a.reshape(*a.shape[:3],-1).mean(-1)
    if objective not in ('classification','reconstruction','regression'):raise ValueError('Named task contract required')
    return a

def permission_spec(phase,mode,steps,dimension,outputs,decay=.8):
    if phase not in ('output','pre','post') or mode not in MODES or min(steps,dimension,outputs)<1:
        raise ValueError('Declared positive access dimensions')
    d=steps*dimension if mode=='trajectory' else dimension
    return {'carrier':phase,'summary':mode,'horizon':steps,'feature_dimension':d,
            'head_parameters':d*outputs+outputs,'retained_summary_scalars':d,
            'scalar_bookkeeping':'count/clock and exponential denominator are additional',
            'additional_internal_access':phase!='output',
            'temporal_history_access':mode!='last','decay':decay if mode=='exponential' else None,
            'scope':'Logical summary/head counts; a last-state observation may alias existing cell state. Actual runtime registry and allocator memory are different.'}

def output_subspaces(weight,tolerance=None):
    """weight maps column feature vectors to the DECLARED outputs only."""
    w=np.asarray(weight,float)
    if w.ndim!=2 or not np.isfinite(w).all():raise ValueError('Finite output x feature matrix')
    _,s,vt=np.linalg.svd(w,full_matrices=True)
    tol=(s.max(initial=0)*max(w.shape)*np.finfo(float).eps) if tolerance is None else tolerance
    rank=int(np.sum(s>tol));potent=vt[:rank].T;null=vt[rank:].T
    return {'rank':rank,'output_dimension':w.shape[0],'feature_dimension':w.shape[1],
            'potent_basis':potent,'null_basis':null,'singular_values':s,
            'scope':'Exact affine HEAD null space, not concept erasure or future recurrent-system null space.'}

def observability_stack(transition,readout,horizon):
    a,c=np.asarray(transition,float),np.asarray(readout,float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or c.ndim!=2 or c.shape[1]!=len(a) or horizon<1:
        raise ValueError('Linear state/readout and positive horizon')
    rows=[];power=np.eye(len(a))
    for _ in range(horizon):rows.append(c@power);power=power@a
    return np.concatenate(rows)

def refinement_relation(first,second):
    """Only assert inclusions with an explicit deterministic reconstruction."""
    if first['carrier']!=second['carrier'] or first['horizon']!=second['horizon']:
        return 'not established'
    if first['summary']=='trajectory' or first['summary']==second['summary']:return 'first can deterministically reconstruct second'
    if {first['summary'],second['summary']}=={'mean','sum'}:return 'equivalent at this fixed known positive horizon'
    return 'not established; no total ordering asserted'
