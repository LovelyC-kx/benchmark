"""Pair-only target-time -> source-time maps with explicit geometry/budgets."""
import numpy as np

def fit_transport(destination,source,family='orthogonal',alpha=1e-3):
    x,y=np.asarray(destination,float),np.asarray(source,float)
    if x.shape!=y.shape or x.ndim!=2 or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite paired scenes, same feature coordinates')
    n,d=x.shape;xm,ym=x.mean(0),y.mean(0);xc,yc=x-xm,y-ym
    identity=np.eye(d);singular=np.linalg.svd(xc.T@yc/n,compute_uv=False)
    rank=int(np.linalg.matrix_rank(xc.T@yc/n));scale=1.
    if family=='raw':w=identity;xm=ym=np.zeros(d)
    elif family=='center':w=identity
    elif family in ('orthogonal','rotation','similarity'):
        u,_,vt=np.linalg.svd(xc.T@yc,full_matrices=False);w=u@vt
        if family=='rotation' and np.linalg.det(w)<0:
            u[:,-1]*=-1;w=u@vt
        if family=='similarity':
            scale=max(0.,float(np.sum((xc@w)*yc)/max(np.sum(xc*xc),1e-30)));w=w*scale
    elif family=='affine':
        if alpha<=0:raise ValueError('Fixed positive affine regularization')
        w=np.linalg.solve(xc.T@xc/n+alpha*identity,xc.T@yc/n)
    else:raise ValueError('Declared map family')
    # No self-alignment processor is fitted on the diagonal.
    if np.array_equal(x,y):w=identity;scale=1.
    pred=(x-xm)@w+ym
    return {'family':family,'w':w,'destination_mean':xm,'source_mean':ym,
            'rank_cross_covariance':rank,'dimension':d,'singular_values_cross_covariance':singular,
            'orthogonal_identifiable_full_rank':rank==d,'determinant':float(np.linalg.det(w)),
            'scale':scale,'alpha':alpha if family=='affine' else None,
            'paired_scenes':n,'map_parameters_stored':int(w.size+xm.size+ym.size),
            'training_state_MSE':float(np.mean((pred-y)**2)),
            'label_access':False,'direction':'destination time -> source time'}

def transport(fit,x):
    return (np.asarray(x,float)-fit['destination_mean'])@fit['w']+fit['source_mean']

def push_reader(fit,w,origin,bias):
    """Compose a transport with source readers, without mapping all test states."""
    w,o,b=np.asarray(w,float),np.asarray(origin,float),np.asarray(bias,float)
    result=np.einsum('de,reo->rdo',fit['w'],w)
    offset=np.einsum('rd,rdo->ro',fit['source_mean'][None]-o,w)+b
    offset-=np.einsum('d,rdo->ro',fit['destination_mean'],result)
    return result,offset

def cycle_error(q_ab,q_bc,q_ac):
    a,b,c=map(lambda x:np.asarray(x,float),(q_ab,q_bc,q_ac))
    if a.shape!=b.shape or b.shape!=c.shape or a.shape[0]!=a.shape[1]:raise ValueError('Equal square maps')
    return float(np.linalg.norm(a@b-c,'fro')/np.sqrt(len(a)))

def affine_parts(fit):
    """Row-vector map x W + b, including the two fitted means."""
    w=np.asarray(fit['w'],float)
    return w,np.asarray(fit['source_mean'])-np.asarray(fit['destination_mean'])@w

def compose_transports(first,second):
    """Apply first then second. Offsets must participate in the cycle."""
    a,b=affine_parts(first);c,d=affine_parts(second)
    return {'w':a@c,'destination_mean':np.zeros(a.shape[0]),'source_mean':b@c+d,
            'family':'composed','label_access':False}

def cycle_diagnostics(first,second,direct,heldout):
    chain=compose_transports(first,second);a,b=affine_parts(chain);c,d=affine_parts(direct)
    p,q=transport(chain,heldout),transport(direct,heldout)
    denom=max(float(np.mean(q*q)),1e-12)
    return {'matrix_disagreement':float(np.linalg.norm(a-c,'fro')/np.sqrt(len(a))),
            'offset_disagreement':float(np.linalg.norm(b-d)/np.sqrt(len(b))),
            'heldout_prediction_MSE':float(np.mean((p-q)**2)),
            'heldout_relative_disagreement':float(np.mean((p-q)**2)/denom),
            'includes_centering_offsets':True}

def principal_cosines(a,b,tol=1e-10):
    """Subspaces only: this intentionally does not identify stimulus orientation."""
    x,y=np.asarray(a,float),np.asarray(b,float)
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0]:raise ValueError('Common state coordinates')
    def basis(z):
        u,s,_=np.linalg.svd(z,full_matrices=False)
        return u[:,s>tol*max(1.,s.max(initial=0))]
    q,r=basis(x),basis(y)
    if min(q.shape[1],r.shape[1])==0:return np.array([])
    return np.clip(np.linalg.svd(q.T@r,compute_uv=False),0,1)
