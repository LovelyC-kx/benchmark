"""Shared-scene temporal readers; independent-reader risks are not ensembles."""
import hashlib
import numpy as np

def fit_head(x,y,alpha=1e-3):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or alpha<=0:
        raise ValueError('Paired matrices and fixed positive ridge')
    xm,ym=x.mean(0),y.mean(0);u,v=x-xm,y-ym
    w=np.linalg.solve(u.T@u/len(x)+alpha*np.eye(x.shape[1]),u.T@v/len(x))
    return w,xm,ym

def fit_time_heads(x,y,indices,valid_times,alpha=1e-3):
    n,t,d=x.shape;r=len(indices);o=y.shape[1]
    weights=np.empty((r,t,d,o));origins=np.empty((r,t,d));bias=np.empty((r,t,o))
    pooled=[];pooled_origin=[];pooled_bias=[]
    for i,ids in enumerate(indices):
        for j in range(t):weights[i,j],origins[i,j],bias[i,j]=fit_head(x[ids,j],y[ids],alpha)
        # Each content has its own legal time mask; unique labels remain len(ids).
        pws=[];pos=[];pbs=[]
        for columns,times in valid_times:
            a=x[ids][:,times].reshape(-1,d);b=np.repeat(y[ids][:,columns],len(times),axis=0)
            w,m,bm=fit_head(a,b,alpha);pws.append(w);pos.append(m);pbs.append(bm)
        pooled.append(pws);pooled_origin.append(pos);pooled_bias.append(pbs)
    return {'weights':weights,'origins':origins,'bias':bias,
            'pooled_weights':pooled,'pooled_origins':pooled_origin,'pooled_bias':pooled_bias,
            'alpha':alpha,'unique_labels_per_reader':len(indices[0]),
            'pooled_time_rows_not_independent_scenes':True}

def score(w,b,x):return np.einsum('nd,rdo->rno',np.asarray(x,float),w)+b[:,None]

def loss_by_group(pred,target,binary_columns,continuous_columns):
    p,y=np.asarray(pred),np.asarray(target)
    bc,cc=np.asarray(binary_columns),np.asarray(continuous_columns)
    margin=p[:,:,bc]*y[None,:,bc]
    binary=(margin<0).astype(float)+.5*(p[:,:,bc]==0)
    continuous=(p[:,:,cc]-y[None,:,cc])**2
    return binary.mean((0,2)),continuous.mean((0,2))

def paired_bound(a,b,family_size,delta=.05):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.ndim!=1 or not len(a) or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())< -1e-12 or max(a.max(),b.max())>1+1e-12:
        raise ValueError('Independent-scene bounded paired losses')
    if not isinstance(family_size,(int,np.integer)) or family_size<1 or not 0<delta<1:
        raise ValueError('Positive integer comparison family and delta in (0,1)')
    # Repeated averaging can put a mathematical unit loss a few ulps above 1.
    # Clip only roundoff-sized excursions; never admit genuinely unbounded MSE.
    a,b=np.clip(a,0.,1.),np.clip(b,0.,1.)
    radius=np.sqrt(2*np.log(2*family_size/delta)/len(a));gain=float(np.mean(a-b))
    return {'gain':gain,'interval':[float(gain-radius),float(gain+radius)],
            'family_size':family_size,'delta':delta,'independent_scenes':len(a)}

def tree_hash(tree):
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,np.ndarray):h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k,w in sorted(v.items()):h.update(str(k).encode());walk(w)
        elif isinstance(v,(list,tuple)):
            for w in v:walk(w)
        else:h.update(str(v).encode())
    walk(tree);return h.hexdigest()
