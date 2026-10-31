"""Calibrated probes and diagnostics; none of these estimates is called MI."""
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,accuracy_score,balanced_accuracy_score,log_loss

def flat(a): return np.asarray(a).reshape(len(a),-1)

def linear_probe(train,val,test,yt,yv,ye,alphas=(.01,1.,100.),scale=False):
    X,Xv,Xe=map(flat,(train,val,test));yt,yv,ye=map(flat,(yt,yv,ye))
    sc=StandardScaler() if scale else None
    if sc is not None:X=sc.fit_transform(X);Xv=sc.transform(Xv);Xe=sc.transform(Xe)
    scores=[];models=[]
    for alpha in alphas:
        m=Ridge(alpha=alpha,solver='lsqr',tol=1e-5).fit(X,yt)
        scores.append(float(np.mean((m.predict(Xv)-yv)**2)));models.append(m)
    k=int(np.argmin(scores));m=models[k];p=m.predict(Xe)
    r2=r2_score(ye,p,multioutput='raw_values')
    # Constant targets have no normalized capacity interpretation.
    active=np.var(ye,axis=0)>1e-12
    return {'r2_mean':float(np.mean(r2[active])) if active.any() else None,
            'r2_sum':float(np.sum(r2[active])),'mse':float(np.mean((p-ye)**2)),
            'active_target_dimensions':int(active.sum()),'alpha':float(alphas[k]),
            'validation_mse':scores[k],'finite_probe_not_total_information':True},m,sc

def class_probe(train,val,test,yt,yv,ye,Cs=(.1,1.,10.)):
    X,Xv,Xe=map(flat,(train,val,test));sc=StandardScaler();X=sc.fit_transform(X);Xv=sc.transform(Xv);Xe=sc.transform(Xe)
    best=None
    for C in Cs:
        m=LogisticRegression(C=C,max_iter=250,random_state=0).fit(X,yt)
        loss=log_loss(yv,m.predict_proba(Xv),labels=m.classes_)
        if best is None or loss<best[0]:best=(loss,C,m)
    m=best[2];p=m.predict(Xe)
    return {'accuracy':float(accuracy_score(ye,p)),
            'balanced_accuracy':float(balanced_accuracy_score(ye,p)),
            'cross_entropy':float(log_loss(ye,m.predict_proba(Xe),labels=m.classes_)),
            'C':best[1],'converged':bool(np.max(m.n_iter_)<m.max_iter)},m,sc

def nonlinear_probe(train,val,test,yt,yv,ye,classification=False,seed=0,max_iter=150):
    X,Xv,Xe=map(flat,(train,val,test));sc=StandardScaler();X=sc.fit_transform(X);Xv=sc.transform(Xv);Xe=sc.transform(Xe)
    # Explicit validation chooses architecture; sklearn early stopping splits TRAIN only.
    best=None
    for alpha in (.001,.1):
        cls=MLPClassifier if classification else MLPRegressor
        m=cls(hidden_layer_sizes=(64,32),alpha=alpha,max_iter=max_iter,random_state=seed,
              early_stopping=True,n_iter_no_change=10).fit(X,yt if classification else flat(yt))
        p=m.predict(Xv)
        loss=log_loss(yv,m.predict_proba(Xv),labels=m.classes_) if classification else np.mean((p-flat(yv))**2)
        if best is None or loss<best[0]:best=(loss,m,alpha)
    m=best[1];p=m.predict(Xe)
    if classification:r={'accuracy':float(accuracy_score(ye,p))}
    else:r={'mse':float(np.mean((p-flat(ye))**2)),
            'r2_mean':float(r2_score(flat(ye),p,multioutput='uniform_average'))}
    r.update(alpha=best[2],seed=seed,iterations=int(m.n_iter_),finite_probe_not_information_upper_bound=True)
    return r,m,sc

def geometry(r,z,q,permutations=0,seed=0):
    X=flat(r);X=X-X.mean(0);z=flat(z);q=flat(q)
    design=np.concatenate([z,q],1);design-=design.mean(0)
    coef=np.linalg.lstsq(design,X,rcond=1e-6)[0]
    A=coef[:z.shape[1]].T;B=coef[z.shape[1]:].T
    cov=X.T@X/max(1,len(X)-1);e=np.linalg.eigvalsh(cov).clip(0)
    rank=np.linalg.matrix_rank(cov,tol=max(1e-12,e.max()*1e-6))
    residual=X-design@coef;noise=residual.T@residual/max(1,len(X)-1)
    eps=max(1e-6,np.trace(noise)/max(1,noise.shape[0])*1e-3)
    signal=A.T@np.linalg.solve(noise+eps*np.eye(noise.shape[0]),A)
    angles=subspace_angles(A,B) if np.linalg.norm(A)>1e-10 and np.linalg.norm(B)>1e-10 else []
    out={'algebraic_rank_numeric_tolerance':int(rank),
         'participation_ratio':float(e.sum()**2/(np.sum(e**2)+1e-12)),
         'euclidean_encoding_subspace_angles_radians':np.asarray(angles).tolist(),
         'target_noise_normalized_eigenvalues':np.linalg.eigvalsh(signal).tolist(),
         'noise_regularization':eps,'coordinate_dependent_descriptors':True}
    if permutations:
        rng=np.random.default_rng(seed);null=[]
        for _ in range(permutations):
            perm=np.concatenate([z[rng.permutation(len(z))],q],1)
            c=np.linalg.lstsq(perm,X,rcond=1e-6)[0][:z.shape[1]].T
            null.append(float(np.linalg.norm(c,'fro')))
        out['target_encoding_norm_null']=null
        out['null_unit']='independent complete sequences; q kept intact'
    return out

def stable_capacity(T,beta,sigma,new_steps=None):
    w=beta**np.arange(T-1,-1,-1,dtype=float);B=np.sum(w*w)
    m=T if new_steps is None else new_steps
    an=w[-m:].sum();ao=w[:-m].sum() if m<T else 0
    den=an*an+ao*ao+sigma*sigma*B
    return {'target':float(an*an/den),'distractor':float(ao*ao/den),
            'current':float(1/(1+sigma*sigma)),'A_new':float(an),'A_old':float(ao),'B':float(B)}

def simulate(x,kind,beta=.8,gain=1.,threshold=1.,dc=True):
    x=np.asarray(x);phi=np.concatenate([np.maximum(x,0),np.maximum(-x,0)],axis=-1)
    v=np.zeros_like(phi[:,0]);trace={k:[] for k in ('pre','post','output','current')}
    for t in range(x.shape[1]):
        I=gain*phi[:,t]*((1-beta) if dc else 1.)
        pre=beta*v+I;s=(pre>=threshold).astype('float32')
        if kind=='continuous':v=pre;o=pre
        elif kind=='threshold':v=pre;o=s
        elif kind=='lif_soft':v=pre-threshold*s;o=s
        elif kind=='lif_hard':v=pre*(1-s);o=s
        else:raise ValueError(kind)
        for k,val in [('pre',pre),('post',v),('output',o),('current',I)]:trace[k].append(val.copy())
    return {k:np.stack(v,1) for k,v in trace.items()}

def observe(trace,kind):
    if kind=='pre':return trace['pre'][:,-1]
    if kind=='post':return trace['post'][:,-1]
    if kind=='last_spike':return trace['output'][:,-1]
    if kind=='count':return trace['output'].sum(1)
    if kind=='trajectory':return trace['output'].reshape(len(trace['output']),-1)
    if kind=='joint':return np.concatenate([trace['post'][:,-1],trace['output'].sum(1)],1)
    raise ValueError(kind)

def paired_interval(values,confidence=.95):
    from scipy.stats import t
    a=np.asarray(values,float)
    if len(a)<2:return {'mean':float(a.mean()),'ci':None,'n':len(a)}
    radius=t.ppf((1+confidence)/2,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    return {'mean':float(a.mean()),'ci':[float(a.mean()-radius),float(a.mean()+radius)],'n':len(a),
            'unit':'independent seeds; paired contrast expected'}
