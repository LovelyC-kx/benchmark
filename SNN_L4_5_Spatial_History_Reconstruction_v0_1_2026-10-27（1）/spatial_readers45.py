"""Frozen-core decoder controls; spatial axes are explicit, supervision and costs are separate."""
import numpy as np

def spatial_design(features,kernel=1):
    f=np.asarray(features,float)
    if f.ndim!=4 or kernel not in (1,3):raise ValueError('NCHW features; 1 or 3 kernel')
    if kernel==1:return np.moveaxis(f,1,-1)
    a=np.pad(f,((0,0),(0,0),(1,1),(1,1)))
    return np.concatenate([np.moveaxis(a[:,:,dy:dy+f.shape[2],dx:dx+f.shape[3]],1,-1) for dy in range(3) for dx in range(3)],axis=-1)

def ridge_fit(x,y,alpha):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or alpha<=0:raise ValueError('Ridge shapes and positive regularization')
    mu=x.mean(0);scale=np.maximum(x.std(0),1e-5);ym=y.mean(0);z=(x-mu)/scale;t=y-ym
    if x.shape[1]<=len(x):w=np.linalg.solve(z.T@z+alpha*len(x)*np.eye(z.shape[1]),z.T@t)
    else:w=z.T@np.linalg.solve(z@z.T+alpha*len(x)*np.eye(len(x)),t)
    return {'w':w,'mean':mu,'scale':scale,'y_mean':ym}

def ridge_predict(m,x):return ((np.asarray(x,float)-m['mean'])/m['scale'])@m['w']+m['y_mean']

def fit_decoder(features,target,kind,alpha):
    f=np.asarray(features);n,c,h,w=f.shape;y=np.asarray(target)
    if kind=='gap_dense':
        m=ridge_fit(f.mean((2,3)),y.reshape(n,-1),alpha)
    else:
        k=1 if kind=='spatial1' else 3 if kind=='spatial3' else None
        if k is None:raise ValueError('Reader family')
        x=spatial_design(f,k);m=ridge_fit(x.reshape(-1,x.shape[-1]),np.moveaxis(y,1,-1).reshape(-1,y.shape[1]),alpha)
    m['kind']=kind;m['target_shape']=list(y.shape[1:]);m['fit_scene_count']=n;m['alpha']=alpha
    m['parameter_count']=int(m['w'].size+m['y_mean'].size)
    m['persistent_feature_scalars_per_scene']=int(c if kind=='gap_dense' else c*h*w)
    return m

def decode(m,f):
    f=np.asarray(f);n=len(f)
    if m['kind']=='gap_dense':return ridge_predict(m,f.mean((2,3))).reshape(n,*m['target_shape'])
    x=spatial_design(f,1 if m['kind']=='spatial1' else 3);p=ridge_predict(m,x.reshape(-1,x.shape[-1]))
    return np.moveaxis(p.reshape(*x.shape[:-1],m['target_shape'][0]),-1,1)

def select_decoder(train,target,val,val_target,kind,alphas=(.001,.01,.1)):
    candidates=[];models=[]
    for a in alphas:
        m=fit_decoder(train,target,kind,a);p=decode(m,val)
        candidates.append({'alpha':a,'validation_raw_MSE':float(np.mean((p-val_target)**2))});models.append(m)
    best=int(np.argmin([r['validation_raw_MSE'] for r in candidates]));return models[best],candidates

def aggregate(features,view):
    if view=='last':return features[:,-1]
    if view=='mean':return features.mean(1)
    raise ValueError('Declared last/mean access')
