"""L3.5--3.7: carrier, cross-time access and communication diagnostics.

Existing regression/Procrustes/Fisher tools, independently implemented for a
controlled SNN contract. Statistical predictability is not a causal estimate.
"""
import numpy as np


def carrier_inventory(carriers):
    result={}
    for name,value in carriers.items():
        x=np.asarray(value)
        if x.ndim<2 or not np.isfinite(x).all():raise ValueError('Finite carrier arrays')
        result[name]={'shape':list(x.shape),'stored_bytes':int(x.nbytes),
            'bytes_per_scene_in_this_export':int(x[0].nbytes),
            'nonzero_fraction':float(np.mean(x!=0)),
            'zero_vector_fraction':float(np.mean(np.all(x==0,axis=-1))),
            'dtype':str(x.dtype),'scope':'Exported array storage, not hardware energy, entropy, or the model persistent-state budget.'}
    return result


def count_matched_timing(n=128,steps=16,neurons=4,seed=35101):
    """Exactly one spike/channel; binary labels differ only in spike time."""
    if steps<8 or steps%4 or n%2:raise ValueError('Even n and steps divisible by four')
    rng=np.random.default_rng(seed);y=np.tile([-1.,1.],n//2);rng.shuffle(y)
    raster=np.zeros((n,steps,neurons),np.uint8)
    for i in range(n):
        # Random global within-window timing does not alter exact counts.
        t=int(rng.integers(1,steps//4))+(steps//2 if y[i]>0 else 0)
        raster[i,t,:]=1
    return raster,y[:,None]


def count_preserving_circular_jitter(raster,std,seed=35102):
    """Sample/channel circular shift, conserving every channel's count.

    This changes timing across channels; it is not independent per-spike
    jitter. Circular wrap is declared rather than hidden boundary clipping.
    """
    r=np.asarray(raster)
    if r.ndim!=3 or std<0 or not np.isin(r,[0,1]).all():raise ValueError('Binary N,T,D raster and nonnegative jitter')
    if std==0:return r.copy()
    rng=np.random.default_rng(seed);out=np.empty_like(r)
    for i in range(len(r)):
        for j in range(r.shape[2]):out[i,:,j]=np.roll(r[i,:,j],int(np.rint(rng.normal(0,std))))
    return out


def ridge_fit(x,y,alpha=1e-3):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or alpha<=0:raise ValueError('Paired matrices, strictly positive fixed ridge')
    xm,ym=x.mean(0),y.mean(0);xc=x-xm;yc=y-ym
    w=np.linalg.solve(xc.T@xc/len(x)+alpha*np.eye(x.shape[1]),xc.T@yc/len(x))
    return {'w':w,'xm':xm,'ym':ym,'alpha':alpha}


def ridge_predict(f,x):return (np.asarray(x)-f['xm'])@f['w']+f['ym']


def temporal_generalization(train,evaluation,train_y,evaluation_y,alpha=1e-3,align=False):
    x,e=np.asarray(train,float),np.asarray(evaluation,float);y,ey=np.asarray(train_y,float),np.asarray(evaluation_y,float)
    if x.ndim!=3 or e.ndim!=3 or x.shape[1:]!=e.shape[1:] or len(x)!=len(y) or len(e)!=len(ey):raise ValueError('Independent sequence x time x feature arrays')
    times=x.shape[1];raw=np.empty((times,times));aligned=np.empty_like(raw)
    for s in range(times):
        fit=ridge_fit(x[:,s],y,alpha)
        for t in range(times):
            pred=ridge_predict(fit,e[:,t]);raw[s,t]=np.mean((pred-ey)**2)
            if align:
                # Destination-to-source map is fit on paired TRAIN scenes.
                # Neither evaluation states nor labels enter the alignment.
                destination=x[:,t]-x[:,t].mean(0);source=x[:,s]-x[:,s].mean(0)
                u,_,vh=np.linalg.svd(destination.T@source,full_matrices=False);q=u@vh
                mapped=(e[:,t]-x[:,t].mean(0))@q+x[:,s].mean(0)
                aligned[s,t]=np.mean((ridge_predict(fit,mapped)-ey)**2)
    return {'train_time_by_test_time_mse':raw,'orthogonally_aligned_mse':aligned if align else None,
        'within_time_mse':np.diag(raw),'alignment_fitted_on':'paired training states only' if align else None,
        'scope':'A transfer gap measures fixed-reader mismatch; alignment gain is compatibility with an orthogonal transport, not proof of a unique rotation or memory loss.'}


def reduced_rank_fit(x,y,rank,alpha=1e-3):
    x,y=np.asarray(x,float),np.asarray(y,float);f=ridge_fit(x,y,alpha)
    if rank not in range(min(x.shape[1],y.shape[1])+1):raise ValueError('Rank includes explicit zero model')
    if rank==0:f['w']=np.zeros_like(f['w'])
    else:
        xc=x-f['xm'];metric=xc.T@xc/len(x)+alpha*np.eye(x.shape[1])
        vals,vec=np.linalg.eigh(metric);sqrt_metric=(vec*np.sqrt(vals))@vec.T
        _,_,vh=np.linalg.svd(sqrt_metric@f['w'],full_matrices=False);v=vh[:rank].T
        f['w']=f['w']@v@v.T
    f['rank']=rank;return f


def selected_communication(tx,ty,vx,vy,ex,ey,ranks=None,alpha=1e-3):
    if ranks is None:ranks=range(min(tx.shape[1],ty.shape[1])+1)
    fits={r:reduced_rank_fit(tx,ty,r,alpha) for r in ranks}
    val={r:float(np.mean((ridge_predict(f,vx)-vy)**2)) for r,f in fits.items()}
    chosen=min(val,key=lambda r:(val[r],r));pred=ridge_predict(fits[chosen],ex)
    baseline=float(np.mean((ey-np.mean(ty,axis=0))**2));mse=float(np.mean((pred-ey)**2))
    return {'selected_rank':chosen,'validation_mse_by_rank':val,'evaluation_mse':mse,
        'evaluation_R2_against_training_mean':1-mse/baseline if baseline>0 else None,
        'coefficient':fits[chosen]['w'],'rank_zero_included':0 in fits,
        'scope':'Held-out statistical response prediction with independent rank selection; not causal communication or task information by itself.'}


def gaussian_information_transfer(tuning,noise_covariance,operator,receiver_noise):
    a,s,b,n=map(lambda v:np.asarray(v,float),(tuning,noise_covariance,operator,receiver_noise))
    if a.ndim!=1 or s.shape!=(len(a),len(a)) or b.shape[1]!=len(a) or n.shape!=(len(b),len(b)):raise ValueError('Linear Gaussian transfer schema')
    if np.linalg.eigvalsh(s).min()<=0 or np.linalg.eigvalsh(n).min()<0:raise ValueError('Positive source covariance, PSD independent receiver noise')
    received=b@s@b.T+n;ar=b@a
    if np.linalg.eigvalsh(received).min()<=0:raise ValueError('Nondegenerate receiver covariance; no silent pseudoinverse')
    source=float(a@np.linalg.solve(s,a));target=float(ar@np.linalg.solve(received,ar))
    return {'source_linear_Fisher':source,'receiver_linear_Fisher':target,
        'receiver_no_more_than_source':target<=source+1e-10*max(source,1),
        'scope':'Gaussian location family, constant covariance, known linear operator and independent receiver noise. No attribution of independent Fisher pieces when source noise is correlated.'}
