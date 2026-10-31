"""Scene-level regional risks, exact spectral accounting and explicit PSNR order."""
import numpy as np
from scipy.fft import dctn,idctn

def validate(pred,target):
    p=np.asarray(pred,float);y=np.asarray(target,float)
    if p.shape!=y.shape or p.ndim!=4 or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Finite matching NCHW arrays')
    return p,y

def regional_losses(pred,target,masks):
    p,y=validate(pred,target);e=(p-y)**2
    out={'full':e.mean(axis=(1,2,3))}
    for name,mask in masks.items():
        m=np.asarray(mask,bool)
        if m.shape!=p.shape[-2:]:raise ValueError('Spatial mask shape')
        out[name]=e[:,:,m].mean(axis=(1,2)) if m.any() else None
    return out

def frequency_losses(pred,target,masks):
    p,y=validate(pred,target);e=p-y;h,w=e.shape[-2:]
    d=dctn(e,axes=(-2,-1),norm='ortho');low=np.zeros((h,w),bool);low[:h//4,:w//4]=1
    dl=np.where(low,d,0.);dh=d-dl;el=idctn(dl,axes=(-2,-1),norm='ortho');eh=e-el
    total=np.mean(e*e,axis=(1,2,3));a=np.mean(el*el,axis=(1,2,3));b=np.mean(eh*eh,axis=(1,2,3))
    rows={'full_low':a,'full_high':b,'parseval_residual':total-a-b}
    for name,mask in masks.items():
        if not np.any(mask):continue
        u=el[:,:,mask];v=eh[:,:,mask]
        rows[name+'_low']=np.mean(u*u,axis=(1,2));rows[name+'_high']=np.mean(v*v,axis=(1,2))
        rows[name+'_cross']=2*np.mean(u*v,axis=(1,2))
    return rows

def spectral_angle(pred,target):
    p,y=validate(pred,target);pn=np.linalg.norm(p,axis=1);yn=np.linalg.norm(y,axis=1);ok=(pn>1e-12)&(yn>1e-12)
    cos=np.sum(p*y,axis=1)/np.maximum(pn*yn,1e-24);angle=np.arccos(np.clip(cos,-1,1))
    result=[];counts=[]
    for a,m in zip(angle,ok):result.append(float(a[m].mean()) if m.any() else None);counts.append(int(m.sum()))
    return {'scene_mean_radians':result,'valid_pixel_counts':counts,'zero_norm_pixels_excluded':int((~ok).sum())}

def psnr_summary(mse):
    v=np.asarray(mse,float)
    if v.ndim!=1 or not len(v) or np.any(v<0) or not np.isfinite(v).all():raise ValueError('Finite scene MSE >=0')
    positive=v>0;mean=float(v.mean())
    return {'PSNR_of_mean_MSE':float(-10*np.log10(mean)) if mean>0 else None,
        'mean_scene_PSNR':float(np.mean(-10*np.log10(v))) if positive.all() else None,
        'zero_error_scenes':int((~positive).sum()),'infinite_PSNR_if_zero_error':True,'peak_value':1.}

def summary(pred,target,masks,include_spectral=False):
    p,y=validate(pred,target);raw=regional_losses(p,y,masks);clipped=regional_losses(np.clip(p,0,1),y,masks)
    out={'raw_MSE':{k:None if v is None else float(v.mean()) for k,v in raw.items()},
        'clipped_MSE':{k:None if v is None else float(v.mean()) for k,v in clipped.items()},
        'PSNR':{k:None if v is None else psnr_summary(v) for k,v in clipped.items()},
        'out_of_range_fraction':float(np.mean((p<0)|(p>1))),'regions':{k:int(np.sum(v)) for k,v in masks.items()}}
    if include_spectral:
        f=frequency_losses(np.clip(p,0,1),y,masks);out['frequency']={k:float(v.mean()) for k,v in f.items()};out['SAM']=spectral_angle(np.clip(p,0,1),y)
    return out

def bounded_interval(values,family=1,delta=.05,half_width=1.):
    v=np.asarray(values,float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all() or np.any(np.abs(v)>half_width+1e-12):raise ValueError('Bounded independent scene statistics')
    if family<1 or not 0<delta<1 or half_width<=0:raise ValueError('Declared interval contract')
    m=float(v.mean());r=float(half_width*np.sqrt(2*np.log(2*family/delta)/len(v)))
    return {'mean':m,'lower':m-r,'upper':m+r,'n_scenes':len(v),'family':family,'delta':delta,'half_width':half_width}

def transport_map(source_predictions,clean,target,masks):
    return {source:{region:float(np.mean(v)) for region,v in regional_losses(p,clean,masks).items() if v is not None}
        for source,p in source_predictions.items()}
