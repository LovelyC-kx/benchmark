"""State energy vs task-aligned moments, with explicit finite-fit scope."""
import numpy as np


def crossfit_task_spectrum(M,ma,mb,relative_floor=1e-10):
    M,ma,mb=map(lambda a:np.asarray(a,float),(M,ma,mb))
    if M.ndim!=2 or M.shape[0]!=M.shape[1] or ma.ndim!=2 or ma.shape!=mb.shape or ma.shape[0]!=len(M):raise ValueError('Second moment and independent task-mean matrices')
    if not all(np.isfinite(x).all() for x in (M,ma,mb)) or relative_floor<0:raise ValueError('Finite inputs and explicit support tolerance')
    values,vectors=np.linalg.eigh((M+M.T)/2);idx=np.argsort(values)[::-1];values,vectors=values[idx],vectors[:,idx]
    scale=max(float(values[0]),1e-300)
    if values[-1]<-1e-9*scale:raise ValueError('PSD moment matrix')
    values=np.maximum(values,0);active=values>relative_floor*scale
    ga,gb=vectors.T@ma,vectors.T@mb;power=ga*gb
    capacity=np.zeros_like(power);capacity[active]=power[active]/values[active,None]
    return {'mode_values':values,'eigenvectors':vectors,'energy_fraction':values/values.sum() if values.sum()>0 else np.zeros_like(values),
        'crossfit_mode_task_power':power,'crossfit_capacity_proxy':capacity,
        'crossfit_total_capacity_proxy_per_task':capacity.sum(0),
        'active_modes':active,'relative_support_tolerance':relative_floor,
        'scope':'Independent cross-fit task powers in a separately fitted state basis. Signed proxy may exceed capacity bounds because the covariance/basis is estimated; not a population IPC theorem or an absence certificate.'}


def population_linear_capacity(M,m,noise_variance=0.):
    M,m=map(lambda a:np.asarray(a,float),(M,m))
    if noise_variance<0 or M.shape!=(len(m),len(m)):raise ValueError('Covariance shape and nonnegative observation noise')
    values,vectors=np.linalg.eigh((M+M.T)/2);proj=vectors.T@m
    if values[0]<-1e-12*max(values[-1],1e-300):raise ValueError('PSD covariance')
    values=np.maximum(values,0)+noise_variance
    active=values>0
    if np.any(np.abs(proj[~active])>1e-10):raise ValueError('Incompatible nonzero signal in a zero-variance mode')
    terms=np.zeros_like(proj);terms[active]=proj[active]**2/values[active,None] if m.ndim==2 else proj[active]**2/values[active]
    return terms.sum(0)


def harmonic_capacity_reference(M,m):
    M,m=map(lambda a:np.asarray(a,float),(M,m))
    if m.ndim==1:m=m[:,None]
    norm2=np.sum(m*m,axis=0);den=np.einsum('it,ij,jt->t',m,M,m)
    harmonic=np.divide(norm2**2,den,out=np.zeros_like(norm2),where=den>0)
    optimal=population_linear_capacity(M,m)
    return {'harmonic':harmonic,'optimal_linear':optimal,'gap':optimal-harmonic,
        'scope':'Unit-variance centered regression targets and exact population moments. Harmonic value is the best scalar-calibrated readout in the mean/Hebbian direction; not classification accuracy.'}


def pcr_noise_scan(M,m,noise_variances,absolute_cutoff):
    M,m=map(lambda a:np.asarray(a,float),(M,m))
    if m.ndim==1:m=m[:,None]
    if absolute_cutoff<0:raise ValueError('Declared nonnegative absolute PCR cutoff')
    vals,vec=np.linalg.eigh((M+M.T)/2);g=vec.T@m;rows=[]
    for variance in noise_variances:
        if variance<0:raise ValueError('Nonnegative added observation-noise variance')
        v=vals+variance;active=v>absolute_cutoff
        cap=np.sum(g[active]**2/v[active,None],axis=0)
        rows.append({'added_observation_noise_variance':float(variance),'retained_modes':int(active.sum()),
            'PCR_capacity':cap,'unrestricted_capacity':population_linear_capacity(M,m,variance)})
    return {'absolute_cutoff':absolute_cutoff,'rows':rows,
        'scope':'Known fixed representation plus independent isotropic readout noise. PCR support changes with noise; an increase in PCR score is not an increase in unrestricted information.'}
