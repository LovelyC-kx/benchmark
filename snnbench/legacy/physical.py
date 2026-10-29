"""Shared underlying continuous process for explicit sampling-versus-simulation controls."""
import numpy as np

def physical_sequences(n=128,T=16,duration=1.,latent_tau=.3,noise=.2,
                       latent_dim=4,input_dim=8,seed=0,split='train',interpolate_from=None):
    offsets={'train':0,'val':1,'test':2};rng=np.random.default_rng(np.random.SeedSequence([seed,offsets[split]]))
    dense_steps=1024;dt=duration/dense_steps;rho=np.exp(-dt/latent_tau)
    z=np.empty((n,dense_steps+1,latent_dim),dtype='float32');z[:,0]=rng.normal(size=(n,latent_dim))
    for t in range(1,dense_steps+1):z[:,t]=rho*z[:,t-1]+np.sqrt(1-rho*rho)*rng.normal(size=(n,latent_dim))
    W=np.linalg.qr(np.random.default_rng(901).normal(size=(input_dim,latent_dim)))[0]
    raw=z@W.T+rng.normal(scale=noise,size=(n,dense_steps+1,input_dim))
    times=np.linspace(duration/T,duration,T)
    if interpolate_from is None:
        idx=np.rint(times/duration*dense_steps).astype(int);x=raw[:,idx]
        acquisition='distinct measurement times on a shared fine-grid process; more samples can add evidence'
    else:
        base_times=np.linspace(duration/interpolate_from,duration,interpolate_from)
        idx=np.rint(base_times/duration*dense_steps).astype(int);base=raw[:,idx];x=np.empty((n,T,input_dim))
        for i in range(n):
            for f in range(input_dim):x[i,:,f]=np.interp(times,base_times,base[i,:,f])
        acquisition='interpolated same coarse measurements; extra points are deterministic, not new observations'
    target=z[:,-1];return {'x':x.astype('float32'),'y':target.astype('float32'),'target':target,
        'timestamps':times.astype('float32'),'meta':{'duration_seconds':duration,'sample_dt_seconds':duration/T,
        'latent_tau_seconds':latent_tau,'acquisition':acquisition,'interpolate_from':interpolate_from,
        'target_time_seconds':duration,'no_future_observation':True}}
