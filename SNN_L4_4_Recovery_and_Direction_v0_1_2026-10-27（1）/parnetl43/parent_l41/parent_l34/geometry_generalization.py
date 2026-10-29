"""Named Wakhloo statistics and independent moment estimates.

Equations are existing theory, not our new result. No silent pseudoinverse,
latent re-scaling, signal clipping or fitted correction to classification error.
"""
import math
import numpy as np


def matrix(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():raise ValueError('Finite nonempty matrix required')
    return x


def sample_geometry(responses,latents):
    x,z=matrix(responses),matrix(latents)
    if len(x)!=len(z) or len(x)<2:raise ValueError('Paired independent samples required')
    xm,zm=x.mean(0),z.mean(0);xc,zc=x-xm,z-zm
    n=len(x)
    return {'Psi':xc.T@xc/n,'Phi':xc.T@zc/n,'Omega':zc.T@zc/n,
        'response_mean':xm,'latent_mean':zm,'sample_count':n,
        'moment_denominator':'N, consistent with the pinned author implementation'}


def geometry_terms(Psi,Phi,Omega):
    psi,phi,om=map(matrix,(Psi,Phi,Omega))
    d=om.shape[0];n=psi.shape[0]
    if psi.shape!=(n,n) or om.shape!=(d,d) or phi.shape!=(n,d):raise ValueError('Covariance dimensions')
    if not np.allclose(psi,psi.T,atol=1e-12) or not np.allclose(om,om.T,atol=1e-12):raise ValueError('Symmetric covariances required')
    op=np.linalg.eigvalsh(om);pp=np.linalg.eigvalsh(psi)
    if op[0]<=max(op[-1],1e-300)*1e-12:
        return {'status':'singular_or_ill_conditioned_latent_covariance','latent_dimension':d,'Omega_eigenvalues':op}
    scale=max(float(np.linalg.norm(psi,2)),1e-300)
    if pp[0]<-1e-10*scale:raise ValueError('Response covariance is not PSD')
    trp=float(np.trace(psi));tro=float(np.trace(om))
    if trp==0:return {'status':'zero_response','PR':None,'c':None,'f':None,'s':None,'inv_f':None,'inv_s':None,'signal_power':0.}
    a=np.linalg.solve(om,phi.T).T
    remainder=psi-a@phi.T;remainder=(remainder+remainder.T)/2
    ev=np.linalg.eigvalsh(remainder)
    if ev[0]<-1e-9*scale:
        return {'status':'incompatible_joint_covariance','remainder_min_eigenvalue':float(ev[0]),'scale':scale}
    signal=float(np.sum(phi*phi));pr=trp*trp/float(np.sum(psi*psi))
    if signal==0:return {'status':'no_linear_crosscovariance','PR':pr,'c':0.,'f':None,'s':None,'inv_f':None,'inv_s':None,'signal_power':0.,'response_dimension':n,'latent_dimension':d}
    gram=phi.T@phi
    inv_f=tro*float(np.trace(gram@np.linalg.solve(om,gram)))/(signal*signal)
    raw_inv_s=tro*float(np.trace(phi.T@remainder@phi))/(signal*signal)
    if raw_inv_s < -1e-9:raise ValueError('Negative noise factor outside numerical tolerance')
    inv_s=max(0.,raw_inv_s)
    return {'status':'computed','PR':pr,'c':signal/(trp*tro),'inv_f':inv_f,'inv_s':inv_s,
        'f':1/inv_f,'s':1/inv_s if inv_s>0 else None,'s_is_infinite':inv_s==0,
        'signal_power':signal,'response_trace':trp,'latent_trace':tro,
        'remainder_min_eigenvalue':float(ev[0]),'noise_inverse_numerical_floor_applied':raw_inv_s<0,
        'linear_encoding':a,'linear_remainder_covariance':remainder,
        'response_dimension':n,'latent_dimension':d,
        'definitions':'f=signal-signal factorization, s=signal-noise factorization; named output avoids author tuple reordering.'}


def predict_error(terms,samples):
    if not isinstance(samples,(int,np.integer)) or samples<1:raise ValueError('Positive number of newly labeled training samples')
    status=terms['status']
    if status in ('zero_response','no_linear_crosscovariance'):
        return {'status':'Gaussian_reference_limit','predicted_error':.5,'samples':int(samples),
                'scope':'Chance limit of the Gaussian/moment reference; not a claim that nonlinear target signal is absent.'}
    if status!='computed':return {'status':status,'predicted_error':None,'samples':int(samples)}
    sample_term=np.pi/(2*samples*terms['c']**2*terms['PR'])
    asymptote=terms['inv_f']+terms['inv_s']-1
    rad=sample_term+asymptote
    if rad < -1e-9:raise ValueError('Invalid theoretical radicand')
    return {'status':'computed','predicted_error':float(np.arctan(np.sqrt(max(0.,rad)))/np.pi),
        'sample_term':float(sample_term),'asymptotic_term':float(asymptote),'samples':int(samples),
        'scope':'Specified task-averaged Hebbian Gaussian-equivalence prediction, not arbitrary reader accuracy.'}


def crossfit_signal_power(ra,za,rb,zb):
    a=sample_geometry(ra,za);b=sample_geometry(rb,zb)
    if a['Phi'].shape!=b['Phi'].shape:raise ValueError('Common observation and latent coordinates required')
    return {'crossfit_power':float(np.sum(a['Phi']*b['Phi'])),
        'single_fit_squared_power':float(np.sum(a['Phi']**2)),
        'negative_crossfit_retained':True,
        'scope':'Independent-fit signal-norm diagnostic only. Ratios/fourth-order f/s are not thereby made unbiased.'}


def gaussian_surrogate(geometry,n,seed):
    terms=geometry_terms(geometry['Psi'],geometry['Phi'],geometry['Omega'])
    if terms['status'] not in ('computed','no_linear_crosscovariance','zero_response'):raise ValueError('Compatible geometry required')
    rng=np.random.default_rng(seed);om=geometry['Omega'];psi=geometry['Psi'];phi=geometry['Phi']
    a=np.linalg.solve(om,phi.T).T
    rem=(psi-a@phi.T);rem=(rem+rem.T)/2
    vals,vec=np.linalg.eigh(rem);scale=max(float(np.linalg.norm(psi,2)),1e-300)
    if vals[0]<-1e-9*scale:raise ValueError('Incompatible Gaussian surrogate covariance')
    z=rng.normal(size=(n,len(om)))@np.linalg.cholesky(om).T
    x=z@a.T+rng.normal(size=(n,len(psi)))@(vec*np.sqrt(np.maximum(vals,0))).T
    return x,z
