"""Exact reference laws; none of these substitutes a learned SNN Bayes decoder."""
import numpy as np
from scipy.linalg import expm
from scipy.special import erf,ndtr,log_ndtr

def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])

def isotropic_risks(gain,noise,angle=0.):
    if gain<0 or noise<0 or gain==noise==0:raise ValueError('Nonzero observation scale')
    rho=gain/np.hypot(gain,noise)
    return {'Bayes_MSE_per_coordinate':noise**2/(gain**2+noise**2),
            'Bayes_binary_error':float(np.arccos(np.clip(rho,-1,1))/np.pi),
            'fixed_direction_binary_error':float(np.arccos(np.clip(rho*np.cos(angle),-1,1))/np.pi)}

def gaussian_fixed_reader_mse(a_t,sigma_t,a_s,sigma_s,q_t,q_s):
    t,s=np.asarray(q_t,float),np.asarray(q_s,float)
    if min(a_t,sigma_t,a_s,sigma_s)<0 or a_s*a_s+sigma_s*sigma_s<=0 or t.shape!=s.shape or t.ndim!=2 or t.shape[0]!=t.shape[1] or a_t*a_t+sigma_t*sigma_t<=0:
        raise ValueError('Equal orthogonal coordinate systems and nonzero source scale')
    for q in (t,s):
        if not np.allclose(q.T@q,np.eye(len(q))):raise ValueError('Orthogonal orientation')
    alpha=a_t/(a_t*a_t+sigma_t*sigma_t)
    fixed=1+alpha*alpha*(a_s*a_s+sigma_s*sigma_s)-2*alpha*a_s*np.trace(s@t.T)/len(s)
    bayes=sigma_s*sigma_s/(a_s*a_s+sigma_s*sigma_s)
    return {'fixed_MSE':float(fixed),'within_Bayes_MSE':float(bayes),'excess_risk':float(fixed-bayes)}

def loss_moment(coefficients,mean,covariance,target_age):
    """Exact scalar serial-recall loss from arbitrary time means/covariance."""
    a,m,k=np.asarray(coefficients,float),np.asarray(mean,float),int(target_age)
    cov=np.asarray(covariance,float)
    if a.ndim!=1 or a.shape!=m.shape or cov.shape!=(len(a),len(a)) or k<0:
        raise ValueError('Aligned finite lag moments')
    c=a.copy()
    if k<len(c):c[k]-=1
    return float(c@cov@c+(c@m)**2)

def sparse_delay_terms(coefficients,k,p,mu,second_moment):
    a=np.asarray(coefficients,float)
    if a.ndim!=1 or k<0 or not 0<=p<=1 or second_moment<mu*mu:
        raise ValueError('Independent Bernoulli-gated scalar moments')
    other=a.copy();ak=a[k] if k<len(a) else None
    if ak is not None:other[k]=0
    task=p*second_moment*(ak-1)**2 if ak is not None else 0.
    projection=p*second_moment*np.dot(other,other)
    mean_correction=-2*p*p*mu*mu*other.sum() if ak is not None else 0.
    composition=p*p*mu*mu*(a.sum()**2-np.dot(a,a))
    return {'total':float(task+projection+mean_correction+composition),
            'task':float(task),'projection':float(projection),
            'mean_correction':float(mean_correction),'composition':float(composition)}

def markov_parameters(a,b,c,lags):
    a,b,c=map(lambda x:np.asarray(x,float),(a,b,c))
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or b.shape[0]!=len(a) or c.shape[1]!=len(a) or lags<1:
        raise ValueError('State transition, input and output operators')
    w=b.copy();out=[]
    for _ in range(lags):out.append(c@w);w=a@w
    return np.stack(out)

def gaussian_location_information(tuning,noise):
    j,s=np.asarray(tuning,float),np.asarray(noise,float)
    if j.ndim!=2 or s.shape!=(len(j),len(j)) or np.linalg.eigvalsh(s).min()<=0:
        raise ValueError('Positive noise covariance, column task tangents')
    return j.T@np.linalg.solve(s,j)

def continuous_linear_moments(a,j0,sigma0,diffusion,times):
    """Van Loan block exponential, column-vector continuous-time convention."""
    a,j0,s0,d=map(lambda x:np.asarray(x,float),(a,j0,sigma0,diffusion))
    n=len(a)
    if a.shape!=(n,n) or s0.shape!=(n,n) or d.shape!=(n,n) or j0.shape[0]!=n:
        raise ValueError('Linear state and covariance dimensions')
    if np.linalg.eigvalsh(s0).min() < -1e-12 or np.linalg.eigvalsh(d).min() < -1e-12:
        raise ValueError('Positive semidefinite initial/diffusion covariance')
    answer=[]
    for t in times:
        if t<0:raise ValueError('Nonnegative elapsed time')
        block=np.block([[a,d],[np.zeros_like(a),-a.T]])
        e=expm(block*t);f=e[:n,:n];s=f@s0@f.T+e[:n,n:]@f.T;s=(s+s.T)/2
        answer.append({'t':float(t),'tuning':f@j0,'noise':s})
    return answer

def central_metric(plus,minus,delta):
    """Scene x coordinate x time x state finite differences, not SG gradients."""
    p,m=np.asarray(plus,float),np.asarray(minus,float)
    if p.shape!=m.shape or p.ndim!=4 or delta<=0:raise ValueError('Paired perturbation arrays')
    j=(p-m)/(2*delta)
    return {'jacobian':j,'euclidean_pullback':np.einsum('nktd,nltd->ntkl',j,j)}

def threshold_finite_scale_moment(delta):
    """Exact Gaussian z, f(z)=1[z>=0]: central finite-difference square."""
    if not np.isfinite(delta) or delta<=0:raise ValueError('Finite positive perturbation scale')
    crossing=float(erf(delta/np.sqrt(2)))
    return {'delta':float(delta),'event_boundary_crossing_probability':crossing,
            'expected_squared_central_difference':crossing/(4*delta*delta),
            'small_delta_asymptote':1/(2*np.sqrt(2*np.pi)*delta),
            'ordinary_derivative_almost_everywhere':0.,'Bayes_binary_decoding_error_from_event':0.}

def threshold_probability_geometry(z,sigma):
    """Gaussian-noise binary threshold: distribution geometry, not pathwise J."""
    if not np.isfinite(z) or not np.isfinite(sigma) or sigma<=0:raise ValueError('Finite location and positive Gaussian noise')
    a=z/sigma;log_derivative=-.5*a*a-.5*np.log(2*np.pi)-np.log(sigma)
    information=np.exp(2*log_derivative-log_ndtr(a)-log_ndtr(-a))
    return {'event_probability':float(ndtr(a)),'derivative_of_mean':float(np.exp(log_derivative)),
            'Bernoulli_Fisher_information':float(information),
            'pathwise_derivative_almost_everywhere':0.}
