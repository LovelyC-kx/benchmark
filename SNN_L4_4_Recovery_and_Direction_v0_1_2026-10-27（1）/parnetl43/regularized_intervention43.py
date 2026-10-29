"""Declared covariance metric regularization; exact sample cross-covariance constraint."""
import numpy as np
from intervention_math43 import finite_matrix,orthogonal_range

def fit_covariance_regularized_eraser(h,f,shrinkage=.01):
    h,f=finite_matrix(h),finite_matrix(f)
    if len(h)!=len(f) or len(h)<2 or not 0<shrinkage<=1:raise ValueError('Paired fit data and shrinkage in (0,1]')
    mean=h.mean(0);x=h-mean;z=f-f.mean(0);s=x.T@x/(len(x)-1);c=x.T@z/(len(x)-1);d=x.shape[1];mu=np.trace(s)/d
    if mu<1e-30:
        return {'method':'leace_fixed_covariance_shrinkage','mean':mean,'matrix':np.eye(d),'removed_rank':0,'shrinkage':shrinkage,'fit_cross_covariance_after_norm':0.,'operator_norm':1.,'condition_bound':1.}
    metric=(1-shrinkage)*s+shrinkage*mu*np.eye(d)
    val,vec=np.linalg.eigh((metric+metric.T)/2);white=(vec/np.sqrt(val))@vec.T;root=(vec*np.sqrt(val))@vec.T
    u=orthogonal_range(white@c);p=np.eye(d)-root@u@u.T@white
    return {'method':'leace_fixed_covariance_shrinkage','mean':mean,'matrix':p.T,'removed_rank':u.shape[1],
        'shrinkage':shrinkage,'fit_cross_covariance_after_norm':float(np.linalg.norm(p@c)),
        'operator_norm':float(np.linalg.norm(p,2)),'condition_bound':float(1+np.sqrt(val.max()/val.min())),
        'guarantee':'Exact fit cross-covariance removal in full-rank regularized covariance metric. Not the original unregularized empirical minimum-distortion solution; no population or nonlinear guarantee.'}
