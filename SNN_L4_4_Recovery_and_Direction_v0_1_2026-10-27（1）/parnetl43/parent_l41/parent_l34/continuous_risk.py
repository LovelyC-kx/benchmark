"""Exact population finite-p MSE of an explicitly defined Hebbian regressor.

No joint-Gaussian assumption. Requires IID train/test and finite mixed moment
E[||y||^2 ||r||^2]. Plug-in estimates are diagnostics, not exact population
numbers. This is not a risk formula for a trained nonlinear image decoder.
"""
import numpy as np


def continuous_moments(x,y):
    x,y=map(lambda a:np.asarray(a,float),(x,y))
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Paired fixed-origin feature and continuous target matrices')
    ynorm=np.sum(y*y,axis=1)
    return {'M':x.T@x/len(x),'C':x.T@y/len(x),
        'T':x.T@(x*ynorm[:,None])/len(x),
        'target_squared_norm':float(ynorm.mean()),'target_dimension':y.shape[1],
        'sample_count':len(x),'scope':'Estimated mixed moments on a separately declared pool.'}


def exact_continuous_risk(M,C,T,target_squared_norm,target_dimension,p,gain=1.):
    M,C,T=map(lambda a:np.asarray(a,float),(M,C,T))
    if C.ndim!=2 or M.shape!=(len(C),len(C)) or T.shape!=M.shape or target_dimension!=C.shape[1] or target_dimension<1 or not isinstance(p,(int,np.integer)) or isinstance(p,bool) or p<1:raise ValueError('Population moment schema and positive integer p')
    if not all(np.isfinite(x).all() for x in (M,C,T)) or not np.isfinite(gain) or not np.isfinite(target_squared_norm) or target_squared_norm<0:raise ValueError('Finite population moments/gain')
    signal=float(np.sum(C*C));asymptotic_prediction=float(np.trace(C.T@M@C));single_prediction=float(np.trace(M@T))
    noise=single_prediction-asymptotic_prediction
    scale=max(abs(single_prediction),abs(asymptotic_prediction),1e-300)
    if noise < -1e-9*scale:raise ValueError('Incompatible mixed moments: training variance cannot be negative')
    noise=max(noise,0.)
    quadratic=asymptotic_prediction+noise/p
    risk=(target_squared_norm-2*gain*signal+gain*gain*quadratic)/target_dimension
    if risk < -1e-9*max(target_squared_norm,1e-300):raise ValueError('Incompatible moments: negative population risk')
    oracle_gain=signal/quadratic if quadratic>0 else 0.
    oracle=(target_squared_norm-signal*oracle_gain)/target_dimension
    return {'samples':int(p),'gain':float(gain),'expected_MSE_per_target':float(max(0.,risk)),
        'population_target_power':float(target_squared_norm/target_dimension),
        'mean_map_cross_term':float(-2*gain*signal/target_dimension),
        'mean_prediction_power':float(gain*gain*asymptotic_prediction/target_dimension),
        'finite_training_variability_power':float(gain*gain*noise/(p*target_dimension)),
        'oracle_common_scalar_gain':float(oracle_gain),'oracle_scalar_expected_MSE':float(oracle),
        'scope':'Exact for the supplied TRUE population moments and W_p=mean(r*y^T), fixed gain and origin. An oracle gain estimated from data requires independent development fitting and its own budget.'}
