"""Task-independent, invertible readout preconditioning selected BEFORE test.

Classical covariance/spectral filters, evaluated as one concrete design use of
the diagnostics. They change the finite Hebbian learner, not information.
"""
import numpy as np
from geometry_generalization import geometry_terms,predict_error
from task_moments import empirical_task_moments,conditional_forecast


def spectral_candidates(Psi,etas=(.1,1.,10.)):
    p=np.asarray(Psi,float);dim=len(p);tau=float(np.trace(p)/dim)
    candidates={'identity':np.eye(dim)}
    if tau<=0:return candidates
    values,vectors=np.linalg.eigh((p+p.T)/2)
    if values.min() < -1e-10*max(tau,1e-300):raise ValueError('PSD covariance required')
    for eta in etas:
        if eta<=0:raise ValueError('Only full-rank regularized filters in this family')
        # Global scalar is irrelevant to exact Hebbian decisions. Normalize
        # largest gain to one to avoid avoidable floating overflow.
        gain=1/np.sqrt(np.maximum(values,0)/tau+eta);gain/=gain.max()
        candidates['spectral_eta_'+str(eta)]=(vectors*gain)@vectors.T
    return candidates


def transported_geometry(geometry,transform):
    b=np.asarray(transform,float)
    return geometry_terms(b@geometry['Psi']@b.T,b@geometry['Phi'],geometry['Omega'])


def select_before_evaluation(geometry,candidates,moment_x,moment_dev_y,reference_x,reference_dev_y,sample_sizes):
    theory={};conditional={};details={}
    for name,b in candidates.items():
        terms=transported_geometry(geometry,b)
        moments=empirical_task_moments(moment_x@b.T,moment_dev_y)
        details[name]={'condition_number':float(np.linalg.cond(b)),'full_rank':int(np.linalg.matrix_rank(b))==len(b),
            'theory':[predict_error(terms,p) for p in sample_sizes],
            'development_task_moment_forecast':[conditional_forecast(moments,reference_x@b.T,reference_dev_y,p) for p in sample_sizes]}
    for i,p in enumerate(sample_sizes):
        eligible=[name for name in candidates if details[name]['theory'][i]['predicted_error'] is not None]
        theory[str(p)]=min(eligible,key=lambda name:(details[name]['theory'][i]['predicted_error'],name!='identity',name)) if eligible else 'identity'
        conditional[str(p)]=min(candidates,key=lambda name:(details[name]['development_task_moment_forecast'][i]['normal_weight_forecast'],name!='identity',name))
    return {'four_statistic_choices':theory,'development_task_moment_choices':conditional,'candidates':details,
        'selection_uses_evaluation_outcomes':False,
        'scope':'Candidates fixed, fit geometry only. Moment selector uses separate DEVELOPMENT task directions; final risk is on held-out directions and scenes. No claim of extra population information.'}
