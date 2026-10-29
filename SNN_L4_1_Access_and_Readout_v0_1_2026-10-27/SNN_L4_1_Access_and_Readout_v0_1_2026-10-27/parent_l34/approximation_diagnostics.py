"""Prediction-gap decomposition and a plug-in Berry--Esseen diagnostic.

The bound theorem requires true population moments. This estimated diagnostic
does NOT certify its own finite-sample moment error. The held-out gain bound
in paired_readouts uses bounded losses and has a separate valid contract.
"""
import numpy as np


def berry_esseen_diagnostic(moment_x,moment_y,reference_x,p,constant=.56,chunk=16):
    x,y,r=map(lambda a:np.asarray(a,float),(moment_x,moment_y,reference_x))
    if x.ndim!=2 or y.ndim!=2 or r.ndim!=2 or len(x)!=len(y) or x.shape[1]!=r.shape[1] or not np.isin(y,[-1,1]).all() or p<1 or constant<.56:raise ValueError('Independent moment/reference matrices and conservative BE constant')
    rows=[];zero=0;total=0
    for first in range(0,len(r),chunk):
        score=x@r[first:first+chunk].T
        q=score[:,:,None]*y[:,None,:];centered=q-q.mean(0)
        variance=np.mean(centered*centered,axis=0);third=np.mean(np.abs(centered)**3,axis=0)
        positive=variance>0;ratio=np.zeros_like(variance)
        ratio[positive]=third[positive]/variance[positive]**1.5
        rows.append(np.minimum(1.,constant*ratio/np.sqrt(p)))
        zero+=int(np.sum(~positive));total+=variance.size
    bound=np.concatenate(rows,axis=0)
    return {'samples':int(p),'plug_in_mean_BE_bound':float(bound.mean()),
        'per_task_plug_in_BE_bound':bound.mean(0),'estimated_zero_variance_fraction':zero/total,
        'constant':constant,'scope':'A diagnostic using estimated conditional second/third moments, not a certified bound for the original neural distribution. The true-moment theorem has additional finite-third-moment and IID conditions.'}


def prediction_gap(real_risk,moment_clone_real_test,gaussian_train_real_test,
                   gaussian_train_gaussian_test,four_statistic):
    r,tm,gr,gg,e=map(float,(real_risk,moment_clone_real_test,gaussian_train_real_test,gaussian_train_gaussian_test,four_statistic))
    pieces={'training_shape_beyond_task_moments':r-tm,
        'training_task_moment_replacement':tm-gr,
        'reference_test_distribution_replacement':gr-gg,
        'Gaussian_reference_to_four_statistic':gg-e}
    return {'observed_minus_formula':r-e,'components':pieces,
        'arithmetic_closure_error':sum(pieces.values())-(r-e),
        'scope':'Declared distribution-replacement risk contrasts. Finite fit, reference, training and task-average errors remain; the arithmetic decomposition is not causal or uniquely mechanistic attribution.'}
