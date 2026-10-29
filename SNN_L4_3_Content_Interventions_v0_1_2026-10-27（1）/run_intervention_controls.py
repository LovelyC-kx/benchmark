"""Exact references for selective operations, task utility and SNN boundaries."""
import argparse
from pathlib import Path
import numpy as np
from bridge43 import write_json
from intervention_math43 import (fit_eraser,apply_eraser,transport,norm_matched,matched_binary_message,
    decomposition,gaussian_erasure_utility_floor)

def run():
    rng=np.random.default_rng(43043);z=rng.normal(size=(8192,3));h=np.c_[z[:,0]+2*z[:,1],z[:,1],z[:,2]];f=z[:,0:1]
    leace=fit_eraser(h,f);orth=fit_eraser(h,f,'orth');el,eo=apply_eraser(h,leace),apply_eraser(h,orth)
    empirical={'leace_cross_covariance_norm':leace['fit_cross_covariance_after_norm'],'orth_cross_covariance_norm':orth['fit_cross_covariance_after_norm'],
        'leace_displacement_MSE':float(np.mean((el-h)**2)),'orth_displacement_MSE':float(np.mean((eo-h)**2))}
    # Centered finite symmetric population carries F^2 after linear F is erased.
    v=np.array([-2.,-1.,1.,2.]);code=np.c_[v,v*v];e=fit_eraser(code,v[:,None]);erased=apply_eraser(code,e)
    nonlinear={'linear_F_covariance_after':float(np.linalg.norm((erased-erased.mean(0)).T@v/4)),
        'F_squared_recovery_MSE':float(np.mean((erased[:,1]-v*v)**2)),
        'meaning':'F^2 remains; zero linear cross-covariance does not imply universal nonlinear erasure.'}
    random=norm_matched(h,leace,42)
    dose={'max_per_scene_norm_gap':float(np.max(np.abs(np.linalg.norm(random-h,axis=1)-np.linalg.norm(el-h,axis=1))))}
    a=rng.integers(0,2,size=(32,5,12));b=rng.integers(0,2,size=a.shape);c=matched_binary_message(a,b,2)
    bits={'count_mismatches':int(np.sum(c.sum(-1)!=b.sum(-1))),
        'flip_mismatches':int(np.sum((c!=a).sum(-1)!=(b!=a).sum(-1)))}
    # Right output can result from two nonzero mechanisms cancelling.
    cancel=decomposition(np.zeros((8,1)),np.zeros((8,1)),np.ones((8,1)),np.zeros((8,1)))
    # Erasing a correlated F necessarily costs some G predictability.
    floors=[gaussian_erasure_utility_floor(rho) for rho in (0,.5,.9,1.)]
    # Hard reset makes a small pre perturbation create a large post difference.
    pre=np.array([.999,1.001]);events=(pre>=1).astype(float);post=pre*(1-events)
    event={'pre_difference':float(abs(np.diff(pre)[0])),'post_difference':float(abs(np.diff(post)[0])),
        'event_difference':1,'hard_reset_bound':float(abs(np.diff(pre)[0])+max(abs(pre)))}
    # Zero class mean difference: the safe binary MP extension is the identity.
    x=np.array([[-1.,0],[1.,0],[-2.,0],[2.,0]]);binary=np.array([[-1.],[-1.],[1.],[1.]])
    mp=fit_eraser(x,binary,'binary_mp')
    mp_control={'removed_rank':mp['removed_rank'],'identity_max_abs':float(np.max(np.abs(apply_eraser(x,mp)-x)))}
    # L2 convex guardedness and zero-one linear threshold risk differ.
    threshold=(x[:,0]>1.5);truth=binary[:,0]>0
    guarded={'class_centroids_equal':True,'best_constant_binary_error':.5,'one_sided_threshold_error':float(np.mean(threshold!=truth)),
        'scope':'Zero-one loss is not convex in an affine score; do not replace the LEACE convex-loss theorem with this claim.'}
    transport_control={'identity_transport_max_abs':float(np.max(np.abs(transport(h,h,leace)-h)))}
    assert empirical['leace_cross_covariance_norm']<1e-10 and empirical['leace_displacement_MSE']<=empirical['orth_displacement_MSE']+1e-10
    assert nonlinear['F_squared_recovery_MSE']<1e-20 and bits['count_mismatches']==bits['flip_mismatches']==0
    return {'module':'L4.3 known-answer mathematical controls','Gaussian_erasure':empirical,'nonlinear_residual':nonlinear,
        'matched_dose':dose,'binary_message_budget':bits,'operator_scope_cancellation':cancel,'correlated_content_floor':floors,
        'threshold_boundary':event,'zero_mean_MP':mp_control,'convex_vs_zero_one':guarded,'transport_identity':transport_control,
        'scope':'Analytic or finite controls. No new SNN training or empirical architecture conclusion.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=run();write_json(a.output,r);print('Intervention controls passed')
