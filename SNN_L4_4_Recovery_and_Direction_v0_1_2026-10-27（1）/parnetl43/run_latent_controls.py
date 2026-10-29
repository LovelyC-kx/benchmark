"""Known-answer controls for latent availability, geometry and task utility."""
import argparse
import numpy as np
from bridge42 import write_json
from latent_theory42 import (gaussian_risk, gaussian_risk_chain, fixed_linear_risk,
    query_readability, task_metric, gaussian_extra_risk_reduction, participation_ratio, affine_gauge, spectral_ranks)

def run():
    rng=np.random.default_rng(42042);eye=np.eye(5)
    # Each step removes a different coordinate needed by the target.
    a=np.diag([1.,1.,1.,1.,0.]);g=eye[:3];p=np.array([[1.,0,0],[0,1.,0]])
    f=np.array([[1.,0]]);w=np.zeros((5,1));target=eye;noise=.1*eye
    chain=gaussian_risk_chain(eye,a,g,p,f,target,w,noise)
    # Same readable signal, arbitrarily large task-irrelevant variance.
    low=np.c_[rng.normal(size=8192),rng.normal(scale=.1,size=8192)]
    high=low*np.array([1.,100.]);L=np.array([[1.,0.]])
    pca={'low_noise_variance_risk':gaussian_risk(np.eye(2),np.array([[1.,0.]]),L),
        'high_nuisance_first_PC_risk':gaussian_risk(np.eye(2),np.array([[0.,10.]]),L),
        'full_code_risk_before':gaussian_risk(np.eye(2),np.eye(2),L),
        'full_code_risk_after':gaussian_risk(np.eye(2),np.diag([1.,10.]),L),
        'participation_ratio_before':participation_ratio(low),'participation_ratio_after':participation_ratio(high)}
    # Full hypercube population: affine recovery of product has zero covariance.
    z=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    y=z[:,0]*z[:,1]
    linear=np.linalg.lstsq(np.c_[z,np.ones(4)],y,rcond=None)[0]
    poly=np.linalg.lstsq(np.c_[z,z.prod(1),np.ones(4)],y,rcond=None)[0]
    xor={'affine_MSE':float(np.mean((np.c_[z,np.ones(4)]@linear-y)**2)),
        'product_reader_MSE':float(np.mean((np.c_[z,z.prod(1),np.ones(4)]@poly-y)**2)),
        'target_deterministic_from_code':True,'marginal_coordinate_correlations':[float(z[:,i]@y/4) for i in range(2)]}
    # An exact Bayes gain with the extra F is not native reliance on F.
    baseline=np.array([[1.,0.]]);extra=np.array([[0.,1.]])
    conditional=gaussian_extra_risk_reduction(np.eye(2),baseline,extra,np.array([[1.,1.]]))
    W=np.array([[1.,0.]]);D=np.diag([100.,.01]);Bp,Wp=affine_gauge(np.eye(2),W,D)
    dx=np.array([.3,.4]);q1=float(dx@task_metric(W)@dx);q2=float((D@dx)@task_metric(Wp)@(D@dx))
    gauge={'prediction_difference':float(np.max(np.abs(W-Wp@Bp))),
        'original_Euclidean_step':float(dx@dx),'transformed_Euclidean_step':float((D@dx)@(D@dx)),
        'original_task_step':q1,'transformed_task_step':q2}
    # One known task needs one coordinate; unknown late query needs both.
    fixed=query_readability(np.array([[1.,0.]]),np.array([[1.,0.]]))
    late=query_readability(np.array([[1.,0.]]),np.eye(2))
    query={'fixed_task':{k:v for k,v in fixed.items() if k!='reader'},
        'two_late_queries':{k:v for k,v in late.items() if k!='reader'},'minimum_linear_code_rank_for_both_queries':2}
    # Accurate all-latent reconstruction can hide failure on one vital latent.
    zcov=np.diag([.01,100.]);B=np.array([[0.,1.]])
    mismatch={'all_latent_MSE':gaussian_risk(zcov,B,np.eye(2)),
        'target_variance':.01,'target_relative_MSE':gaussian_risk(zcov,B,np.array([[1.,0.]]))/.01}
    # Reader ignores a perfectly accessible target; nonlinear causal use remains L4.3.
    stored={'Bayes_task_risk':gaussian_risk(np.eye(2),np.eye(2),np.array([[1.,0.]])),
        'native_task_risk':fixed_linear_risk(np.eye(2),np.eye(2),np.array([[1.,0.]]),np.array([[0.,1.]]))}
    # Exact image loss: spatial mean loses position while the full latent is readable.
    pixels=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    image={'full_spatial_MSE':gaussian_risk(np.eye(2),np.eye(2),pixels),
        'GAP_MSE':gaussian_risk(np.eye(2),np.array([[.5,.5]]),pixels),
        'scope':'Four-pixel exact linear image reference, not a trained image reconstructor.'}
    # A marginally useful historical feature can have zero conditional utility.
    redundant=gaussian_extra_risk_reduction(np.eye(2),np.array([[1.,0.]]),np.array([[2.,0.]]),np.array([[1.,0.]]))
    # Log-loss utility differs from zero-one: a calibrated improvement can leave decisions unchanged.
    logloss={'true_label_probability':.8,'baseline_probability':.6,'better_probability':.8,
        'zero_one_risk_both':.2,'baseline_logloss':float(-.8*np.log(.6)-.2*np.log(.4)),
        'better_logloss':float(-.8*np.log(.8)-.2*np.log(.2))}
    assert abs(sum(chain['pieces'].values())-chain['native_risk'])<1e-10
    assert xor['affine_MSE']>.99 and xor['product_reader_MSE']<1e-20
    assert abs(q1-q2)<1e-10 and image['GAP_MSE']>0
    return {'module':'L4.2 latent theory known controls','risk_chain':chain,'variance_vs_task':pca,
        'nonlinear_task_in_linear_latents':xor,'conditional_utility':conditional,'coordinate_gauge':gauge,
        'query_family':query,'all_latents_vs_task':mismatch,'accessible_but_native_ignored':stored,
        'spatial_reconstruction':image,'redundant_extra_content':redundant,'loss_specificity':logloss,
        'spectral_normalization':spectral_ranks(np.array([[1.,0.],[.6,.8],[-1.,0.],[-.6,-.8]])),
        'scope':'Analytic/exhaustive finite controls; no trained SNN evidence.'}

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=run();write_json(a.output,r);print('Latent theory controls passed')
