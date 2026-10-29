"""Known-answer mechanisms, not trained-SNN findings."""
import argparse
from pathlib import Path
import numpy as np
from temporal_theory import rotation,isotropic_risks,gaussian_fixed_reader_mse,sparse_delay_terms,loss_moment,markov_parameters,gaussian_location_information,continuous_linear_moments,threshold_finite_scale_moment
from temporal_alignment import fit_transport,transport,principal_cosines,cycle_error
from temporal_probes import fit_head
from temporal_clocks import rhythmic_step,causal_complex_signal
from artifacts35 import write_json

def run():
    rng=np.random.default_rng(360301);cal=rng.normal(size=(512,2));evaluation=rng.normal(size=(131072,2));out={}
    rows=[]
    for angle in (0.,np.pi/6,np.pi/2,np.pi):
        q=rotation(angle);x=evaluation@q
        m=fit_transport(cal@q,cal,'orthogonal');restored=transport(m,x)
        rows.append({'angle':angle,'raw_MSE':float(np.mean((x-evaluation)**2)),
                     'aligned_MSE':float(np.mean((restored-evaluation)**2)),
                     'fixed_binary_error':float(np.mean((x[:,0]*evaluation[:,0])<0)),
                     'theory_error':isotropic_risks(1,0,angle)['fixed_direction_binary_error']})
    out['pure_rotation']=rows
    reflection=np.diag([1.,-1.]);o=fit_transport(cal@reflection,cal,'orthogonal');r=fit_transport(cal@reflection,cal,'rotation')
    out['reflection']={'O2_determinant':o['determinant'],'SO2_determinant':r['determinant'],
                       'O2_MSE':float(np.mean((transport(o,evaluation@reflection)-evaluation)**2)),
                       'SO2_MSE':float(np.mean((transport(r,evaluation@reflection)-evaluation)**2)),
                       'principal_cosines':principal_cosines(np.eye(2),reflection)}
    out['isotropic_noise_gain']=[]
    noise=rng.normal(size=evaluation.shape)
    for a,s in ((1.,0.),(.05,0.),(1.,.5),(.05,.5)):
        x=a*evaluation+s*noise;pred=x*a/(a*a+s*s)
        out['isotropic_noise_gain'].append({'gain':a,'noise':s,'MSE':float(np.mean((pred-evaluation)**2)),
            'binary_error':float(np.mean(x[:,0]*evaluation[:,0]<0)),**isotropic_risks(a,s)})
    def shear(z):return np.c_[z[:,0],10*(z[:,0]+z[:,1])]
    def source(z):return np.c_[z[:,0],10*z[:,1]]
    # These four calibration points have exactly the Gaussian population's
    # zero mean and identity covariance, so the quadratic map objective is exact.
    moment_cal=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    f=fit_transport(shear(moment_cal),source(moment_cal),'orthogonal');raw=shear(evaluation);mapped=transport(f,raw)
    out['dominant_nuisance_alignment']={'raw_target_error':float(np.mean(raw[:,0]*evaluation[:,0]<0)),
        'aligned_target_error':float(np.mean(mapped[:,0]*evaluation[:,0]<0)),
        'raw_state_MSE':float(np.mean((raw-source(evaluation))**2)),
        'aligned_state_MSE':float(np.mean((mapped-source(evaluation))**2)),
        'calibration':'Four-point exact first/second moments; evaluation is independent Gaussian',
        'lesson':'A better whole-state geometric fit can worsen a previously perfect target reader.'}
    def warp(z):return np.c_[z[:,0],z[:,1]+z[:,0]**2]
    x=warp(evaluation);inverse=np.c_[x[:,0],x[:,1]-x[:,0]**2]
    fit=fit_transport(warp(cal),cal,'affine')
    out['invertible_nonlinear_warp']={'known_inverse_MSE':float(np.mean((inverse-evaluation)**2)),
        'affine_alignment_MSE':float(np.mean((transport(fit,x)-evaluation)**2)),
        'lesson':'Affine failure is compatible with information preserved by a nonlinear bijection.'}
    a=np.array([[.6,-.2],[.2,.6]]);b=np.array([[1.],[.2]]);c=np.array([[.7,-.3]])
    kernels=markov_parameters(a,b,c,8).reshape(-1);p=.3;mu=.5;nu=1/3;k=3
    terms=sparse_delay_terms(kernels,k,p,mu,nu)
    generic=loss_moment(kernels,np.full(8,p*mu),np.eye(8)*(p*nu-p*p*mu*mu),k)
    data=(rng.uniform(size=(131072,8))*(rng.random((131072,8))<p));emp=float(np.mean((data@kernels-data[:,k])**2))
    cov=.2*.7**np.abs(np.arange(8)[:,None]-np.arange(8)[None,:]);mean=np.full(8,.5)
    corr=rng.multivariate_normal(mean,cov,size=131072)
    out['serial_recall_loss']={'four_terms':terms,'exact_general_moment':generic,'Monte_Carlo_iid':emp,
        'correlated_exact':loss_moment(kernels,mean,cov,k),
        'correlated_Monte_Carlo':float(np.mean((corr@kernels-corr[:,k])**2)),
        'incorrect_if_independence_assumed':sparse_delay_terms(kernels,k,1,.5,.45)['total'],
        'second_moment_nu_not_variance':nu}
    A=np.diag([.5,2.]);B=np.array([[1.],[0.]]);C=np.array([[1.,0.]])
    out['spectral_visibility']={'full_spectral_radius':2.,'observed_impulse_kernel':markov_parameters(A,B,C,10).reshape(-1),
        'lesson':'Input-output memory depends on reachable and observable modes; a full spectral radius alone is insufficient.'}
    J=np.array([[1.,.2],[.4,1.]]);S=np.array([[2.,.3],[.3,1.]])
    M=np.array([[2.,.5],[0.,.2]])
    information=gaussian_location_information(J,S);changed=gaussian_location_information(M@J,M@S@M.T)
    out['metric_invariance']={'noise_normalized_max_abs':float(np.max(np.abs(information-changed))),
        'Euclidean_metric_change':float(np.linalg.norm(J.T@J-J.T@M.T@M@J))}
    v,a,e=rhythmic_step(np.array([.2]),np.array([.5]),np.array([0.]),np.array([0]),aux_clock='membrane_only')
    va,aa,ea=rhythmic_step(np.array([.2]),np.array([.5]),np.array([0.]),np.array([0]),aux_clock='all')
    out['rhythmic_auxiliary_clock']={'membrane_only_v':v,'membrane_only_a':a,'all_frozen_v':va,'all_frozen_a':aa,
        'lesson':'An OFF membrane does not by itself imply all persistent states are frozen.'}
    signal=rng.normal(size=(4,64));other=signal.copy();other[:,32:]+=100
    out['causal_phase_prefix_max_abs']=float(np.max(np.abs(causal_complex_signal(signal,.1)[:,:32]-causal_complex_signal(other,.1)[:,:32])))
    q01=rotation(.3);q12=rotation(.5);q02=rotation(.8)
    out['known_cycle_consistency']=cycle_error(q01,q12,q02)
    out['threshold_scale_dependent_geometry']=[threshold_finite_scale_moment(d) for d in (.1,.03,.01,.003,.001)]
    out['scope']='Known distributions and independent references. Zero trained SNN runs, no architecture or energy ranking.'
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();dst=Path(args.output)
    if dst.exists():raise FileExistsError(dst)
    dst.parent.mkdir(parents=True,exist_ok=True);result=run();write_json(dst,result)
    print('Completed',len(result)-1,'known-answer control groups')
