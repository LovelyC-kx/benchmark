"""Known-answer finite Gaussian, task prior, spectrum and uncertainty controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
from finite_gaussian import finite_gaussian_forecast
from task_moments import labels,teachers
from task_structure import hermite3,hermite_task_mean,crossfit_higher_alignment
from task_spectrum import population_linear_capacity,harmonic_capacity_reference,pcr_noise_scan
from paired_readouts import draw_plan,fit_linear_family,evaluate,paired_gain_certificate
from continuous_risk import exact_continuous_risk
from run_generalization import serial


def gaussian_controls():
    rng=np.random.default_rng(36001);om=np.diag([1.,2.]);encoder=np.array([[1.,.1],[.2,.7],[.3,-.2]])
    noise=.4;psi=encoder@om@encoder.T+noise**2*np.eye(3);phi=encoder@om;t=teachers(2,8,36002)
    rows=[]
    for offset in (np.zeros(3),np.array([.5,-.3,.8])):
        z=rng.normal(size=(128,2))@np.linalg.cholesky(om).T;x=z@encoder.T+noise*rng.normal(size=(128,3))+offset;y=labels(z,t)
        for p in (1,8,32,128):
            # Direct full-dimensional independent Gaussian training, compared
            # with exact conditional-law integration on the SAME test pool.
            repetitions=2048;ztrain=rng.normal(size=(repetitions,p,2))@np.linalg.cholesky(om).T
            rtrain=ztrain@encoder.T+noise*rng.normal(size=(repetitions,p,3))+offset
            yt=np.where(ztrain@t>=0,1.,-1.);w=np.einsum('rpn,rpt->rnt',rtrain,yt)/p
            score=np.einsum('en,rnt->ret',x,w);loss=(score*y[None]<0)+.5*(score==0)
            observed=float(loss.mean());sd=float(loss.mean((1,2)).std(ddof=1)/np.sqrt(repetitions))
            forecast=finite_gaussian_forecast(psi,phi,om,t,x,y,p,response_mean=offset,draws=4096)
            rows.append({'response_mean':offset,'p':p,'direct_training_repetitions':repetitions,'direct_risk':observed,
                'direct_training_MC_standard_error':sd,'conditional_exact_law':forecast,
                'absolute_difference':abs(observed-forecast['predicted_error'])})
    return rows


def hermite_controls():
    d=16;supports=(1,2,3,4,5,8,16)
    theta=np.column_stack([np.r_[np.ones(k)/np.sqrt(k),np.zeros(d-k)] for k in supports])
    exact=hermite_task_mean(theta);rng=np.random.default_rng(36101);n=200000
    z=rng.normal(size=(n,d));x=.1*z+hermite3(z);y=labels(z,theta);mean=x.T@y/n
    split=n//2;cross=crossfit_higher_alignment(x[:split],z[:split],x[split:],z[split:],theta,control_encoding=.1*np.eye(d))
    return {'dimension':d,'linear_coefficient':.1,'same_Psi_for_every_task':1.01,'same_Phi_for_every_task':.1,
        'critical_equal_support':float(1/(.1*np.sqrt(6))),
        'rows':[{'support':k,'exact_teacher_projection':float(exact['projection_on_teacher'][i]),
            'empirical_teacher_projection':float(theta[:,i]@mean[:,i]),
            'exact_projection_MC_standard_error':float(np.sqrt((1.01-exact['projection_on_teacher'][i]**2)/n)),
            'exact_higher_alignment_power':float(exact['higher_alignment_squared_norm'][i]),
            'crossfit_higher_alignment_power':float(cross['crossfit_power_per_task'][i])} for i,k in enumerate(supports)],
        'scope':'Known Gaussian latents and explicit nonlinear code; no SNN findings. The k=4 projection is near zero and empirical sign is not treated as proof; the sign transition follows the exact identity.'}


def spectral_controls():
    M=np.diag([.01,1.]);m=np.array([.1,0.]);noise=pcr_noise_scan(M,m,[0.,.005,.009,.011,.015,.03,.1],.02)
    rng=np.random.default_rng(36201);train=rng.normal(size=(1024,2));test=rng.normal(size=(8192,2))
    # The nuisance is the high-energy mode, target is the low-energy mode.
    tx=train*np.array([.1,1.]);ex=test*np.array([.1,1.]);ty=np.sign(train[:,:1]);ey=np.sign(test[:,:1])
    plan=draw_plan(1024,(1,8,32,128),repeats=16);whitening=np.diag([10.,1.]);raw=fit_linear_family(tx,ty,plan);white=fit_linear_family(tx@whitening.T,ty,plan)
    curves=[]
    for p in plan['sample_sizes']:
        a=evaluate(raw[p],ex,ey);b=evaluate(white[p],ex@whitening.T,ey)
        curves.append({'p':p,'raw_Hebbian':a['mean_error'],'whitened_Hebbian':b['mean_error'],
            'conditional_scene_gain_interval':paired_gain_certificate(a['per_independent_scene_loss'],b['per_independent_scene_loss'],family_size=4)})
    return {'target_energy_fraction':.01/1.01,'target_capacity':float(population_linear_capacity(M,m)),
        'harmonic_reference':harmonic_capacity_reference(M,m),'PCR_noise_control':noise,
        'capacity_after_invertible_whitening':float(population_linear_capacity(whitening@M@whitening.T,whitening@m)),
        'paired_finite_p_curves':curves,
        'scope':'Known code and fixed observation-noise/PCR protocol; distinct from adding input noise to a nonlinear trained SNN.'}


def confidence_control():
    # Actual coverage check for a known probability law, not a proof of the
    # theorem and not extra model seeds.
    rng=np.random.default_rng(36301);covered=0;repetitions=500;n=256;true_gain=.2
    for _ in range(repetitions):
        u=rng.random(n);a=(u<.4).astype(float);b=(u<.2).astype(float)
        r=paired_gain_certificate(a,b);lo,hi=r['simultaneous_Hoeffding_interval'];covered+=lo<=true_gain<=hi
    return {'law':'paired Bernoulli scene losses with exact gain 0.2','trials':repetitions,'scenes_per_trial':n,
        'empirical_coverage':covered/repetitions,'declared_confidence':.95,
        'scope':'Coverage calibration; valid bound comes from the bounded-loss Hoeffding proof, not this simulation.'}


def continuous_controls():
    rng=np.random.default_rng(36401);sizes=(1,4,16,64,256);rows=[];repetitions=30000
    for p in sizes:
        z=rng.normal(size=(repetitions,p));weight=np.mean(z*z,axis=1)
        reference=exact_continuous_risk(np.eye(1),np.eye(1),np.array([[3.]]),1.,1,p)
        gain=reference['oracle_common_scalar_gain']
        rows.append({'p':p,'exact_identity_risk':reference['expected_MSE_per_target'],
            'empirical_training_integrated_test_MSE':float(np.mean((weight-1)**2)),
            'exact_oracle_gain_risk':reference['oracle_scalar_expected_MSE'],
            'empirical_oracle_gain_risk':float(np.mean((gain*weight-1)**2))})
    a=.1;M=np.array([[1.01]]);C=np.array([[a]])
    t_gaussian=1+3*a*a;t_hermite=7+2*a*np.sqrt(6)+3*a*a
    matched=[]
    for p in sizes:
        matched.append({'p':p,'Gaussian':exact_continuous_risk(M,C,np.array([[t_gaussian]]),1.,1,p),
            'Hermite':exact_continuous_risk(M,C,np.array([[t_hermite]]),1.,1,p)})
    return {'identity_code_classification_error_for_all_p':0.,'identity_regression':rows,
        'same_second_moments_different_mixed_fourth_moment':{'Gaussian_T':t_gaussian,'Hermite_T':t_hermite,'curves':matched},
        'scope':'Known-code continuous target reference, not a fitted nonlinear reconstruction model or a PSNR prediction.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter();result={'kind':'known-answer controls, no SNN training or data download',
        'finite_Gaussian':gaussian_controls(),'task_support_Hermite':hermite_controls(),
        'task_energy_spectrum':spectral_controls(),'conditional_gain_coverage':confidence_control(),
        'continuous_target_exact_risk':continuous_controls(),
        'elapsed_seconds':time.perf_counter()-start}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print(json.dumps({'output':str(out),'Gaussian_max_absolute_difference':max(r['absolute_difference'] for r in result['finite_Gaussian']),
        'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
