"""Observation-level population references derived for our controlled worlds.

These are task identifiability/noise references with unrestricted storage and
known generators, not trained-model or finite-state capacity guarantees.
"""
import numpy as np


def _sign_error(posterior_variance):
    return float(np.arccos(np.sqrt(np.clip(1-posterior_variance,0,1)))/np.pi)


def task_reference(task):
    if task.family=='tagged':
        v=task.sigma**2/(1+task.sigma**2)
        return {'status':'analytic_reference','coordinate_MSE':v,'attribute_sign_error':_sign_error(v),
            'assumptions':'All requested sources present; known source tags, full-rank orthonormal projection, Gaussian unit prior and independent isotropic Gaussian measurement noise; unrestricted history storage.',
            'derivation':'Each requested coordinate reduces to z+epsilon; posterior variance sigma^2/(1+sigma^2). Common signal/noise scaling cancels. Gaussian sign error uses correlation sqrt(1-v).',
            'meaning':'Age and requested-source load do not change this observation-level floor in this generator; any learned degradation requires a model/learning/readout explanation, within finite-sample uncertainty.'}
    if task.family=='spatial' or (task.family=='protocol' and task.name in ('G07','G08')):
        h=task.image_size//2;yy,xx=np.mgrid[:h,:h]
        basis=np.stack([np.ones_like(xx),np.cos(np.pi*(xx+.5)/h),np.cos(np.pi*(yy+.5)/h),(-1.)**(xx+yy)])
        spectral=np.array([[1,.7,.8,.9],[.9,1,.6,.7],[.7,.8,1,.6],[.8,.6,.9,1.]])
        variance=.07**2/3*np.mean(np.einsum('ck,khw->chw',spectral**2,basis**2))
        return {'status':'analytic_reference','unseen_region_expected_MSE':float(variance),'full_image_partial_lower_bound':float(variance/4),
            'assumptions':'Never-visible tile has independent Uniform[-1,1] coefficients; mask/schedule/noise do not reveal them; square risk.',
            'derivation':'Conditional mean is .5. Var(z_k)=1/3, so pixel variance is .07^2/3 times summed squared spectral/basis weights.',
            'scope':'Exact expected risk floor on the unseen tile; only a partial lower bound for full-image risk, not a finite-sample clipping threshold.'}
    if task.family=='protocol' and task.name=='G05':
        count=int(task.params.get('measurements',max(1,task.T//4)));mode=task.params.get('acquisition','independent');effective=1 if mode=='repeat' else count
        v=0. if task.sigma==0 else 1/(1+effective/task.sigma**2)
        return {'status':'analytic_reference','coordinate_MSE':v,'attribute_sign_error':_sign_error(v),'independent_measurements':effective,
                'assumptions':'Gaussian unit signal, independent Gaussian corruptions; repeated values share exactly the same corruption.'}
    if task.family=='protocol' and task.name=='G06':
        T=task.T;times=np.arange(T)/T;f=float(task.params.get('target_frequency',2));d=float(task.params.get('distractor_frequency',5))
        A=np.stack([np.cos(2*np.pi*f*times),np.sin(2*np.pi*f*times),np.cos(2*np.pi*d*times),np.sin(2*np.pi*d*times)],1)
        if task.sigma:cov=np.linalg.inv(np.eye(4)+A.T@A/task.sigma**2)
        else:
            _,s,vt=np.linalg.svd(A,full_matrices=False);rank=int(np.sum(s>s.max()*1e-10));null=vt[rank:].T;cov=null@null.T
        L=np.c_[A[:,:2],np.zeros((T,2))] if task.params.get('waveform_target',True) else np.eye(4)[:2]
        return {'status':'analytic_reference','continuous_target_MSE':float(np.trace(L@cov@L.T)/len(L)),
                'attribute_sign_error':_sign_error(float(cov[0,0])),'design_rank':int(np.linalg.matrix_rank(A)),
                'assumptions':'Known target/distractor frequencies, independent Gaussian quadratures, additive Gaussian observation noise, full sequence access.',
                'derivation':'Posterior quadrature covariance is (I+A^T A/sigma^2)^-1; zero noise uses the identified row space.'}
    if task.family=='protocol' and task.name=='G09':
        T=task.T;mode=task.params.get('acquisition','new');count=T if mode=='new' else int(task.params.get('measurements',max(1,T//4)))
        duration=float(task.params.get('duration',1));tau=float(task.params.get('tau',.3));times=np.rint(np.linspace(1024/count,1024,count)).astype(int)
        variance=1.;last=0
        for at in times:
            rho=np.exp(-(at-last)*duration/1024/tau);variance=rho*rho*variance+1-rho*rho
            variance=variance*task.sigma**2/(variance+task.sigma**2) if task.sigma else 0.;last=at
        return {'status':'analytic_reference','raw_measurement_terminal_MSE':float(variance),'independent_measurements':count,
                'assumptions':'Stationary unit-variance OU fine-grid process and independent per-measurement Gaussian corruption; all raw measurements through the final deadline.',
                'scope':'Kalman observation reference on underlying measurements. Processed input is no more informative; repetitions do not add independent measurements. Interpolation requires its declared delayed availability.'}
    return {'status':'not_available','reason':'No analytic observation-level floor is asserted for this task/version; use explicit empirical observation controls'}
