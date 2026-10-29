import unittest
import numpy as np
from finite_gaussian import finite_gaussian_forecast
from task_structure import support_tasks,gaussian_label_residual,crossfit_higher_alignment,hermite_task_mean
from task_spectrum import population_linear_capacity,harmonic_capacity_reference,pcr_noise_scan,crossfit_task_spectrum
from paired_readouts import draw_plan,fit_linear_family,evaluate,paired_gain_certificate
from approximation_diagnostics import berry_esseen_diagnostic,prediction_gap
from continuous_risk import exact_continuous_risk,continuous_moments
from higher_order_controls import fit_weighted_affine,contribution_stats


class FiniteGaussian(unittest.TestCase):
    def test_noiseless_positive_weights_at_every_p(self):
        x=np.array([[.2],[-1.3],[2.1]]);y=np.sign(x)
        for p in (1,8,128):
            r=finite_gaussian_forecast(np.eye(1),np.eye(1),np.eye(1),np.eye(1),x,y,p,draws=256)
            self.assertLess(r['predicted_error'],1e-14)

    def test_p1_known_quarter_error(self):
        r=finite_gaussian_forecast(np.array([[2.]]),np.eye(1),np.eye(1),np.eye(1),np.array([[1.],[-1.]]),np.array([[1.],[-1.]]),1)
        self.assertAlmostEqual(r['predicted_error'],.25)

    def test_against_direct_fresh_gaussian_training(self):
        rng=np.random.default_rng(901);p=4;z=rng.normal(size=(30000,p));e=rng.normal(size=z.shape)
        error=float(np.mean(np.sum(np.sign(z)*(z+e),axis=1)<0))
        r=finite_gaussian_forecast(np.array([[2.]]),np.eye(1),np.eye(1),np.eye(1),np.array([[1.]]),np.ones((1,1)),p,draws=8192)
        self.assertLess(abs(error-r['predicted_error']),.012)

    def test_nonzero_origin_against_direct_training(self):
        rng=np.random.default_rng(902);p=4;z=rng.normal(size=(40000,p));e=rng.normal(size=z.shape)
        error=float(np.mean(np.sum(np.sign(z)*(.8+z+.7*e),axis=1)<0))
        r=finite_gaussian_forecast(np.array([[1.49]]),np.eye(1),np.eye(1),np.eye(1),np.array([[1.]]),np.ones((1,1)),p,response_mean=np.array([.8]),draws=8192)
        self.assertLess(abs(error-r['predicted_error']),.012)

    def test_zero_code_tie_half(self):
        r=finite_gaussian_forecast(np.zeros((2,2)),np.zeros((2,1)),np.eye(1),np.eye(1),np.zeros((3,2)),np.ones((3,1)),8,draws=64)
        self.assertEqual(r['predicted_error'],.5)

    def test_global_scale_invariance(self):
        a=finite_gaussian_forecast(np.array([[2.]]),np.eye(1),np.eye(1),np.eye(1),np.array([[1.]]),np.ones((1,1)),8,draws=512)
        scale=1e-12
        b=finite_gaussian_forecast(np.array([[2.]])*scale**2,np.eye(1)*scale,np.eye(1),np.eye(1),np.array([[1.]])*scale,np.ones((1,1)),8,draws=512)
        self.assertAlmostEqual(a['predicted_error'],b['predicted_error'])

    def test_incompatible_covariance_rejected(self):
        with self.assertRaises(ValueError):finite_gaussian_forecast(np.eye(1),np.eye(1)*2,np.eye(1),np.eye(1),np.ones((1,1)),np.ones((1,1)),8)

    def test_fractional_and_boolean_p_rejected(self):
        for p in (1.2,True,0):
            with self.assertRaises(ValueError):finite_gaussian_forecast(np.eye(1),np.eye(1),np.eye(1),np.eye(1),np.ones((1,1)),np.ones((1,1)),p)


class TaskStructure(unittest.TestCase):
    def test_exact_support_and_norm(self):
        t=support_tasks(8,11,3);np.testing.assert_array_equal(np.sum(t!=0,axis=0),np.repeat(3,11));np.testing.assert_allclose(np.sum(t*t,axis=0),1.)

    def test_hermite_task_direction_transition(self):
        t=np.column_stack([np.r_[np.ones(k)/np.sqrt(k),np.zeros(8-k)] for k in (1,4,5,8)])
        r=hermite_task_mean(t)
        self.assertTrue(np.all(r['projection_on_teacher'][:2]<0));self.assertTrue(np.all(r['projection_on_teacher'][2:]>0))
        np.testing.assert_allclose(r['projection_on_teacher'],r['teacher_projection_identity'],atol=1e-14)

    def test_linear_exact_control_has_no_high_order_signal(self):
        rng=np.random.default_rng(903);za,zb=rng.normal(size=(2,100,3));a=rng.normal(size=(5,3));t=support_tasks(3,4,2)
        r=crossfit_higher_alignment(za@a.T,za,zb@a.T,zb,t,control_encoding=a)
        self.assertEqual(r['mean_crossfit_power'],0.)

    def test_anisotropic_label_residual_is_linearly_orthogonal(self):
        rng=np.random.default_rng(904);om=np.array([[2.,.5],[.5,1.]]);z=rng.normal(size=(180000,2))@np.linalg.cholesky(om).T;t=np.array([[.7],[-.4]])
        h=gaussian_label_residual(z,t,om)
        self.assertLess(float(np.max(np.abs(z.T@h/len(z)))),.015)

    def test_crossfit_negative_estimates_not_removed(self):
        rng=np.random.default_rng(905);vals=[]
        for _ in range(32):
            za,zb=rng.normal(size=(2,100,2));xa,xb=rng.normal(size=(2,100,3));r=crossfit_higher_alignment(xa,za,xb,zb,np.eye(2));vals.append(r['mean_crossfit_power'])
        self.assertTrue(any(v<0 for v in vals));self.assertTrue(any(v>0 for v in vals))

    def test_unknown_singular_latent_covariance_rejected(self):
        with self.assertRaises(ValueError):gaussian_label_residual(np.ones((3,2)),np.eye(2),np.diag([1.,0.]))


class SpectrumAndPairing(unittest.TestCase):
    def test_low_energy_can_carry_all_target_capacity(self):
        M=np.diag([.01,1.]);m=np.array([.1,0.]);self.assertAlmostEqual(float(population_linear_capacity(M,m)),1.)
        r=pcr_noise_scan(M,m,[0.,.015],.02)
        self.assertEqual(float(r['rows'][0]['PCR_capacity'][0]),0.)
        self.assertAlmostEqual(float(r['rows'][1]['PCR_capacity'][0]),.4)
        self.assertLess(float(r['rows'][1]['unrestricted_capacity'][0]),float(r['rows'][0]['unrestricted_capacity'][0]))

    def test_invertible_linear_capacity_invariance(self):
        M=np.diag([1.,20.]);m=np.array([.4,.2]);b=np.array([[1.,.4],[.2,2.]])
        self.assertAlmostEqual(float(population_linear_capacity(M,m)),float(population_linear_capacity(b@M@b.T,b@m)))

    def test_harmonic_lower_and_whitening_closes_gap(self):
        M=np.diag([1.,20.]);m=np.array([.4,.2]);r=harmonic_capacity_reference(M,m)
        self.assertGreater(float(r['gap'][0]),0)
        b=np.diag([1.,1/np.sqrt(20)]);s=harmonic_capacity_reference(b@M@b.T,b@m)
        self.assertLess(float(np.max(np.abs(s['gap']))),1e-14)

    def test_scene_certificate_does_not_count_tasks_as_iid(self):
        a=np.linspace(.1,.9,1000);b=a-.05
        r=paired_gain_certificate(a,b,family_size=36)
        self.assertEqual(r['independent_evaluation_scenes'],1000)
        self.assertGreater(r['simultaneous_Hoeffding_interval'][1],.05)
        self.assertLess(r['simultaneous_Hoeffding_interval'][0],.05)

    def test_more_comparisons_widen_bound(self):
        a=np.repeat(.4,1000);b=np.repeat(.3,1000)
        r=paired_gain_certificate(a,b,family_size=1);s=paired_gain_certificate(a,b,family_size=20)
        self.assertGreater(s['radius'],r['radius'])

    def test_nested_same_label_indices(self):
        p=draw_plan(80,(1,8,32),repeats=3);rng=np.random.default_rng(906);x=rng.normal(size=(80,4));y=np.where(x[:,:2]>0,1.,-1.)
        a=fit_linear_family(x,y,p);b=fit_linear_family(x,y,p,'ridge')
        for n in p['sample_sizes']:self.assertEqual(a[n]['draw_plan_sha256'],b[n]['draw_plan_sha256'])
        self.assertEqual(p['indices'].shape,(3,32))

    def test_readers_are_not_mutated_by_evaluation(self):
        rng=np.random.default_rng(907);x=rng.normal(size=(50,3));y=np.where(x[:,:1]>0,1.,-1.);p=draw_plan(50,(8,),2);f=fit_linear_family(x,y,p)[8];w=f['weights'].copy()
        r=evaluate(f,x,y);np.testing.assert_array_equal(w,f['weights']);self.assertEqual(r['per_independent_scene_loss'].shape,(50,))

    def test_berry_diagnostic_decreases_with_p(self):
        rng=np.random.default_rng(908);x=rng.normal(size=(2000,3));y=np.where(rng.normal(size=(2000,2))>0,1.,-1.);r=rng.normal(size=(32,3))
        a=berry_esseen_diagnostic(x,y,r,4);b=berry_esseen_diagnostic(x,y,r,64)
        self.assertLess(b['plug_in_mean_BE_bound'],a['plug_in_mean_BE_bound'])

    def test_replacement_gap_closes_with_signed_pieces(self):
        r=prediction_gap(.2,.3,.25,.4,.42);self.assertLess(abs(r['arithmetic_closure_error']),1e-15)
        self.assertLess(r['components']['training_shape_beyond_task_moments'],0)


class ContinuousRisk(unittest.TestCase):
    def test_identity_finite_p_against_fresh_training(self):
        rng=np.random.default_rng(909);p=8;z=rng.normal(size=(50000,p));w=np.mean(z*z,axis=1)
        r=exact_continuous_risk(np.eye(1),np.eye(1),np.array([[3.]]),1.,1,p)
        self.assertLess(abs(float(np.mean((w-1)**2))-r['expected_MSE_per_target']),.008)

    def test_oracle_gain_against_fresh_training(self):
        rng=np.random.default_rng(910);p=4;z=rng.normal(size=(50000,p));w=np.mean(z*z,axis=1)
        r=exact_continuous_risk(np.eye(1),np.eye(1),np.array([[3.]]),1.,1,p);g=r['oracle_common_scalar_gain']
        self.assertLess(abs(float(np.mean((g*w-1)**2))-r['oracle_scalar_expected_MSE']),.008)

    def test_same_second_moments_different_regression_risk(self):
        M=np.array([[1.01]]);C=np.array([[.1]])
        a=exact_continuous_risk(M,C,np.array([[1.03]]),1.,1,8)
        b=exact_continuous_risk(M,C,np.array([[7+.2*np.sqrt(6)+.03]]),1.,1,8)
        self.assertGreater(b['expected_MSE_per_target'],a['expected_MSE_per_target']+.5)

    def test_vector_target_mixed_moments(self):
        rng=np.random.default_rng(911);z=rng.normal(size=(200000,2));r=continuous_moments(z,z)
        np.testing.assert_allclose(r['C'],np.eye(2),atol=.02);np.testing.assert_allclose(r['T'],4*np.eye(2),atol=.08)

    def test_incompatible_mixed_moments_rejected(self):
        with self.assertRaises(ValueError):exact_continuous_risk(np.eye(1),np.eye(1),np.zeros((1,1)),1.,1,8)
        for power in (np.nan,np.inf,-1.):
            with self.assertRaises(ValueError):exact_continuous_risk(np.eye(1),np.eye(1),3*np.eye(1),power,1,8)

    def test_vector_risk_against_fresh_sample_covariances(self):
        rng=np.random.default_rng(912);d,p=3,8;z=rng.normal(size=(25000,p,d))
        w=np.einsum('rpi,rpj->rij',z,z)/p
        actual=float(np.mean(np.sum((w-np.eye(d))**2,axis=(1,2))/d))
        r=exact_continuous_risk(np.eye(d),np.eye(d),(d+2)*np.eye(d),float(d),d,p)
        self.assertLess(abs(actual-r['expected_MSE_per_target']),.01)


class OptionalTorchPairing(unittest.TestCase):
    def test_mlp_initialization_and_data_draw_are_separate_axes(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch integration')
        from paired_readouts import fit_independent_mlps
        rng=np.random.default_rng(913);x=rng.normal(size=(16,3));y=np.where(x[:,:2]>0,1.,-1.)
        plan=draw_plan(16,(8,),2);fit=fit_independent_mlps(x,y,plan,width=4,epochs=2,seeds=(1,2))[8]
        self.assertEqual({(m['training_draw'],m['initialization_seed']) for m in fit['models']},{(0,1),(0,2),(1,1),(1,2)})
        self.assertEqual(fit['draw_plan_sha256'],fit_linear_family(x,y,plan)[8]['draw_plan_sha256'])


class WeightedControls(unittest.TestCase):
    def test_weighted_latent_covariance_identity(self):
        from task_structure import KAPPA
        # This fourth/sixth-moment identity has high Monte Carlo variance.
        # Integrate the conditional Gaussian second moment instead of using
        # a random assertion with a fixed, uncalibrated absolute tolerance.
        from scipy.integrate import quad
        om=np.array([[2.,.6],[.6,1.]]);theta=np.array([.7,-.3])
        direction=om@theta/np.sqrt(theta@om@theta)
        residual=om-np.outer(direction,direction)
        def weighted_moment(degree):
            return 2*quad(lambda u:u**degree*(1-KAPPA*u)**2*np.exp(-u*u/2)/np.sqrt(2*np.pi),0,np.inf,epsabs=1e-11)[0]
        observed=residual*weighted_moment(0)+np.outer(direction,direction)*weighted_moment(2)
        np.testing.assert_allclose(observed,(1-KAPPA*KAPPA)*om,atol=1e-10)

    def test_known_hermite_variance_optimal_control(self):
        from task_structure import KAPPA,hermite3
        rng=np.random.default_rng(915);z=rng.normal(size=(180000,1));x=.1*z+hermite3(z)
        slope=.1+2*KAPPA*KAPPA/(np.sqrt(6)*(1-KAPPA*KAPPA))
        a=contribution_stats(x,z,np.eye(1),np.array([[[0.,.1]]]))
        b=contribution_stats(x,z,np.eye(1),np.array([[[0.,slope]]]))
        self.assertLess(b['mean_contribution_variance'],a['mean_contribution_variance']*.9)

    def test_weighted_fit_on_known_affine_code(self):
        rng=np.random.default_rng(916);z=rng.normal(size=(200000,2));a=np.array([[.2,-.3],[.4,.1]]);bias=np.array([.5,-.1]);x=z@a.T+bias
        c=fit_weighted_affine(x,z,np.eye(2))
        for j in range(2):np.testing.assert_allclose(c[j],np.c_[bias,a],atol=.03)


if __name__=='__main__':unittest.main()
