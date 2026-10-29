import unittest,math
import numpy as np
from geometry_generalization import sample_geometry,geometry_terms,predict_error,gaussian_surrogate
from task_moments import classification_error,teachers,labels,empirical_task_moments,exact_weight_moments,conditional_forecast,nonlinear_alignment
from readout_design import spectral_candidates,transported_geometry,select_before_evaluation
from followup_diagnostics import count_matched_timing,count_preserving_circular_jitter,temporal_generalization,reduced_rank_fit,selected_communication,gaussian_information_transfer


class Theory(unittest.TestCase):
    def test_isotropic_noiseless(self):
        f=geometry_terms(np.eye(4),np.eye(4),np.eye(4))
        self.assertAlmostEqual(f['PR'],4);self.assertAlmostEqual(f['c'],.25);self.assertAlmostEqual(f['f'],1);self.assertTrue(f['s_is_infinite'])

    def test_nonisotropic_noiseless(self):
        o=np.diag([1.,2.]);f=geometry_terms(o,o,o)
        self.assertGreater(f['inv_f'],1)

    def test_zero_code_half_error(self):
        f=geometry_terms(np.zeros((2,2)),np.zeros((2,1)),np.eye(1))
        self.assertEqual(predict_error(f,1)['predicted_error'],.5)

    def test_zero_phi_reference_is_not_absence_claim(self):
        f=geometry_terms(np.eye(2),np.zeros((2,1)),np.eye(1))
        self.assertEqual(predict_error(f,10)['predicted_error'],.5)

    def test_singular_latents_status(self):
        self.assertIn('singular',geometry_terms(np.eye(2),np.zeros((2,2)),np.diag([0.,1.]))['status'])

    def test_incompatible_joint_rejected(self):
        self.assertEqual(geometry_terms(np.eye(1),np.array([[2.]]),np.eye(1))['status'],'incompatible_joint_covariance')

    def test_rotational_and_global_scale_invariance(self):
        psi=np.diag([1.,2.,3.]);phi=np.array([[.2,.1],[.1,.3],[.1,.2]]);o=np.eye(2)
        f=geometry_terms(psi,phi,o);q=np.linalg.qr(np.random.default_rng(2).normal(size=(3,3)))[0]
        g=geometry_terms(9*q@psi@q.T,3*q@phi,o)
        for k in ('PR','c','inv_f','inv_s'):self.assertAlmostEqual(f[k],g[k])

    def test_tie_is_not_correct(self):
        y=np.array([[-1.,1.]])
        self.assertEqual(classification_error(np.zeros_like(y),y)['mean_error'],.5)
        self.assertEqual(classification_error(np.zeros_like(y),y,'legacy_strict')['mean_error'],0)

    def test_task_prior_retains_all_directions(self):
        t=teachers(3,17);self.assertEqual(labels(np.ones((3,3)),t).shape,(3,17))

    def test_exact_weight_moments(self):
        rng=np.random.default_rng(99);p=7
        x=rng.choice((-1.,1.),size=(20000,p,2));y=x[:,:,0]
        weights=np.mean(x*y[:,:,None],axis=1)
        r=exact_weight_moments(np.eye(2),np.array([[1.],[0.]]),p)
        np.testing.assert_allclose(weights.mean(0),r['mean_weight'][:,0],atol=.015)
        np.testing.assert_allclose(np.cov(weights,rowvar=False),r['weight_covariance_by_task'][0],atol=.01)

    def test_forecast_scale_invariance(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(200,3));y=np.where(x[:,:1]>0,1.,-1.);e=rng.normal(size=(100,3));ey=np.where(e[:,:1]>0,1.,-1.)
        a=conditional_forecast(empirical_task_moments(x,y),e,ey,8)
        b=conditional_forecast(empirical_task_moments(x*1e-12,y),e*1e-12,ey,8)
        self.assertAlmostEqual(a['normal_weight_forecast'],b['normal_weight_forecast'])

    def test_residual_task_alignment_identity(self):
        rng=np.random.default_rng(4);z=rng.normal(size=(500,2));x=np.c_[z,z[:,0]**3];t=teachers(2,5);g=sample_geometry(x,z)
        a=nonlinear_alignment(x,z,labels(z,t),t,g);self.assertLess(a['finite_moment_identity_max_error'],1e-14)

    def test_hermite_task_mean_sign_reversal(self):
        nonlinear=np.sqrt(2/np.pi)*(.1-1/np.sqrt(6));self.assertLess(nonlinear,0)
        self.assertGreater(np.sqrt(2/np.pi)*.1,0)

    def test_invertible_spectral_candidates(self):
        c=spectral_candidates(np.diag([0.,.01,2.]))
        for b in c.values():self.assertEqual(np.linalg.matrix_rank(b),3)

    def test_spectral_transport(self):
        rng=np.random.default_rng(5);z=rng.normal(size=(200,2));x=z@rng.normal(size=(2,4))+.1*rng.normal(size=(200,4));g=sample_geometry(x,z)
        b=spectral_candidates(g['Psi'])['spectral_eta_1.0'];f=transported_geometry(g,b);h=sample_geometry(x@b.T,z);j=geometry_terms(h['Psi'],h['Phi'],h['Omega'])
        for k in ('PR','c','inv_f','inv_s'):self.assertAlmostEqual(f[k],j[k])

    def test_selection_has_no_evaluation_argument(self):
        import inspect
        self.assertFalse(any('eval' in k for k in inspect.signature(select_before_evaluation).parameters))

    def test_count_matching_and_zero_jitter(self):
        x,y=count_matched_timing();self.assertTrue(np.all(x.sum(1)==1))
        np.testing.assert_array_equal(count_preserving_circular_jitter(x,0),x)
        np.testing.assert_array_equal(count_preserving_circular_jitter(x,3).sum(1),x.sum(1))

    def test_orthogonal_time_transport(self):
        rng=np.random.default_rng(6);z=rng.normal(size=(500,2));e=rng.normal(size=(200,2));q=np.array([[0.,-1.],[1.,0.]])
        r=temporal_generalization(np.stack([z,z@q],1),np.stack([e,e@q],1),z,e,alpha=1e-8,align=True)
        self.assertGreater(r['train_time_by_test_time_mse'][0,1],1.)
        self.assertLess(np.max(r['orthogonally_aligned_mse']),1e-12)

    def test_rrr_rank_zero(self):
        x=np.random.default_rng(7).normal(size=(100,4));y=np.ones((100,3));f=reduced_rank_fit(x,y,0)
        self.assertTrue(np.all(f['w']==0));np.testing.assert_array_equal(f['ym'],np.ones(3))

    def test_rrr_rank_constraint(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(100,4));y=rng.normal(size=(100,3));f=reduced_rank_fit(x,y,1)
        self.assertLessEqual(np.linalg.matrix_rank(f['w']),1)

    def test_known_gaussian_bottleneck_drops_target(self):
        r=gaussian_information_transfer(np.array([1.,0.]),np.eye(2),np.array([[0.,1.]]),np.eye(1)*.1)
        self.assertEqual(r['source_linear_Fisher'],1);self.assertEqual(r['receiver_linear_Fisher'],0)

    def test_gaussian_information_data_processing(self):
        r=gaussian_information_transfer(np.array([1.,.2]),np.array([[1.,.7],[.7,2.]]),np.eye(2),np.eye(2)*.1)
        self.assertTrue(r['receiver_no_more_than_source'])

if __name__=='__main__':unittest.main()
