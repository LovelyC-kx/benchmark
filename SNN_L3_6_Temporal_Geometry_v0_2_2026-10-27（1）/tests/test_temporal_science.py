import unittest
import numpy as np
from temporal_contracts import independent_ids,source_clock,event_age_rows,source_at_rank,prefix_invariant
from temporal_alignment import fit_transport,transport,push_reader,principal_cosines,cycle_error
from temporal_theory import rotation,isotropic_risks,gaussian_fixed_reader_mse,sparse_delay_terms,loss_moment,markov_parameters,gaussian_location_information,continuous_linear_moments,central_metric
from temporal_clocks import rhythmic_step,causal_complex_signal
from temporal_theory import threshold_finite_scale_moment
from temporal_theory import threshold_probability_geometry
from temporal_probes import fit_head,paired_bound

class IdentityContract(unittest.TestCase):
    def test_independent_scenes(self):self.assertTrue(independent_ids([1,2],[3,4]))
    def test_same_scene_at_other_time_is_leakage(self):
        with self.assertRaises(ValueError):independent_ids([1,2],[2,3])
    def test_duplicates_are_not_independent_trials(self):
        with self.assertRaises(ValueError):independent_ids([1,1])
    def test_onset_included_in_post_update(self):
        a=source_clock(np.array([[2]]),[[True]],4)
        np.testing.assert_array_equal(a['available'].ravel(),[False,False,True,True])
    def test_missing_source_not_available(self):
        a=source_clock(np.array([[2]]),[[False]],4)
        self.assertFalse(a['available'].any())
    def test_event_age_keeps_identity(self):
        x=np.arange(12).reshape(2,6,1);a=event_age_rows(x,np.array([1,3]),np.array([0,2]))
        np.testing.assert_array_equal(a[:,:,0],[[1,3],[9,11]])
    def test_age_outside_recording_rejected(self):
        with self.assertRaises(ValueError):event_age_rows(np.zeros((2,4,1)),np.array([2,2]),np.array([3]))
    def test_rank_is_not_source(self):
        ids=source_at_rank(np.array([[1,3,5],[1,5,3]]),np.ones((2,3),bool),1)
        np.testing.assert_array_equal(ids,[1,2])

class MappingContract(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(16).normal(size=(128,3));self.y=np.random.default_rng(17).normal(size=(30,3))
    def test_pure_rotation_restored(self):
        q=np.eye(3);q[:2,:2]=rotation(.8)
        f=fit_transport(self.x@q,self.x)
        np.testing.assert_allclose(transport(f,self.y@q),self.y,atol=1e-12)
    def test_reflection_is_not_proper_rotation(self):
        q=np.diag([1,1,-1]);o=fit_transport(self.x@q,self.x);r=fit_transport(self.x@q,self.x,'rotation')
        self.assertLess(o['determinant'],0);self.assertGreater(r['determinant'],0)
        self.assertGreater(np.mean((transport(r,self.y@q)-self.y)**2),.2)
    def test_orthogonality_does_not_remove_gain(self):
        f=fit_transport(self.x*3,self.x)
        self.assertGreater(np.mean((transport(f,self.y*3)-self.y)**2),1.)
    def test_similarity_explicit_gain(self):
        f=fit_transport(self.x*3,self.x,'similarity')
        np.testing.assert_allclose(transport(f,self.y*3),self.y,atol=1e-12)
    def test_train_mean_shift(self):
        f=fit_transport(self.x+7,self.x,'center')
        np.testing.assert_allclose(transport(f,self.y+7),self.y,atol=1e-12)
    def test_self_map_is_identity_even_affine(self):
        f=fit_transport(self.x,self.x,'affine')
        np.testing.assert_allclose(transport(f,self.y),self.y,atol=1e-12)
    def test_rank_deficiency_is_declared(self):
        z=np.c_[self.x[:,0],self.x[:,0],np.zeros(len(self.x))];f=fit_transport(z,z)
        self.assertFalse(f['orthogonal_identifiable_full_rank']);self.assertEqual(f['rank_cross_covariance'],1)
    def test_reader_composition(self):
        rng=np.random.default_rng(18);w=rng.normal(size=(2,3,5));o=rng.normal(size=(2,3));b=rng.normal(size=(2,5))
        f=fit_transport(self.x+2,self.x);ww,bb=push_reader(f,w,o,b)
        direct=np.einsum('rnd,rdo->rno',transport(f,self.y)[None]-o[:,None],w)+b[:,None]
        fast=np.einsum('nd,rdo->rno',self.y,ww)+bb[:,None]
        np.testing.assert_allclose(fast,direct,atol=1e-12)
    def test_subspace_angle_cannot_detect_label_flip(self):
        np.testing.assert_allclose(principal_cosines(np.eye(3),-np.eye(3)),1)
    def test_rotation_cycle(self):self.assertLess(cycle_error(rotation(.3),rotation(.4),rotation(.7)),1e-14)
    def test_bad_pairing_destroys_known_rotation_fit(self):
        q=np.eye(3);q[:2,:2]=rotation(.8);f=fit_transport((self.x@q)[::-1],self.x)
        self.assertGreater(np.mean((transport(f,self.y@q)-self.y)**2),.1)

class ExactReferenceLaws(unittest.TestCase):
    def test_gain_without_noise_has_no_Bayes_loss(self):self.assertEqual(isotropic_risks(1e-8,0)['Bayes_MSE_per_coordinate'],0)
    def test_noise_gain_ratio_matters(self):self.assertGreater(isotropic_risks(.1,1)['Bayes_MSE_per_coordinate'],isotropic_risks(1,1)['Bayes_MSE_per_coordinate'])
    def test_quarter_turn_binary_chance(self):self.assertAlmostEqual(isotropic_risks(1,0,np.pi/2)['fixed_direction_binary_error'],.5)
    def test_half_turn_binary_all_wrong(self):self.assertEqual(isotropic_risks(1,0,np.pi)['fixed_direction_binary_error'],1.)
    def test_same_reader_excess_zero(self):
        r=gaussian_fixed_reader_mse(1,.2,1,.2,np.eye(2),np.eye(2));self.assertAlmostEqual(r['excess_risk'],0)
    def test_rotated_reader_excess_positive(self):
        r=gaussian_fixed_reader_mse(1,.2,1,.2,np.eye(2),rotation(1));self.assertGreater(r['excess_risk'],.2)
    def test_exact_k_delay_has_zero_loss(self):
        a=np.zeros(6);a[2]=1;self.assertAlmostEqual(sparse_delay_terms(a,2,.2,.5,1/3)['total'],0)
    def test_four_terms_equal_general_moments(self):
        a=np.array([.5,-.2,.3,.1]);p=.3;mu=.5;m2=1/3
        direct=loss_moment(a,np.full(4,p*mu),np.eye(4)*(p*m2-p*p*mu*mu),2)
        self.assertAlmostEqual(direct,sparse_delay_terms(a,2,p,mu,m2)['total'])
    def test_correlated_inputs_need_covariance(self):
        a=np.array([1.,.3]);i=loss_moment(a,[0,0],np.eye(2),1);c=loss_moment(a,[0,0],np.array([[1,.8],[.8,1]]),1)
        self.assertGreater(abs(i-c),.5)
    def test_invisible_unstable_mode(self):
        a=np.diag([.5,2]);b=np.array([[1.],[0]]);c=np.array([[1.,0]])
        np.testing.assert_allclose(markov_parameters(a,b,c,8).ravel(),.5**np.arange(8))
    def test_noise_normalized_information_coordinate_invariance(self):
        j=np.array([[1.,.5],[.2,1.]]);s=np.array([[2.,.3],[.3,1.]]);m=np.array([[2.,.5],[0.,.2]])
        np.testing.assert_allclose(gaussian_location_information(j,s),gaussian_location_information(m@j,m@s@m.T),atol=1e-12)
    def test_nonpositive_noise_rejected(self):
        with self.assertRaises(ValueError):gaussian_location_information(np.eye(2),np.zeros((2,2)))
    def test_ou_exact_moments(self):
        rows=continuous_linear_moments(np.array([[-.7]]),np.ones((1,1)),np.zeros((1,1)),np.array([[.4]]),[0,1,2])
        for row in rows:
            self.assertAlmostEqual(row['noise'][0,0],.4*(1-np.exp(-1.4*row['t']))/1.4)
    def test_local_metric_invertible_warp(self):
        z=np.array([.2,.7]);delta=1e-5
        def f(x):return np.array([x[0],x[1]+x[0]**2])
        plus=[];minus=[]
        for k in range(2):
            d=np.eye(2)[k]*delta;plus.append(f(z+d));minus.append(f(z-d))
        r=central_metric(np.array(plus)[None,:,None],np.array(minus)[None,:,None],delta)
        j=np.array([[1.,0.],[.4,1.]])
        np.testing.assert_allclose(r['euclidean_pullback'][0,0],j.T@j,atol=1e-10)
    def test_zero_tangent_can_coexist_with_perfect_event_decoding(self):
        r=threshold_finite_scale_moment(.01)
        self.assertEqual(r['ordinary_derivative_almost_everywhere'],0)
        self.assertEqual(r['Bayes_binary_decoding_error_from_event'],0)
    def test_threshold_scale_limit_is_singular(self):
        a=threshold_finite_scale_moment(.01);b=threshold_finite_scale_moment(.001)
        self.assertGreater(b['expected_squared_central_difference'],9*a['expected_squared_central_difference'])
        self.assertLess(b['event_boundary_crossing_probability'],a['event_boundary_crossing_probability'])
    def test_small_delta_threshold_asymptote(self):
        r=threshold_finite_scale_moment(1e-5)
        self.assertAlmostEqual(r['expected_squared_central_difference']/r['small_delta_asymptote'],1,places=9)
    def test_zero_target_scale_rejected(self):
        with self.assertRaises(ValueError):gaussian_fixed_reader_mse(1,0,0,0,np.eye(2),np.eye(2))
    def test_probability_geometry_survives_zero_pathwise_derivative(self):
        r=threshold_probability_geometry(0,.5)
        self.assertEqual(r['pathwise_derivative_almost_everywhere'],0)
        self.assertAlmostEqual(r['event_probability'],.5)
        self.assertAlmostEqual(r['Bernoulli_Fisher_information'],2/(np.pi*.25))
    def test_mean_metric_and_mean_path_metric_differ(self):
        j=np.array([[[1.,0.]],[[0.,1.]]]);mean=j.mean(0)
        path=np.einsum('rkd,rld->kl',j,j)/len(j);metric=mean@mean.T
        self.assertGreater(float((path-metric)[0,0]),0)

class CausalClock(unittest.TestCase):
    def test_no_future_phase_leakage(self):
        rng=np.random.default_rng(21);a=rng.normal(size=(2,30));b=a.copy();b[:,15:]+=20
        np.testing.assert_array_equal(causal_complex_signal(a,.1)[:,:15],causal_complex_signal(b,.1)[:,:15])
    def test_off_membrane_auxiliary_can_evolve(self):
        v,a,e=rhythmic_step([.2],[.5],[0],[0],aux_clock='membrane_only')
        self.assertEqual(v[0],.2);self.assertAlmostEqual(a[0],.45);self.assertEqual(e[0],0)
    def test_all_states_frozen_is_another_policy(self):
        v,a,e=rhythmic_step([.2],[.5],[0],[0],aux_clock='all');self.assertEqual(a[0],.5)
    def test_paired_bound_identity_zero_gain(self):
        x=np.array([0.,.5,1.]);r=paired_bound(x,x,4);self.assertEqual(r['gain'],0)

if __name__=='__main__':unittest.main()
