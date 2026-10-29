import unittest
import numpy as np
import torch
from bridge43 import BenchmarkModel,ModelConfig,NEURONS,parameter_hash,Cell,capture,load_model,RUN_DIRECTORY,fixed_batch,hybrid
from intervention_math43 import (fit_eraser,apply_eraser,random_eraser,norm_matched,transport,
    matched_binary_message,decomposition,gaussian_erasure_utility_floor)
from runtime_interventions43 import run_post_edit,after_step_edit,full_snapshot_donor
from run_intervention_controls import run as known
from regularized_intervention43 import fit_covariance_regularized_eraser

class MathematicalControls(unittest.TestCase):
    def test_gaussian_cross_covariance_is_erased(self):
        r=known()['Gaussian_erasure'];self.assertLess(r['leace_cross_covariance_norm'],1e-10)
    def test_leace_distortion_not_larger_than_orthogonal(self):
        r=known()['Gaussian_erasure'];self.assertLessEqual(r['leace_displacement_MSE'],r['orth_displacement_MSE']+1e-12)
    def test_erasure_preserves_fit_mean(self):
        x=np.random.default_rng(1).normal(size=(100,5))+3;f=x[:,:1]
        np.testing.assert_allclose(apply_eraser(x,fit_eraser(x,f)).mean(0),x.mean(0),atol=1e-12)
    def test_erasure_target_units_do_not_change_relative_rank(self):
        x=np.random.default_rng(2).normal(size=(100,5))
        a=fit_eraser(x,x[:,:1]);b=fit_eraser(x,x[:,:1]*1e-5)
        np.testing.assert_allclose(a['matrix'],b['matrix'],atol=1e-12)
    def test_zero_variance_state_does_not_create_directions(self):
        x=np.zeros((20,4));f=np.random.default_rng(3).normal(size=(20,1));r=fit_eraser(x,f)
        self.assertEqual(r['removed_rank'],0);np.testing.assert_array_equal(apply_eraser(x,r),x)
    def test_rank_deficient_covariance_remains_finite(self):
        x=np.c_[np.arange(20),np.arange(20),np.zeros(20)];r=fit_eraser(x,x[:,:1])
        self.assertTrue(np.isfinite(r['matrix']).all());self.assertEqual(r['removed_rank'],1)
    def test_nonlinear_information_can_remain_after_linear_erasure(self):
        r=known()['nonlinear_residual'];self.assertLess(r['linear_F_covariance_after'],1e-10);self.assertEqual(r['F_squared_recovery_MSE'],0)
    def test_mp_requires_two_actual_classes(self):
        with self.assertRaises(ValueError):fit_eraser(np.ones((5,4)),np.ones((5,1)),'binary_mp')
    def test_identical_binary_class_means_are_safe_identity(self):
        r=known()['zero_mean_MP'];self.assertEqual(r['removed_rank'],0);self.assertEqual(r['identity_max_abs'],0)
    def test_binary_count_and_flip_budgets_match_per_step(self):
        r=known()['binary_message_budget'];self.assertEqual(r['count_mismatches'],0);self.assertEqual(r['flip_mismatches'],0)
    def test_fractional_spikes_rejected(self):
        with self.assertRaises(ValueError):matched_binary_message(np.zeros((2,3,4)),np.ones((2,3,4))*.5,1)
    def test_exact_per_sample_norm_matching(self):self.assertLess(known()['matched_dose']['max_per_scene_norm_gap'],1e-10)
    def test_random_projection_rank_is_matched(self):
        x=np.random.default_rng(1).normal(size=(100,7));r=fit_eraser(x,x[:,:2]);q=random_eraser(r,5)
        self.assertEqual(np.linalg.matrix_rank(np.eye(7)-q['matrix']),r['removed_rank'])
    def test_identity_donor_transport_is_noop(self):self.assertEqual(known()['transport_identity']['identity_transport_max_abs'],0)
    def test_operator_and_scope_errors_can_cancel(self):
        r=known()['operator_scope_cancellation'];self.assertEqual(r['natural_fidelity_MSE'],0);self.assertEqual(r['interaction'],-2.)
    def test_measured_finite_output_risk_bound(self):
        rng=np.random.default_rng(6);a=[rng.normal(size=(100,4)) for _ in range(4)];r=decomposition(*a)
        self.assertLess(r['identity_max_abs'],1e-12);self.assertLessEqual(abs(r['risk_difference']),r['risk_difference_absolute_upper_bound']+1e-12)
    def test_correlated_cotarget_has_nonzero_necessary_cost(self):self.assertAlmostEqual(gaussian_erasure_utility_floor(.9)['minimum_MSE_for_G_after_independent_F_erasure'],.81)
    def test_invalid_correlation_rejected(self):
        with self.assertRaises(ValueError):gaussian_erasure_utility_floor(1.1)
    def test_hard_reset_amplifies_a_small_crossing(self):
        r=known()['threshold_boundary'];self.assertGreater(r['post_difference'],100*r['pre_difference']);self.assertLess(r['post_difference'],r['hard_reset_bound'])
    def test_convex_loss_guarantee_is_not_zero_one_guarantee(self):self.assertEqual(known()['convex_vs_zero_one']['one_sided_threshold_error'],.25)
    def test_nonfinite_operations_rejected(self):
        with self.assertRaises(ValueError):fit_eraser(np.ones((4,2)),np.array([[np.nan]]*4))
    def test_regularized_metric_erases_fit_cross_covariance(self):
        x=np.random.default_rng(44).normal(size=(100,5));x[:,-1]*=1e-9;f=x[:,:2]
        e=fit_covariance_regularized_eraser(x,f);self.assertLess(e['fit_cross_covariance_after_norm'],1e-10)
    def test_regularized_operator_respects_condition_bound(self):
        x=np.random.default_rng(45).normal(size=(100,5));x[:,-1]*=1e-9
        e=fit_covariance_regularized_eraser(x,x[:,:2]);self.assertLessEqual(e['operator_norm'],e['condition_bound']+1e-12)
    def test_regularized_constant_state_is_identity(self):
        x=np.ones((10,4));e=fit_covariance_regularized_eraser(x,np.arange(10)[:,None]);np.testing.assert_array_equal(apply_eraser(x,e),x)
    def test_invalid_regularization_rejected(self):
        with self.assertRaises(ValueError):fit_covariance_regularized_eraser(np.ones((10,4)),np.ones((10,1)),0)

class ActualRuntimeControls(unittest.TestCase):
    def model(self,readout='last',mode='full',observation='post',cell='lif_hard'):
        torch.manual_seed(43043)
        cfg=ModelConfig(neuron=cell,width=4,depth=2,observation=observation,readout=readout,gradient_mode=mode)
        return BenchmarkModel((3,),(2,),4,cfg,objective='regression').eval()
    def x(self):return np.random.default_rng(43).normal(size=(8,4,3)).astype(np.float32)
    def test_noop_preserves_predictions_and_weights(self):
        m=self.model();x=self.x();base=m(torch.tensor(x)).detach().numpy();sha=parameter_hash(m)
        r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:h)
        np.testing.assert_array_equal(r['prediction'],base);self.assertEqual(parameter_hash(m),sha)
    def test_after_final_step_cannot_retroactively_edit_consumed_readout(self):
        m=self.model();x=self.x();base=m(torch.tensor(x)).detach().numpy()
        after=after_step_edit(m,x,'backbone.cells.1',3,lambda h,start:np.zeros_like(h))
        before=run_post_edit(m,x,'backbone.cells.1',3,lambda h,start:np.zeros_like(h))['prediction']
        np.testing.assert_array_equal(after,base);self.assertGreater(np.max(np.abs(before-base)),0)
    def test_current_emission_branch_is_not_retroactively_recomputed(self):
        m=self.model();x=self.x();base=capture(m,x)
        r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:np.zeros_like(h))
        for key in ('pre','natural_event'):
            np.testing.assert_array_equal(r['traces']['backbone.cells.1/'+key][:,1],base['carriers']['backbone.cells.1/'+key][:,1])
    def test_auxiliary_state_at_cut_is_preserved_by_v_only_operation(self):
        for kind,aux in [('alif_ref','a'),('adlif_ref','a'),('two_state_ref','syn')]:
            m=self.model(cell=kind);x=self.x();base=capture(m,x)
            r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:np.zeros_like(h))
            np.testing.assert_array_equal(r['traces']['backbone.cells.1/'+aux][:,1],base['carriers']['backbone.cells.1/'+aux][:,1])
    def test_state_operation_does_not_change_final_output_observation(self):
        m=self.model(observation='output');x=self.x();base=m(torch.tensor(x)).detach().numpy()
        r=run_post_edit(m,x,'backbone.cells.1',3,lambda h,start:np.zeros_like(h))
        np.testing.assert_array_equal(r['prediction'],base)
    def test_no_td_last_output_forgets_earlier_state_operation(self):
        m=self.model(mode='notd');x=self.x();base=m(torch.tensor(x)).detach().numpy()
        r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:np.zeros_like(h));np.testing.assert_array_equal(r['prediction'],base)
    def test_no_td_mean_readout_retains_earlier_observation_operation(self):
        m=self.model(mode='notd',readout='mean');x=self.x();base=m(torch.tensor(x)).detach().numpy()
        r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:np.zeros_like(h));self.assertGreater(np.max(np.abs(r['prediction']-base)),0)
    def test_invalid_site_and_horizon_rejected(self):
        m=self.model();x=self.x()
        with self.assertRaises(ValueError):run_post_edit(m,x,'not_a_cell',1,lambda h,start:h)
        with self.assertRaises(ValueError):run_post_edit(m,x,'backbone.cells.1',4,lambda h,start:h)
    def test_wrong_shape_and_nan_operation_rejected(self):
        m=self.model();x=self.x()
        for op in [lambda h,start:h[:,:1],lambda h,start:np.full_like(h,np.nan)]:
            with self.assertRaises(ValueError):run_post_edit(m,x,'backbone.cells.1',1,op)
        self.assertEqual(len(m.backbone.cells[1]._forward_hooks),0)
    def test_content_hybrid_holds_other_content_noise_and_future(self):
        m,cfg,_=load_model(RUN_DIRECTORY);a=fixed_batch(cfg.data,4,'probe_val',9990000);b=fixed_batch(cfg.data,4,'probe_val',9991000)
        d=hybrid(cfg.data,a,b,'H1');np.testing.assert_array_equal(d['latents'][:,1:],a['latents'][:,1:]);np.testing.assert_array_equal(d['measurement_noise'],a['measurement_noise'])
        np.testing.assert_array_equal(d['x'][:,3:],a['x'][:,3:])
    def test_whole_donor_snapshot_reproduces_donor_future(self):
        m,cfg,_=load_model(RUN_DIRECTORY);a=fixed_batch(cfg.data,4,'probe_val',9992000);b=fixed_batch(cfg.data,4,'probe_val',9993000);d=hybrid(cfg.data,a,b,'C')
        r=full_snapshot_donor(m,a['x'],d['x'],12);self.assertEqual(r['max_abs'],0)
    def test_auxiliary_owner_edit_changes_future_threshold_not_current_v(self):
        m=self.model(cell='alif_ref');x=self.x()*5;base=capture(m,x)
        r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:h+10,state_keys=('a',))
        np.testing.assert_array_equal(r['traces']['backbone.cells.1/post'][:,1],base['carriers']['backbone.cells.1/post'][:,1])
        self.assertGreater(np.max(np.abs(r['traces']['backbone.cells.1/theta'][:,2]-base['carriers']['backbone.cells.1/theta'][:,2])),1.)
    def test_synaptic_owner_edit_changes_future_pre_not_current_v(self):
        m=self.model(cell='two_state_ref');x=self.x();base=capture(m,x)
        r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:h+10,state_keys=('syn',))
        np.testing.assert_array_equal(r['traces']['backbone.cells.1/post'][:,1],base['carriers']['backbone.cells.1/post'][:,1])
        self.assertGreater(np.max(np.abs(r['traces']['backbone.cells.1/pre'][:,2]-base['carriers']['backbone.cells.1/pre'][:,2])),.1)
    def test_joint_recurrent_owner_noop(self):
        for kind,keys in [('alif_ref',('v','a')),('adlif_ref',('v','a')),('two_state_ref',('v','syn'))]:
            m=self.model(cell=kind);x=self.x();base=m(torch.tensor(x)).detach().numpy()
            r=run_post_edit(m,x,'backbone.cells.1',1,lambda h,start:h,state_keys=keys);np.testing.assert_array_equal(r['prediction'],base)
    def test_missing_owner_rejected_and_hook_removed(self):
        m=self.model()
        with self.assertRaises(ValueError):run_post_edit(m,self.x(),'backbone.cells.1',1,lambda h,start:h,state_keys=('a',))
        self.assertEqual(len(m.backbone.cells[1]._forward_hooks),0)

if __name__=='__main__':unittest.main()
