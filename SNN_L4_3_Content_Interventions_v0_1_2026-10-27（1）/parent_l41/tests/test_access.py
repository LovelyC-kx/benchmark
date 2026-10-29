import unittest
import numpy as np
import torch
from access_operators import aggregate,temporal_weights,spatial_access,permission_spec,refinement_relation,output_subspaces,observability_stack
from native_head_audit import audit_model,parameter_hash
from run_access_controls import run
from bridge35 import BenchmarkModel,ModelConfig,StreamSession,load_model,fixed_batch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class MathematicalAccess(unittest.TestCase):
    def test_last_and_mean_are_not_ordered(self):
        # Same last, different means; same mean, different lasts.
        a=np.array([[[0.],[1.]],[[2.],[1.]]])
        self.assertEqual(aggregate(a,'last')[0,0],aggregate(a,'last')[1,0])
        self.assertNotEqual(aggregate(a,'mean')[0,0],aggregate(a,'mean')[1,0])
        b=np.array([[[0.],[2.]],[[2.],[0.]]])
        self.assertEqual(aggregate(b,'mean')[0,0],aggregate(b,'mean')[1,0])
        self.assertNotEqual(aggregate(b,'last')[0,0],aggregate(b,'last')[1,0])
    def test_fixed_horizon_mean_sum_equivalence(self):
        a=np.random.default_rng(4).normal(size=(3,7,4))
        np.testing.assert_allclose(aggregate(a,'sum'),7*aggregate(a,'mean'),atol=1e-14)
    def test_variable_horizon_sum_does_not_identify_mean(self):
        a=np.ones((1,1,1));b=np.full((1,2,1),.5)
        np.testing.assert_array_equal(aggregate(a,'sum'),aggregate(b,'sum'))
        self.assertNotEqual(aggregate(a,'mean')[0,0],aggregate(b,'mean')[0,0])
    def test_trajectory_can_reconstruct_all_declared_summaries(self):
        a=np.random.default_rng(5).normal(size=(3,5,4))
        recovered=aggregate(a,'trajectory').reshape(a.shape)
        for mode in ('last','mean','sum','exponential'):np.testing.assert_array_equal(aggregate(recovered,mode),aggregate(a,mode))
    def test_refinement_does_not_invent_pre_post_inclusion(self):
        a=permission_spec('post','trajectory',3,4,2);b=permission_spec('output','last',3,4,2)
        self.assertEqual(refinement_relation(a,b),'not established')
    def test_exponential_endpoint_decays(self):
        np.testing.assert_array_equal(temporal_weights(3,'exponential',0),[0,0,1])
        np.testing.assert_allclose(temporal_weights(3,'exponential',1),np.ones(3)/3)
    def test_trajectory_is_not_assigned_scalar_time_weights(self):
        with self.assertRaises(ValueError):temporal_weights(3,'trajectory')
    def test_invalid_horizon_mode_decay_and_nonfinite_rejected(self):
        for n,mode,decay in [(0,'last',.8),(3,'unknown',.8),(3,'exponential',-1)]:
            with self.assertRaises(ValueError):temporal_weights(n,mode,decay)
        with self.assertRaises(ValueError):aggregate(np.full((1,2,3),np.nan),'mean')
    def test_weighted_linear_logits_commute_with_normalized_feature_summary(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(2,5,4));w=rng.normal(size=(4,3));b=rng.normal(size=3)
        for mode in ('mean','exponential','last'):
            np.testing.assert_allclose(aggregate(x@w+b,mode),aggregate(x,mode)@w+b,atol=1e-14)
    def test_sum_logits_requires_bias_correction(self):
        x=np.ones((2,3,4));w=np.ones((4,2));b=np.array([2.,3.])
        np.testing.assert_allclose(aggregate(x@w+b,'sum'),aggregate(x,'sum')@w+3*b)
    def test_spatial_mean_loses_position_required_by_reconstruction(self):
        x=np.array([[[[[1.,0.]]]],[[[[0.,1.]]]]])
        g=spatial_access(x,'classification');np.testing.assert_array_equal(g[0],g[1])
        r=spatial_access(x,'reconstruction');self.assertFalse(np.array_equal(r[0],r[1]))
    def test_query_specific_head_null_depends_on_declared_outputs(self):
        w=np.eye(3);all_outputs=output_subspaces(w);first=output_subspaces(w[:1])
        self.assertEqual(all_outputs['null_basis'].shape[1],0);self.assertEqual(first['null_basis'].shape[1],2)
    def test_null_perturbation_is_exact_for_affine_head(self):
        rng=np.random.default_rng(8);w=rng.normal(size=(3,8));n=output_subspaces(w)['null_basis']
        np.testing.assert_allclose(w@n,0,atol=1e-14)
    def test_head_null_is_not_future_system_null(self):
        a=np.array([[0.,1.],[0.,0.]]);c=np.array([[1.,0.]]);delta=np.array([0.,1.])
        self.assertEqual(float((c@delta)[0]),0);self.assertEqual(float((c@a@delta)[0]),1)
        self.assertEqual(np.linalg.matrix_rank(observability_stack(a,c,2)),2)
    def test_trajectory_head_parameters_are_charged(self):
        a=permission_spec('output','last',16,32,8);b=permission_spec('output','trajectory',16,32,8)
        self.assertEqual(a['head_parameters'],264);self.assertEqual(b['head_parameters'],4104)

class RuntimeAccess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.controls=run()
    def test_known_cancellation_and_ridge_scaling(self):
        self.assertEqual(self.controls['temporal_cancellation']['mean_max_abs'],0)
        self.assertLess(self.controls['sum_mean_ridge_gauge']['prediction_max_abs'],1e-12)
    def test_notd_final_cell_has_no_history_but_mean_output_does(self):
        c=self.controls['notd_with_readout_history'];self.assertTrue(c['final_cell_state_equal'])
        self.assertEqual(list(c['native_mean_outputs']),[.5,0.]);np.testing.assert_array_equal(c['clear_core'],c['baseline_stream'])
        np.testing.assert_array_equal(c['clear_readout'],[0.,0.])
    def test_probability_and_logit_averages_can_change_class(self):
        c=self.controls['logits_probabilities_and_losses'];self.assertNotEqual(c['prediction_from_mean_logits'],c['prediction_from_mean_probabilities'])
        self.assertGreater(c['mean_per_step_CE_target0'],c['CE_mean_logits_target0'])
    def test_shared_activity_and_new_target_information_coexist(self):
        c=self.controls['redundancy_can_coexist_with_new_target_information']
        self.assertEqual(c['I_previous_later_bits'],1);self.assertEqual(c['I_target_later_given_previous_bits'],1)
    def test_symmetric_readout_and_frame_local_encoder_cannot_read_order(self):
        c=self.controls['frame_local_symmetric_readout_order_control']
        self.assertTrue(c['mean_same_after_frame_permutation']);self.assertTrue(c['exponential_differs_after_permutation'])
    def test_adaptive_weights_can_bring_auxiliary_information(self):
        c=self.controls['adaptive_weighting_has_its_own_observation_permission']
        self.assertTrue(c['final_feature_history_identical']);self.assertTrue(c['target_recovered'])
    def test_native_five_summaries_factorize_and_stream(self):
        for mode in ('last','mean','sum','exponential','trajectory'):
            torch.manual_seed(9);model=BenchmarkModel((4,),(3,),4,ModelConfig(width=4,readout=mode))
            r=audit_model(model,np.random.default_rng(9).normal(size=(2,4,4)).astype(np.float32))
            self.assertLess(r['manual_GPR_prediction_max_abs'],1e-6);self.assertLess(r['stream_vs_batch_prediction_max_abs'],1e-6)
            paths=[x['path'] for x in r['stream_registry']['rows']]
            self.assertEqual(any('features.' in p for p in paths),mode=='trajectory')
    def test_native_image_reconstruction_keeps_spatial_axes(self):
        model=BenchmarkModel((1,8,8),(1,8,8),3,ModelConfig(backbone='unet',width=2,observation='pre'),objective='reconstruction')
        r=audit_model(model,np.ones((2,3,1,8,8),np.float32))
        self.assertEqual(r['deployed_feature_shape'],[2,3,2,8,8]);self.assertFalse(r['spatial_coordinates_pooled_for_classification'])
    def test_fixed_T_trajectory_rejects_new_length(self):
        model=BenchmarkModel((4,),(3,),4,ModelConfig(width=2,readout='trajectory'))
        with self.assertRaisesRegex(ValueError,'fixed-T'):model(torch.ones(2,3,4))
    def test_image_trajectory_decoder_is_explicitly_unsupported(self):
        with self.assertRaises(ValueError):BenchmarkModel((1,8,8),(1,8,8),3,ModelConfig(backbone='unet',readout='trajectory'),objective='reconstruction')
    def test_analog_output_contract_uses_pre(self):
        model=BenchmarkModel((4,),(3,),3,ModelConfig(neuron='analog_hard_ref',width=2))
        model(torch.ones(2,3,4),capture=True);trace=model.traces()['backbone.cells.1']
        torch.testing.assert_close(trace['output'],trace['pre'])
    def test_capture_and_stream_do_not_change_parameters(self):
        model=BenchmarkModel((4,),(3,),3,ModelConfig(width=2));before=parameter_hash(model)
        audit_model(model,np.ones((2,3,4),np.float32));self.assertEqual(parameter_hash(model),before)
    def test_snapshot_must_include_readout_accumulator(self):
        model=BenchmarkModel((4,),(3,),3,ModelConfig(width=2,readout='mean'));s=StreamSession(model)
        s.step(torch.ones(2,4));snap=s.snapshot();self.assertIn('accumulator',snap['readout'])
        expected=s.run(torch.ones(2,2,4));s.restore(snap);actual=s.run(torch.ones(2,2,4));torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    def test_folded_trajectory_preserves_native_head_without_history_buffer(self):
        from folded_trajectory import FoldedTrajectorySession
        torch.manual_seed(411);model=BenchmarkModel((4,),(3,),5,ModelConfig(width=4,readout='trajectory'))
        x=torch.randn(2,5,4)
        with torch.no_grad():expected=model(x)
        s=FoldedTrajectorySession(model);actual=s.run(x);torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        self.assertEqual(s.readout_contract()['retained_trajectory_length'],0);self.assertEqual(s.readout_contract()['output_accumulator_shape'],[2,3])
    def test_folded_head_rejects_incomplete_horizon(self):
        from folded_trajectory import FoldedTrajectorySession
        model=BenchmarkModel((4,),(3,),5,ModelConfig(width=2,readout='trajectory'))
        with self.assertRaises(ValueError):FoldedTrajectorySession(model).run(torch.ones(2,4,4))
    def test_folded_head_requires_linear_trajectory_contract(self):
        from folded_trajectory import FoldedTrajectorySession
        model=BenchmarkModel((4,),(3,),5,ModelConfig(width=2,readout='mean'))
        with self.assertRaises(ValueError):FoldedTrajectorySession(model)

if __name__=='__main__':unittest.main()
