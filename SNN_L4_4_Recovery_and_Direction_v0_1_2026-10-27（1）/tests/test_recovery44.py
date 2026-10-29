import unittest
import numpy as np
import torch
from bridge44 import BenchmarkModel,ModelConfig,parameter_hash,StreamSession
from recovery_math44 import four_losses,output_decomposition,norm_match,random_dose,bounded_interval,mobius_outputs
from recovery_runtime44 import scheduled_run,trace_site,Edit,additive_edit
from snapshot_recovery44 import zero_snapshot,restore_scope
from run_recovery_controls import run as known
from run_binary_recovery import assign_operator

class MathematicalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.controls=known()['results']
    def test_zero_damage_has_no_ratio(self):self.assertIsNone(self.controls['zero_damage_ratio']['recovery_ratio'])
    def test_beneficial_damage_has_no_ratio(self):
        a=np.ones(4);z=np.zeros(4);self.assertIsNone(four_losses(a,z,z,z)['recovery_ratio'])
    def test_ratio_can_be_disabled_for_other_loss_units(self):
        a=np.zeros(4);d=np.ones(4);self.assertIsNone(four_losses(a,d,a,a,damage_floor=None)['recovery_ratio'])
    def test_recovery_ratio_is_not_clipped(self):self.assertGreater(self.controls['uncropped_recovery_ratio']['recovery_ratio'],1)
    def test_affine_risk_interaction_does_not_imply_output_interaction(self):
        r=self.controls['affine_risk_interaction_without_output_interaction'];self.assertEqual(r['output_interaction_MSE'],0);self.assertEqual(r['risk_interaction'],2)
    def test_general_squared_risk_identity(self):
        rng=np.random.default_rng(3);r=output_decomposition(*(rng.normal(size=(100,5)) for _ in range(5)));self.assertLess(r['risk_identity_absolute_residual'],1e-12)
    def test_mismatched_output_coordinates_rejected(self):
        with self.assertRaises(ValueError):output_decomposition(np.ones((2,2)),*[np.ones((2,3))]*4)
    def test_and_requires_joint_restoration(self):self.assertEqual(self.controls['AND_owner_interaction']['outputs'],[0,0,0,1])
    def test_or_can_hide_single_owner_necessity(self):self.assertEqual(self.controls['OR_owner_interaction']['outputs'],[0,1,1,1])
    def test_xor_restoration_is_nonmonotonic(self):self.assertEqual(self.controls['XOR_owner_interaction']['outputs'],[0,1,1,0])
    def test_factorial_reconstructs_every_subset(self):
        r=mobius_outputs({k:np.array([[k*k]]) for k in range(4)});self.assertEqual(r['reconstruction_max_abs'],0)
    def test_factorial_requires_joint_condition(self):
        with self.assertRaises(ValueError):mobius_outputs({0:np.zeros((1,1))})
    def test_norm_matching_is_per_scene(self):
        v=np.arange(15).reshape(5,3)+1;r=np.ones((5,3))*np.arange(1,6)[:,None];matched,b=norm_match(v,r)
        np.testing.assert_allclose(np.linalg.norm(matched,axis=1),np.linalg.norm(r,axis=1),atol=1e-12)
    def test_zero_direction_is_reported_not_replaced(self):self.assertEqual(self.controls['zero_direction_matching_failure']['unmatchable_nonzero_reference_scenes'],3)
    def test_random_dose_preserves_declared_norm(self):
        r=np.arange(15).reshape(5,3);p=random_dose(r,3);np.testing.assert_allclose(np.linalg.norm(r,axis=1),np.linalg.norm(p,axis=1),atol=1e-12)
    def test_unbounded_contrast_rejected(self):
        with self.assertRaises(ValueError):bounded_interval(np.array([1.1]),5)
    def test_empty_scene_units_rejected(self):
        with self.assertRaises(ValueError):bounded_interval(np.array([]),5)
    def test_nonfinite_recovery_losses_rejected(self):
        with self.assertRaises(ValueError):four_losses(np.array([np.nan]),*[np.zeros(1)]*3)
    def test_event_boundary_interaction_disappears_when_fixed(self):
        r=self.controls['finite_event_boundary'];self.assertGreater(r['natural_output_interaction'],1);self.assertLess(abs(r['all_events_fixed_off_output_interaction']),1e-12)
    def test_external_donor_adds_information(self):self.assertEqual(self.controls['external_information_permission']['MSE_from_surviving_G'],1.)
    def test_competent_lif_restores_target_and_preserves_cotarget(self):
        r=self.controls['competent_LIF_known_answer'];self.assertLess(r['repaired_F_MSE'],1e-12);self.assertGreater(r['damaged_F_MSE'],.01);self.assertEqual(r['cotarget_prediction_change_max_abs'],0)
    def test_binary_assignment_is_the_same_declared_operator(self):
        clean=np.array([[[1,0,1]]]);damaged=np.array([[[0,0,1]]]);repaired=np.array([[[1,1,1]]]);po,mask=assign_operator(clean,damaged,repaired)
        np.testing.assert_array_equal(po,np.array([[[1,1,1]]]));self.assertTrue(np.isin(po,[0,1]).all())

class RuntimeContracts(unittest.TestCase):
    def model(self,cell='lif_hard',readout='last',backbone='mlp'):
        torch.manual_seed(4444);return BenchmarkModel((3,),(2,),4,ModelConfig(backbone=backbone,neuron=cell,width=4,depth=2,observation='post',readout=readout),objective='regression').eval()
    def x(self):return np.random.default_rng(44).normal(size=(8,4,3)).astype(np.float32)
    def test_noop_predictions_and_parameters(self):
        m=self.model();x=self.x();a=scheduled_run(m,x);sha=parameter_hash(m);r=scheduled_run(m,x,[Edit('backbone.cells.1',1,lambda h,start:h,'noop')]);np.testing.assert_array_equal(r['prediction'],a['prediction']);self.assertEqual(parameter_hash(m),sha)
    def test_immediate_inverse_is_an_execution_ceiling(self):
        m=self.model();x=self.x();a=scheduled_run(m,x);h=trace_site(a,1,1);de=Edit('backbone.cells.1',1,lambda h,start:np.zeros_like(h),'damage');d=scheduled_run(m,x,[de]);delta=h-trace_site(d,1,1)
        r=scheduled_run(m,x,[de,additive_edit(1,1,delta)]);np.testing.assert_array_equal(r['prediction'],a['prediction'])
    def test_edit_order_changes_the_result(self):
        m=self.model();x=self.x();zero=Edit('backbone.cells.1',3,lambda h,start:np.zeros_like(h),'zero');one=Edit('backbone.cells.1',3,lambda h,start:np.ones_like(h),'one')
        a=scheduled_run(m,x,[zero,one]);b=scheduled_run(m,x,[one,zero]);self.assertGreater(np.max(np.abs(a['prediction']-b['prediction'])),0)
    def test_event_replay_preserves_clean_baseline(self):
        m=self.model();x=self.x();a=scheduled_run(m,x);events={p:a['traces'][p+'/emit_event'] for p in ('backbone.cells.0','backbone.cells.1')};r=scheduled_run(m,x,fixed_events=events);np.testing.assert_array_equal(r['prediction'],a['prediction'])
    def test_nonbinary_event_policy_rejected(self):
        with self.assertRaises(ValueError):scheduled_run(self.model(),self.x(),fixed_events={p:np.ones((8,4,4))*.5 for p in ('backbone.cells.0','backbone.cells.1')})
    def test_event_controls_do_not_leak(self):
        m=self.model();x=self.x();a=scheduled_run(m,x);events={p:np.zeros((8,4,4)) for p in ('backbone.cells.0','backbone.cells.1')};scheduled_run(m,x,fixed_events=events);r=scheduled_run(m,x);np.testing.assert_array_equal(r['prediction'],a['prediction'])
    def test_bad_shape_removes_all_hooks(self):
        m=self.model()
        with self.assertRaises(ValueError):scheduled_run(m,self.x(),[Edit('backbone.cells.1',1,lambda h,start:h[:,:1],'bad')])
        for c in m.backbone.cells:self.assertFalse(c._forward_hooks);self.assertFalse(c._forward_pre_hooks)
    def test_nan_state_edit_rejected(self):
        with self.assertRaises(ValueError):scheduled_run(self.model(),self.x(),[Edit('backbone.cells.1',1,lambda h,start:np.full_like(h,np.nan),'bad')])
    def test_duplicate_labels_rejected(self):
        e=Edit('backbone.cells.1',1,lambda h,start:h,'same')
        with self.assertRaises(ValueError):scheduled_run(self.model(),self.x(),[e,e])
    def test_unknown_path_and_horizon_rejected(self):
        for path,t in [('unknown',1),('backbone.cells.1',4)]:
            with self.assertRaises(ValueError):scheduled_run(self.model(),self.x(),[Edit(path,t,lambda h,start:h,'bad')])
    def test_joint_auxiliary_owner_noop(self):
        m=self.model(cell='alif_ref');x=self.x();a=scheduled_run(m,x);r=scheduled_run(m,x,[Edit('backbone.cells.1',1,lambda h,start:h,'joint',('v','a'))]);np.testing.assert_array_equal(r['prediction'],a['prediction'])
    def test_missing_auxiliary_owner_rejected(self):
        with self.assertRaises(ValueError):scheduled_run(self.model(),self.x(),[Edit('backbone.cells.1',1,lambda h,start:h,'aux',('a',))])
    def test_snapshot_zeroing_keeps_clock(self):
        m=self.model(readout='mean');s=StreamSession(m);s.step(torch.tensor(self.x()[:,0]));a=s.snapshot();b=zero_snapshot(a);self.assertEqual(a['readout']['count'],b['readout']['count']);self.assertEqual(a['readout']['cursor'],b['readout']['cursor'])
    def test_snapshot_cross_time_rejected(self):
        m=self.model();s=StreamSession(m);s.step(torch.tensor(self.x()[:,0]));a=s.snapshot();s.step(torch.tensor(self.x()[:,1]));b=s.snapshot()
        with self.assertRaises(ValueError):restore_scope(a,b,'core')
    def test_complete_snapshot_restores_exactly(self):
        m=self.model();x=torch.tensor(self.x());s=StreamSession(m);s.step(x[:,0]);s.step(x[:,1]);a=s.snapshot();s.step(x[:,2]);s.step(x[:,3]);expected=s.finish().numpy();s.restore(restore_scope(zero_snapshot(a),a,'system'));s.step(x[:,2]);s.step(x[:,3]);np.testing.assert_array_equal(s.finish().numpy(),expected)
    def test_core_restoration_leaves_mean_readout_damage(self):
        m=self.model(readout='mean');x=torch.tensor(self.x());s=StreamSession(m);s.step(x[:,0]);s.step(x[:,1]);a=s.snapshot();s.step(x[:,2]);s.step(x[:,3]);expected=s.finish().numpy();s.restore(restore_scope(zero_snapshot(a),a,'core'));s.step(x[:,2]);s.step(x[:,3]);self.assertGreater(np.max(np.abs(s.finish().numpy()-expected)),0)

if __name__=='__main__':unittest.main()
