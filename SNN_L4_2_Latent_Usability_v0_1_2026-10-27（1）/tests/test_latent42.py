import unittest
import numpy as np
import torch
from latent_theory42 import (gaussian_risk, gaussian_risk_chain, query_readability,
    task_metric, gaussian_extra_risk_reduction, affine_gauge, spectral_ranks, safe_whitener)
from run_latent_controls import run
from readers42 import budget_hidden, SmallReader, predict, fit_mlp, ridge_optimization_control
from bridge42 import fit_head, independent_ids

class LatentMathematics(unittest.TestCase):
    def test_exact_nested_gaussian_risks(self):
        r=run()['risk_chain']
        self.assertAlmostEqual(sum(r['pieces'].values()),1.1)
        for v in r['pieces'].values():self.assertGreater(v,0)
    def test_information_preserving_change_can_change_restricted_readability(self):
        r=run()['nonlinear_task_in_linear_latents']
        self.assertAlmostEqual(r['affine_MSE'],1)
        self.assertLess(r['product_reader_MSE'],1e-20)
    def test_query_family_requires_union_of_task_directions(self):
        r=run()['query_family']
        self.assertTrue(r['fixed_task']['readable']);self.assertFalse(r['two_late_queries']['readable'])
    def test_all_latent_average_can_hide_required_feature(self):
        r=run()['all_latents_vs_task']
        self.assertLess(r['all_latent_MSE'],.006);self.assertEqual(r['target_relative_MSE'],1)
    def test_output_metric_is_coordinate_covariant(self):
        r=run()['coordinate_gauge']
        self.assertAlmostEqual(r['prediction_difference'],0)
        self.assertAlmostEqual(r['original_task_step'],r['transformed_task_step'])
        self.assertGreater(r['transformed_Euclidean_step'],100*r['original_Euclidean_step'])
    def test_extra_access_conditional_gain_is_nonnegative(self):
        r=run()['conditional_utility'];self.assertAlmostEqual(r['gain'],1)
    def test_extra_redundant_information_has_no_conditional_gain(self):
        self.assertAlmostEqual(run()['redundant_extra_content']['gain'],0)
    def test_perfect_availability_does_not_force_native_reading(self):
        r=run()['accessible_but_native_ignored'];self.assertEqual(r['Bayes_task_risk'],0);self.assertEqual(r['native_task_risk'],2)
    def test_spatial_pooling_cannot_restore_pixel_identity(self):
        r=run()['spatial_reconstruction'];self.assertEqual(r['full_spatial_MSE'],0);self.assertAlmostEqual(r['GAP_MSE'],.5)
    def test_zero_one_and_logloss_are_distinct(self):
        r=run()['loss_specificity'];self.assertGreater(r['baseline_logloss'],r['better_logloss'])
    def test_gaussian_singular_covariance_is_supported(self):
        self.assertAlmostEqual(gaussian_risk(np.eye(2),np.array([[1.,0],[2.,0]]),np.array([[0.,1]])),1)
    def test_negative_or_asymmetric_covariance_is_rejected(self):
        for a in [np.diag([1.,-1.]),np.array([[1.,1.],[0.,1.]])]:
            with self.assertRaises(ValueError):gaussian_risk(a,np.eye(2),np.eye(2))
    def test_query_dimension_and_gauge_are_checked(self):
        with self.assertRaises(ValueError):query_readability(np.eye(2),np.eye(3))
        with self.assertRaises(ValueError):affine_gauge(np.eye(2),np.eye(2),np.zeros((2,2)))
    def test_nonfinite_code_is_rejected(self):
        with self.assertRaises(ValueError):gaussian_risk(np.eye(2),np.array([[np.nan,0]]),np.eye(2))
    def test_singular_value_entropy_is_not_covariance_entropy(self):
        r=spectral_ranks(np.array([[1.,0.],[.6,.8]]))
        self.assertGreater(r['singular_value_effective_rank'],r['covariance_entropy_rank'])
    def test_zero_code_rank_does_not_create_information(self):
        self.assertEqual(spectral_ranks(np.zeros((4,2)))['covariance_entropy_rank'],0)
        self.assertEqual(gaussian_risk(np.eye(2),np.zeros((1,2)),np.eye(2)),1)
    def test_range_whitening_retains_null_directions(self):
        x=np.c_[np.arange(20),np.arange(20),np.zeros(20)]
        r=safe_whitener(x);self.assertEqual(r['rank'],1)
        y=(x-r['center'])@r['matrix'];self.assertTrue(np.isfinite(y).all())
        self.assertAlmostEqual(np.trace(y.T@y/len(y)),1)
    def test_nonfinite_output_metric_rejected(self):
        with self.assertRaises(ValueError):task_metric(np.eye(2),np.array([[np.inf,0],[0,1.]]))

class ReaderControls(unittest.TestCase):
    def test_parameter_budget_for_all_used_dimensions(self):
        for d in (32,512):
            h=budget_hidden(d,8);m=SmallReader(d,8,h)
            self.assertLessEqual(sum(p.numel() for p in m.parameters()),(d+1)*8)
    def test_numpy_deployment_matches_torch_forward(self):
        torch.manual_seed(42);m=SmallReader(4,2,3);x=np.random.default_rng(4).normal(size=(16,4))
        b={'kind':'budget_mlp','state':{n:p.detach().numpy().copy() for n,p in m.state_dict().items()}}
        np.testing.assert_allclose(predict(b,x),m(torch.tensor(x,dtype=torch.float32)).detach().numpy(),atol=2e-7)
    def test_residual_contains_exact_affine_baseline_at_zero_adapter(self):
        x=np.random.default_rng(1).normal(size=(20,4));y=x[:,:2];h=fit_head(x,y)
        torch.manual_seed(1);m=SmallReader(4,2,3)
        torch.nn.init.zeros_(m.output.weight);torch.nn.init.zeros_(m.output.bias)
        b={'kind':'residual','base_head':h,'state':{n:p.detach().numpy().copy() for n,p in m.state_dict().items()}}
        np.testing.assert_array_equal(predict(b,x),(x-h[1])@h[0]+h[2])
    def test_empirical_optimizer_competes_with_same_closed_form_objective(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(256,4));y=x@rng.normal(size=(4,2))
        r=ridge_optimization_control(x,y,.1,np.zeros((4,2)),np.zeros(2),steps=100)
        for a in r['runs'].values():self.assertLess(a['excess_over_closed_form'],1e-10)
    def test_overlap_in_probe_splits_rejected(self):
        with self.assertRaises(ValueError):independent_ids(np.array([1,2]),np.array([2,3]))
    def test_residual_requires_same_permission_base(self):
        with self.assertRaises(ValueError):fit_mlp(np.ones((4,2)),np.ones((4,1)),np.ones((2,2)),np.ones((2,1)),'residual',1)
    def test_fit_and_selection_labels_are_counted_separately(self):
        from run_reader_ladder import BUDGETS
        self.assertEqual([p+p//4 for p in BUDGETS],[160,640])

if __name__=='__main__':unittest.main()
