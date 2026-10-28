import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_diagnostics import (role_risk_budget, gaussian_readout_reference,
    projected_variance, pickfreeze_statistics, repeated_signal_and_noise,
    repeated_prediction_risk, scalar_lif)


class DeepDiagnosticsTests(unittest.TestCase):
    def test_role_error_budget_keeps_offsets_and_cross_terms(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=(100, 4))+2
        a = rng.normal(size=(3, 4)); r = z @ a.T+rng.normal(size=(100, 3))
        result = role_risk_budget(r, z, a, np.ones(4), np.ones(3),
            ['focal', 'cotarget', 'unrequested', 'focal'], rng.normal(size=(2, 3)), np.ones(2))
        self.assertAlmostEqual(result['direct_mse'], result['accounted_mse'], places=10)
        self.assertLess(result['max_error_reconstruction_difference'], 1e-12)
        self.assertEqual(len(result['cross_second_moments']), 10)

    def test_cross_term_can_cancel_large_component_errors(self):
        z = np.array([[-1., -1.], [1., 1.]])
        r = z[:, :1]
        result = role_risk_budget(r, z, np.array([[1., 1.]]), np.zeros(2), np.zeros(1),
            ['focal', 'unrequested'], np.ones((1, 1)), np.zeros(1))
        self.assertEqual(result['direct_mse'], 0.)
        self.assertLess(result['cross_second_moments']['unrequested_injection__linear_unexplained_through_head'], 0.)

    def test_identical_encoding_directions_have_half_mmse(self):
        ref = gaussian_readout_reference(np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1)))
        self.assertAlmostEqual(ref['mse_per_focal_coordinate'], .5)

    def test_orthogonal_noiseless_coordinates_are_exactly_readable(self):
        ref = gaussian_readout_reference(np.array([[1.], [0.]]), np.array([[0.], [1.]]), np.zeros((2, 2)))
        self.assertAlmostEqual(ref['mse_per_focal_coordinate'], 0.)

    def test_invertible_shear_changes_angle_without_changing_mmse(self):
        af = np.array([[1.], [0.]]); ao = np.array([[0.], [1.]])
        noise = .04*np.eye(2); transform = np.array([[1., 2.], [0., 1.]])
        first = gaussian_readout_reference(af, ao, noise)
        second = gaussian_readout_reference(transform @ af, transform @ ao, transform @ noise @ transform.T)
        self.assertAlmostEqual(first['mse_per_focal_coordinate'], second['mse_per_focal_coordinate'])
        self.assertGreater(float(((transform @ af).T @ (transform @ ao)).item()), 1.)
        np.testing.assert_allclose(second['w'] @ transform, first['w'], atol=1e-12)

    def test_scaling_signal_and_noise_together_does_not_change_risk(self):
        a = np.ones((1, 1)); other = np.empty((1, 0))
        ref = gaussian_readout_reference(a, other, np.array([[.1]]))
        scaled = gaussian_readout_reference(.01*a, other, np.array([[.1e-4]]))
        self.assertAlmostEqual(ref['mse_per_focal_coordinate'], scaled['mse_per_focal_coordinate'])

    def test_equal_noise_trace_can_produce_different_risks(self):
        af = np.array([[1.], [0.]]); ao = np.empty((2, 0))
        parallel = gaussian_readout_reference(af, ao, np.diag([1., .01]))
        perpendicular = gaussian_readout_reference(af, ao, np.diag([.01, 1.]))
        self.assertGreater(parallel['mse_per_focal_coordinate'], perpendicular['mse_per_focal_coordinate']*20)

    def test_invalid_population_covariance_is_rejected(self):
        with self.assertRaises(ValueError):
            gaussian_readout_reference(np.ones((1, 1)), np.empty((1, 0)), np.array([[-1.]]))

    def test_pure_product_has_zero_main_but_nonzero_total_effect(self):
        grid = np.asarray(list(itertools.product([-1., 1.], repeat=4)))
        x, y, xp, yp = grid.T
        result = pickfreeze_statistics((x*y)[:, None], (xp*yp)[:, None], (x*yp)[:, None])
        self.assertEqual(result['main_effect_weighted_variance'], 0.)
        self.assertEqual(result['total_effect_weighted_variance'], 1.)

    def test_additive_factor_main_matches_total(self):
        rng = np.random.default_rng(7); a = rng.normal(size=(100000, 2)); b = rng.normal(size=(100000, 2))
        result = pickfreeze_statistics((2*a[:, 0]+3*a[:, 1])[:, None],
            (2*b[:, 0]+3*b[:, 1])[:, None], (2*a[:, 0]+3*b[:, 1])[:, None])
        self.assertAlmostEqual(result['main_effect_weighted_variance'], 4., delta=.15)
        self.assertAlmostEqual(result['total_effect_weighted_variance'], 4., delta=.1)

    def test_independent_repeat_correction_with_heteroscedastic_noise(self):
        rng = np.random.default_rng(11)
        signal = rng.normal(size=(30000, 1, 1))
        scale = np.sqrt(.5+2*signal**2)
        responses = signal+scale*rng.normal(size=(30000, 4, 1))
        result = repeated_signal_and_noise(responses)
        self.assertAlmostEqual(float(result['corrected_between_covariance'][0, 0]), 1., delta=.06)
        self.assertAlmostEqual(float(result['cross_replica_covariance'][0, 0]), 1., delta=.07)
        self.assertGreater(float(result['raw_between_covariance'][0, 0]), 1.5)

    def test_repeated_risk_exact_identity_and_negative_correction_allowed(self):
        predictions = np.array([[[-1.], [1.]], [[-1.], [1.]]])
        result = repeated_prediction_risk(predictions, np.zeros((2, 1)))
        self.assertEqual(result['direct_mse'], 1.)
        self.assertEqual(result['unbiased_conditional_mean_error_mse'], -1.)
        self.assertEqual(result['identity_error'], 0.)

    def test_constant_output_has_zero_variability_but_cannot_recover_target(self):
        target = np.array([[-1.], [1.]])
        result = repeated_prediction_risk(np.zeros((2, 4, 1)), target)
        self.assertEqual(result['unbiased_conditional_variability_mse'], 0.)
        self.assertEqual(result['unbiased_conditional_mean_error_mse'], 1.)
        self.assertEqual(result['direct_mse'], 1.)

    def test_conditional_risk_recovers_known_gain_error_and_variability(self):
        rng = np.random.default_rng(17); z = rng.normal(size=(30000, 1))
        p = .5*(z[:, None]+rng.normal(size=(30000, 8, 1))+.2*rng.normal(size=(30000, 8, 1)))
        result = repeated_prediction_risk(p, z)
        self.assertAlmostEqual(result['unbiased_conditional_mean_error_mse'], .25, delta=.01)
        self.assertAlmostEqual(result['unbiased_conditional_variability_mse'], .26, delta=.01)
        self.assertAlmostEqual(result['direct_mse'], result['accounted_mse'])

    def test_context_can_cancel_pooled_encoding_and_enable_nonlinear_decoding(self):
        z, context = np.asarray(list(itertools.product([-1., 1.], repeat=2))).T
        r = context*z
        self.assertEqual(float(z @ r), 0.)
        np.testing.assert_array_equal(context*r, z)
        x = np.stack((r, context), axis=1)
        coef = np.linalg.lstsq(x, z, rcond=None)[0]
        self.assertAlmostEqual(float(np.mean((x @ coef-z)**2)), 1.)

    def test_frozen_hard_path_erases_sensitivity_after_reset(self):
        currents = np.array([.5, 1., .1]); events = scalar_lif(currents)['events']
        before = scalar_lif(currents-np.array([1e-5, 0, 0]), forced_events=events)['post'][-1]
        after = scalar_lif(currents+np.array([1e-5, 0, 0]), forced_events=events)['post'][-1]
        self.assertEqual(after-before, 0.)

    def test_frozen_soft_path_retains_beta_power_sensitivity(self):
        currents = np.array([.5, 1., .1]); events = scalar_lif(currents, kind='soft')['events']
        before = scalar_lif(currents-np.array([1e-5, 0, 0]), kind='soft', forced_events=events)['post'][-1]
        after = scalar_lif(currents+np.array([1e-5, 0, 0]), kind='soft', forced_events=events)['post'][-1]
        self.assertAlmostEqual((after-before)/2e-5, .8**2)

    def test_hard_and_soft_reset_both_have_discrete_threshold_jumps(self):
        for kind in ('hard', 'soft'):
            left = scalar_lif([1-1e-6], kind=kind)['post'][0]
            right = scalar_lif([1+1e-6], kind=kind)['post'][0]
            self.assertLess(right-left, -.99)

    def test_bias_drive_can_create_events_during_external_blank(self):
        below = scalar_lif(np.full(30, .1), beta=.8, kind='hard')
        above = scalar_lif(np.full(30, .3), beta=.8, kind='hard')
        self.assertEqual(int(below['events'].sum()), 0)
        self.assertGreater(int(above['events'].sum()), 0)

    def test_natural_reset_can_leave_history_in_event_phase(self):
        base_current = np.full(15, .3)
        changed_current = base_current.copy(); changed_current[0] += .4
        base = scalar_lif(base_current, beta=.8, kind='hard')
        changed = scalar_lif(changed_current, beta=.8, kind='hard')
        # Each realized-path direct kernel from the initial current has died.
        self.assertEqual(int(np.prod(1-base['events'])), 0)
        self.assertEqual(int(np.prod(1-changed['events'])), 0)
        # The intervention changed reset TIMES, retaining a phase difference.
        self.assertEqual(float(base['post'][-1]), 0.)
        self.assertAlmostEqual(float(changed['post'][-1]), .3)
        np.testing.assert_array_equal(np.flatnonzero(base['events']), [4, 9, 14])
        np.testing.assert_array_equal(np.flatnonzero(changed['events']), [3, 8, 13])

    def test_same_spike_count_does_not_imply_same_temporal_readout(self):
        first = np.array([1., 0., 1., 0.]); second = np.array([0., 1., 0., 1.])
        self.assertEqual(first.sum(), second.sum())
        self.assertNotEqual(scalar_lif(first, beta=.5, kind='none')['post'][-1],
                            scalar_lif(second, beta=.5, kind='none')['post'][-1])

    def test_explicit_conditioning_fixes_schedule_across_random_parent_blocks(self):
        from run_snn_deepening import fixed_schedule_batch
        from inherited import LoadConfig
        batch = fixed_schedule_batch(LoadConfig(required_history=1), 19, 'development_test', 840000)
        np.testing.assert_array_equal(batch['event_times'], np.tile([2, 5, 8, 11, 12], (19, 1)))
        np.testing.assert_array_equal(batch['x'][:, :, 12].argmax(1), np.full(19, 12))

    def test_pickfreeze_hybrid_preserves_only_its_declared_role(self):
        from run_snn_deepening import fixed_schedule_batch, pickfreeze_inputs
        from inherited import LoadConfig
        from roles import coordinate_roles
        cfg = LoadConfig(k=1, required_history=1, content_mode='demand_only')
        batch = fixed_schedule_batch(cfg, 12, 'development_test', 840000)
        columns = coordinate_roles(cfg)
        inputs, metadata = pickfreeze_inputs(cfg, batch, columns)
        np.testing.assert_array_equal(metadata['focal']['latents'][:, 4, 0], batch['latents'][:, 4, 0])
        self.assertFalse(np.array_equal(metadata['focal']['latents'][:, 4, 1:], batch['latents'][:, 4, 1:]))
        np.testing.assert_array_equal(metadata['measurement']['measurement_noise'], batch['measurement_noise'])
        self.assertFalse(np.array_equal(metadata['measurement']['latents'], batch['latents']))
        # Same event identities, schedules and all non-sensory input channels.
        for name in inputs:
            np.testing.assert_array_equal(inputs[name][:, :, 8:], inputs['a'][:, :, 8:])


if __name__ == '__main__':
    unittest.main()
