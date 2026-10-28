import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subspace_geometry import (basis, projector, complement, overlap, covariance_capture,
    select_role_rank, fit_axes, independent_replica, random_basis, null_overlaps, svd_geometry,
    variance_matched_basis)
from readout_geometry import (covariance, choose_covariance_metric, noise_metric_geometry,
    mapping_diagnostics, affine_restrict, fit_projected_probe, predict_raw, metrics)
from bridge import fit_joint, gaussian_readout_reference
from sampling import fixed_batch, repeat_panel
from inherited import LoadConfig


class GeometryTests(unittest.TestCase):
    def test_zero_signal_has_no_artificial_axis_or_angle(self):
        q = basis(np.zeros((5, 2)))
        self.assertEqual(q.shape, (5, 0))
        self.assertEqual(overlap(q, np.eye(5)[:, :1])['status'], 'undefined_empty_subspace')

    def test_basis_and_complement_are_orthonormal_and_complete(self):
        q = basis(np.random.default_rng(1).normal(size=(8, 3)))
        c = complement(q)
        np.testing.assert_allclose(q.T @ c, 0., atol=1e-14)
        np.testing.assert_allclose(projector(q)+projector(c), np.eye(8), atol=1e-14)

    def test_asymmetric_overlap_remembers_subspace_dimensions(self):
        first = np.eye(5)[:, :1]; second = np.eye(5)[:, :3]
        result = overlap(first, second)
        self.assertEqual(result['overlap_a_into_b'], 1.)
        self.assertAlmostEqual(result['overlap_b_into_a'], 1/3)

    def test_angle_known_answer(self):
        angle = np.pi/3
        result = overlap(np.array([[1.], [0.]]), np.array([[np.cos(angle)], [np.sin(angle)]]))
        self.assertAlmostEqual(result['principal_angles_degrees'][0], 60.)
        self.assertAlmostEqual(result['overlap_a_into_b'], .25)

    def test_full_space_overlap_is_geometrically_forced(self):
        result = overlap(np.eye(4), np.eye(4))
        self.assertEqual(result['forced_intersection_dimension'], 4)
        self.assertEqual(result['haar_expected_a_into_b'], 1.)
        self.assertIsNone(result['chance_adjusted_a_into_b'])

    def test_haar_dimension_null_has_expected_overlap(self):
        result = null_overlaps(np.eye(10)[:, :2], 3, np.eye(10), draws=1200, seed=2)
        self.assertAlmostEqual(result['haar_dimension_matched']['mean'], .3, delta=.02)

    def test_covariance_shaped_null_differs_from_haar(self):
        cov = np.diag([1000., 1., 1., 1., 1.])
        result = null_overlaps(np.eye(5)[:, :1], 1, cov, draws=600, seed=3)
        self.assertGreater(result['covariance_shaped']['mean'], result['haar_dimension_matched']['mean']+.5)

    def test_covariance_null_rejects_insufficient_support(self):
        with self.assertRaises(ValueError):
            random_basis(4, 2, np.random.default_rng(4), np.diag([1., 0, 0, 0]))

    def test_constructed_control_matches_rank_and_retained_variance(self):
        cov = np.diag([.01, .1, 1., 10., 100.])
        q = variance_matched_basis(cov, 2, 73., np.random.default_rng(41))
        np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)
        self.assertAlmostEqual(covariance_capture(q, cov)['captured_power'], 73.)

    def test_constructed_control_handles_large_rank_via_complement(self):
        cov = np.diag([1., 2., 3., 4., 5.])
        q = variance_matched_basis(cov, 4, 12., np.random.default_rng(42))
        self.assertEqual(q.shape, (5, 4))
        np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-12)
        self.assertAlmostEqual(covariance_capture(q, cov)['captured_power'], 12.)

    def test_constructed_control_rejects_infeasible_variance(self):
        with self.assertRaises(ValueError):
            variance_matched_basis(np.eye(4), 1, 2., np.random.default_rng(43))

    def test_nonorthogonal_encoding_can_be_perfectly_decoded(self):
        af = np.array([[1.], [0.]]); au = np.array([[1.], [1.]])
        ref = gaussian_readout_reference(af, au, np.zeros((2, 2)))
        np.testing.assert_allclose(ref['w'], [[1., -1.]], atol=1e-12)
        self.assertAlmostEqual(ref['mse_per_focal_coordinate'], 0.)
        self.assertAlmostEqual(overlap(basis(af), basis(au))['overlap_a_into_b'], .5)

    def test_encoding_only_projection_can_lose_a_noise_reference(self):
        full = gaussian_readout_reference(np.array([[1.], [0.]]), np.array([[1.], [1.]]), np.zeros((2, 2)))
        reduced = gaussian_readout_reference(np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1)))
        self.assertAlmostEqual(full['mse_per_focal_coordinate'], 0.)
        self.assertAlmostEqual(reduced['mse_per_focal_coordinate'], .5)

    def test_decoder_weights_and_encoding_directions_can_differ(self):
        result = mapping_diagnostics(np.array([[1., -1.]]), np.array([[1.], [0.]]),
            np.array([[1.], [1.]]), np.empty((2, 0)), np.ones((2, 2)))
        self.assertAlmostEqual(result['readout_rowspace_vs_focal_encoding']['overlap_a_into_b'], .5)
        self.assertEqual(result['focal_gain_mixing_error_unit_latents'], 0.)
        self.assertEqual(result['cotarget_linear_injection_unit_latents'], 0.)
        self.assertEqual(result['conditional_output_variability'], 0.)

    def test_same_subspace_wrong_signed_readout_has_error(self):
        result = mapping_diagnostics(np.array([[-1.]]), np.ones((1, 1)), np.empty((1, 0)), np.empty((1, 0)), np.zeros((1, 1)))
        self.assertEqual(result['readout_rowspace_vs_focal_encoding']['overlap_a_into_b'], 1.)
        self.assertEqual(result['focal_gain_mixing_error_unit_latents'], 4.)

    def test_noise_metric_is_invariant_under_matched_invertible_transport(self):
        af = np.array([[1.], [0.]]); ag = np.array([[0.], [1.]])
        noise = np.diag([.1, .4]); t = np.array([[1., 2.], [.1, 1.]])
        first = noise_metric_geometry(af, ag, noise)
        second = noise_metric_geometry(t @ af, t @ ag, t @ noise @ t.T)
        self.assertAlmostEqual(first['metric_overlap']['overlap_a_into_b'], second['metric_overlap']['overlap_a_into_b'])
        np.testing.assert_allclose(first['gram'], second['gram'], atol=1e-12)

    def test_isotropic_shrinkage_does_not_claim_affine_invariance(self):
        af = np.array([[1.], [0.]]); ag = np.array([[0.], [1.]])
        cov = np.eye(2); t = np.array([[1., 2.], [0., 1.]])
        first = choose_covariance_metric(cov, cov, gammas=(1.,))
        second = choose_covariance_metric(t @ cov @ t.T, t @ cov @ t.T, gammas=(1.,))
        left = noise_metric_geometry(af, ag, first['regularized_covariance'])
        right = noise_metric_geometry(t @ af, t @ ag, second['regularized_covariance'])
        self.assertGreater(right['metric_overlap']['overlap_a_into_b'], left['metric_overlap']['overlap_a_into_b']+.5)

    def test_singular_noise_metric_requires_explicit_policy(self):
        with self.assertRaises(ValueError):
            noise_metric_geometry(np.ones((2, 1)), np.ones((2, 1)), np.diag([1., 0.]))

    def test_same_angle_different_signal_amplitude_different_risk(self):
        af = np.array([[1.], [0.]]); other = np.empty((2, 0)); noise = np.eye(2)
        strong = gaussian_readout_reference(af, other, noise)
        weak = gaussian_readout_reference(.1*af, other, noise)
        self.assertGreater(weak['mse_per_focal_coordinate'], strong['mse_per_focal_coordinate'])
        self.assertEqual(overlap(basis(af), basis(.1*af))['overlap_a_into_b'], 1.)

    def test_near_parallel_is_readable_without_noise_but_fragile_with_noise(self):
        af = np.array([[1.], [0.]]); theta = np.deg2rad(1.)
        au = np.array([[np.cos(theta)], [np.sin(theta)]])
        clean = gaussian_readout_reference(af, au, np.zeros((2, 2)))
        noisy = gaussian_readout_reference(af, au, .01*np.eye(2))
        self.assertAlmostEqual(clean['mse_per_focal_coordinate'], 0., places=9)
        self.assertGreater(noisy['mse_per_focal_coordinate'], .45)

    def test_nonlinear_focal_signal_is_not_conditional_noise(self):
        rng = np.random.default_rng(18); f = rng.normal(size=(120000, 1))
        r = np.c_[f[:, 0], f[:, 0]**2-1]+.2*rng.normal(size=(len(f), 2))
        residual = r-np.c_[f[:, 0], np.zeros(len(f))]
        cov = covariance(residual)
        self.assertAlmostEqual(cov[0, 0], .04, delta=.002)
        self.assertAlmostEqual(cov[1, 1], 2.04, delta=.04)
        # Given F, the actual conditional covariance is .04 I, not this residual.
        self.assertGreater(cov[1, 1], .04*40)

    def test_exact_rank_one_encoder_is_selected_on_independent_data(self):
        rng = np.random.default_rng(19); z = rng.normal(size=(500, 3)); zv = rng.normal(size=(300, 3))
        a = np.array([[1., 0., 0.], [0., 0., 1.]])
        _, axes = fit_axes(z @ a.T, z, ['focal', 'focal', 'cotarget'], zv @ a.T, zv)
        self.assertEqual(axes['focal']['selection']['selected_rank'], 1)

    def test_rank_zero_allowed_despite_nonzero_fitted_axis(self):
        z = np.tile([[-1.], [1.]], (30, 1)); r = z.copy()
        fit = fit_joint(r, z, ['focal'])
        result = select_role_rank(fit, 'focal', np.zeros_like(r), z)
        self.assertEqual(result['numeric_rank'], 1)
        self.assertEqual(result['selected_rank'], 0)

    def test_rank_deficient_joint_design_is_not_silently_attributed(self):
        z = np.tile([[-1., -1.], [1., 1.]], (30, 1))
        with self.assertRaises(ValueError): fit_axes(z, z, ['focal', 'cotarget'], z, z)

    def test_independent_encoding_replica_recovers_known_subspaces(self):
        rng = np.random.default_rng(22); a = np.eye(4)[:, :3]
        z, zv, zb = (rng.normal(size=(150, 3)) for _ in range(3))
        fit, axes = fit_axes(z @ a.T, z, ['focal', 'cotarget', 'unrequested'], zv @ a.T, zv)
        _, _, result = independent_replica(fit, axes, zb @ a.T, zb, ['focal', 'cotarget', 'unrequested'])
        for role in result: self.assertAlmostEqual(result[role]['subspace_stability']['overlap_a_into_b'], 1.)

    def test_noise_covariance_selection_uses_an_independent_score(self):
        cov = np.diag([.1, 1., 5.])
        result = choose_covariance_metric(cov, cov)
        self.assertEqual(result['gamma'], 0.)
        np.testing.assert_allclose(result['regularized_covariance'], cov)

    def test_zero_conditional_covariance_has_no_invented_metric(self):
        result = choose_covariance_metric(np.zeros((3, 3)), np.zeros((3, 3)))
        self.assertEqual(result['status'], 'undefined_zero_conditional_covariance')
        self.assertIsNone(result['metric'])

    def test_indefinite_corrected_covariance_is_not_a_power_matrix(self):
        with self.assertRaises(ValueError): covariance_capture(np.eye(2), np.diag([1., -.1]))

    def test_affine_full_observation_is_exact_noop(self):
        r = np.random.default_rng(25).normal(size=(100, 6)); mean = np.arange(6)
        np.testing.assert_allclose(affine_restrict(r, mean, np.eye(6)), r, atol=1e-14)

    def test_isotropic_ridge_is_invariant_to_orthogonal_basis_rotation(self):
        rng = np.random.default_rng(26)
        tr, vr, er = (rng.normal(size=(100, 4)) for _ in range(3))
        a = rng.normal(size=(4, 2)); ty = tr @ a; vy = vr @ a
        q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        first = fit_projected_probe(tr, ty, vr, vy, np.eye(4), np.zeros(4), 2., lambdas=(.1,))
        second = fit_projected_probe(tr, ty, vr, vy, q, np.zeros(4), 2., lambdas=(.1,))
        np.testing.assert_allclose(predict_raw(first, er), predict_raw(second, er), atol=1e-12)
        self.assertEqual(first['alpha'], len(tr)*.1)

    def test_rank_zero_probe_keeps_all_target_coordinates(self):
        rng = np.random.default_rng(27); tr = rng.normal(size=(40, 4)); y = rng.normal(size=(40, 3))
        probe = fit_projected_probe(tr, y, tr, y, np.empty((4, 0)), np.zeros(4), 1.)
        prediction = predict_raw(probe, tr)
        self.assertEqual(prediction.shape, y.shape)
        self.assertEqual(probe['readout_parameter_count'], 3)
        np.testing.assert_allclose(prediction, np.broadcast_to(y.mean(0), y.shape))

    def test_metrics_retain_negative_r2(self):
        y = np.array([[-1.], [1.]])
        self.assertEqual(metrics(-y, y)['mean_r2'], -3.)

    def test_schedules_pair_contents_without_confusing_source_times(self):
        cfg = LoadConfig(required_history=1)
        first = fixed_batch(cfg, 12, 'development_test', 2800000, 0)
        second = fixed_batch(cfg, 12, 'development_test', 2800000, 5)
        np.testing.assert_array_equal(first['latents'], second['latents'])
        np.testing.assert_array_equal(first['measurement_noise'], second['measurement_noise'])
        self.assertFalse(np.array_equal(first['event_times'], second['event_times']))

    def test_repeat_focal_contract_handles_same_source_unused_coordinates(self):
        cfg = LoadConfig(k=1, required_history=1, content_mode='demand_only')
        panel = repeat_panel(cfg, 8, 4, 'geometry_fit', 2400000, 2, 'C')
        np.testing.assert_array_equal(panel['latents'][:, 0, 4, 0], panel['latents'][:, 1, 4, 0])
        self.assertFalse(np.array_equal(panel['latents'][:, 0, 4, 1:], panel['latents'][:, 1, 4, 1:]))

    def test_analysis_focal_relabelling_does_not_change_joint_encoding(self):
        rng = np.random.default_rng(31); z = rng.normal(size=(100, 3)); r = rng.normal(size=(100, 5))
        first = fit_joint(r, z, ['focal', 'cotarget', 'unrequested'])
        second = fit_joint(r, z, ['cotarget', 'focal', 'unrequested'])
        np.testing.assert_array_equal(first['a'], second['a'])


if __name__ == '__main__': unittest.main()
