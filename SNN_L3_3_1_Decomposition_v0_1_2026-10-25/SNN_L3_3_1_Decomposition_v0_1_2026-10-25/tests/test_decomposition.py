import unittest
import numpy as np
from inherited import LoadConfig, generate
from roles import coordinate_roles, flatten_contents
from decomposition import (fit_joint, apply_joint, covariance_accounting,
                           encoding_diagnostics, hermite_second_order, representation_score)
from repeat_panels import repeat_panel, within_scene_covariance


class DefinitionChecks(unittest.TestCase):
    def test_roles_separate_cotarget_and_unrequested(self):
        rows = coordinate_roles(LoadConfig(required_history=1))
        roles = [row['role'] for row in rows]
        self.assertEqual([roles.count(k) for k in ('focal', 'cotarget', 'unrequested')], [4, 4, 12])
        rows = coordinate_roles(LoadConfig(required_history=4))
        self.assertEqual(sum(row['role']=='unrequested' for row in rows), 0)

    def test_demand_only_extra_coordinates_keep_their_roles(self):
        cfg = LoadConfig(k=2, required_history=1, content_mode='demand_only')
        rows = coordinate_roles(cfg)
        roles = [row['role'] for row in rows]
        self.assertEqual([roles.count(k) for k in ('focal', 'cotarget', 'unrequested')], [2, 2, 16])
        self.assertTrue(all(row['role']=='unrequested' for row in rows if row['coordinate']>=2))

    def test_unrequested_focal_and_partial_presence_require_new_contract(self):
        with self.assertRaises(ValueError): coordinate_roles(LoadConfig(required_history=1), 'H2')
        with self.assertRaises(ValueError): coordinate_roles(LoadConfig(), source_present=np.zeros((2,5), bool))

    def test_linear_components_reconstruct_independent_evaluation(self):
        rng = np.random.default_rng(100)
        z, ze = rng.normal(size=(256,3)), rng.normal(size=(128,3))
        a = np.array([[1,1,0], [0,2,1.]])
        fit = fit_joint(z@a.T+2, z, ['focal','cotarget','unrequested'])
        r = ze@a.T+2
        parts = apply_joint(fit, r, ze)
        np.testing.assert_allclose(sum(parts.values())+fit['rmean'], r, atol=1e-12)
        self.assertLess(np.max(np.abs(parts['unexplained_by_joint_linear_model'])), 1e-12)

    def test_zero_joint_residual_does_not_remove_other_content(self):
        rng = np.random.default_rng(101)
        z, ze = rng.normal(size=(4096,2)), rng.normal(size=(20000,2))
        fit = fit_joint(z.sum(1,keepdims=True), z, ['focal','cotarget'])
        r = ze.sum(1,keepdims=True)
        parts = apply_joint(fit, r, ze)
        self.assertLess(np.var(parts['unexplained_by_joint_linear_model']), 1e-24)
        self.assertGreater(np.var(parts['cotarget']), .9)
        # Population best linear readout is 0.5*r, with R²=0.5.
        r2 = 1-np.mean((.5*r[:,0]-ze[:,0])**2)/np.var(ze[:,0])
        self.assertAlmostEqual(r2, .5, delta=.025)

    def test_cross_terms_are_kept_on_shifted_finite_evaluation(self):
        rng = np.random.default_rng(102)
        z, ze = rng.normal(size=(40,3)), rng.normal(size=(30,3))
        r = z@rng.normal(size=(3,4))+rng.normal(size=(40,4))
        fit = fit_joint(r,z,['focal','cotarget','unrequested'])
        er = ze@rng.normal(size=(3,4))+rng.normal(size=(30,4))
        accounting = covariance_accounting(er, apply_joint(fit,er,ze))
        self.assertLess(accounting['max_covariance_identity_error'], 1e-12)
        self.assertGreater(sum(abs(v) for v in accounting['cross_trace'].values()), .01)

    def test_rank_deficiency_is_flagged(self):
        x = np.arange(20, dtype=float)
        z = np.stack([x,x],1)
        fit = fit_joint(x[:,None], z, ['focal','cotarget'])
        self.assertFalse(fit['coefficients_identifiable'])
        self.assertEqual(fit['design_rank'], 1)

    def test_constant_representation_has_no_fake_r2(self):
        r = np.ones((10,3))
        self.assertIsNone(representation_score(r,r,r.mean(0))['heldout_representation_r2'])

    def test_hermite_basis_dimension_and_moments(self):
        rng = np.random.default_rng(103)
        z = rng.normal(size=(50000,2))
        x, terms = hermite_second_order(z, [0,1], 'interaction2')
        self.assertEqual(x.shape[1], 5)
        self.assertEqual(terms[-1]['family'], 'cross_source_product')
        np.testing.assert_allclose(x.mean(0), 0, atol=.035)
        np.testing.assert_allclose(np.diag(np.cov(x,rowvar=False)), 1, atol=.06)

    def test_product_is_unexplained_linearly_but_recovered_by_interaction_basis(self):
        rng = np.random.default_rng(104)
        pairs = []
        for n in (512,256,1024):
            z = rng.normal(size=(n,2)); pairs.append((z, (z[:,0]*z[:,1])[:,None]))
        scores = encoding_diagnostics(*pairs, [0,1], alphas=(0., .01, 1.))
        self.assertGreater(scores[0]['evaluation']['mse_per_state_coordinate'], .5)
        self.assertGreater(scores[1]['evaluation']['mse_per_state_coordinate'], .5)
        self.assertLess(scores[2]['evaluation']['mse_per_state_coordinate'], 1e-20)

    def test_source_local_nonlinearity_is_not_called_cross_source_interaction(self):
        rng = np.random.default_rng(105)
        pairs = []
        for n in (512,256,1024):
            z = rng.normal(size=(n,2)); pairs.append((z, ((z[:,0]**2-1)/np.sqrt(2))[:,None]))
        scores = encoding_diagnostics(*pairs, [0,1], alphas=(0., .01, 1.))
        self.assertGreater(scores[0]['evaluation']['mse_per_state_coordinate'], .5)
        self.assertLess(scores[1]['evaluation']['mse_per_state_coordinate'], 1e-20)

    def test_repeat_modes_preserve_the_declared_inputs(self):
        cfg = LoadConfig(k=2, required_history=1, content_mode='demand_only')
        same = repeat_panel(cfg, scenes=6, repeats=3, mode='identical_input')
        np.testing.assert_array_equal(same['x'][:,0], same['x'][:,1])
        noise = repeat_panel(cfg, scenes=6, repeats=3, mode='measurement_only')
        np.testing.assert_array_equal(noise['latents'][:,0], noise['latents'][:,1])
        self.assertGreater(np.max(abs(noise['x'][:,0]-noise['x'][:,1])), .1)
        focal = repeat_panel(cfg, scenes=6, repeats=3, mode='focal_fixed_full')
        np.testing.assert_array_equal(focal['latents'][:,0,4,:2], focal['latents'][:,1,4,:2])
        self.assertGreater(np.max(abs(focal['latents'][:,0,4,2:]-focal['latents'][:,1,4,2:])), .1)

    def test_measurement_zero_and_independent_panels(self):
        cfg = LoadConfig(sigma=0)
        p = repeat_panel(cfg, scenes=6, repeats=3, mode='measurement_only')
        np.testing.assert_array_equal(p['x'][:,0],p['x'][:,1])
        other = repeat_panel(cfg, scenes=6, repeats=3, split='geometry_fit')
        self.assertFalse(np.array_equal(p['latents'],other['latents']))

    def test_repeated_variability_matches_known_conditioning(self):
        rng = np.random.default_rng(106)
        scenes, repeats = 4000, 16
        focal = rng.normal(size=(scenes,1,1))
        other = rng.normal(size=(scenes,1,1))
        measurement = focal+other+.1*rng.normal(size=(scenes,repeats,1))
        focal_fixed = focal+rng.normal(size=(scenes,repeats,1))+.1*rng.normal(size=(scenes,repeats,1))
        self.assertAlmostEqual(within_scene_covariance(measurement)['trace'], .01, delta=.001)
        self.assertAlmostEqual(within_scene_covariance(focal_fixed)['trace'], 1.01, delta=.03)


if __name__ == '__main__': unittest.main()
