import unittest
import numpy as np
from factor_data import (FactorConfig, SOURCES, TASKS, generate, make_pair,
    regression_target, classification_target, same_target_certified,
    model_payload, observed_sequence_reference, projection)
from paired_geometry import (paired_source_moment, linear_target_basis,
    aligned_power, paired_geometry_report)
from factor_data import make_noise_pair

class DataContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = FactorConfig()

    def test_prefix_and_chunk_identity(self):
        s = generate(self.cfg, 13, "audit_test")
        b = generate(self.cfg, 30, "audit_test")
        c = generate(self.cfg, 5, "audit_test", sample_ids=np.arange(8, 13))
        for key in ("x", "latents", "noise", "event_times", "group_ids"):
            np.testing.assert_array_equal(s[key], b[key][:13])
            np.testing.assert_array_equal(s[key][8:13], c[key])

    def test_balanced_timing_and_tags(self):
        b = generate(self.cfg, 60)
        pairs, counts = np.unique(b["event_times"][:, :2], axis=0, return_counts=True)
        self.assertEqual(len(pairs), 6)
        np.testing.assert_array_equal(counts, np.full(6, 10))
        self.assertTrue(np.all(b["x"][:, 13:] == 0))
        for j in range(3):
            tags = b["x"][:, :, self.cfg.sensory_dim+j]
            np.testing.assert_array_equal(tags.sum(1), np.ones(60))
            np.testing.assert_array_equal(tags.argmax(1), b["event_times"][:, j])
        self.assertEqual(set(model_payload(b)), {"x"})

    def test_split_separation_and_projection(self):
        a = generate(self.cfg, 20, "encoder_train")
        b = generate(self.cfg, 20, "audit_test")
        self.assertFalse(set(a["group_ids"]) & set(b["group_ids"]))
        self.assertFalse(np.array_equal(a["latents"], b["latents"]))
        np.testing.assert_allclose(projection(self.cfg).T@projection(self.cfg), np.eye(4), atol=1e-12)

    def test_single_source_pair_and_shared_noise(self):
        b = generate(self.cfg, 24, "audit_test")
        untouched = b["x"].copy()
        for j, name in enumerate(SOURCES):
            p = make_pair(self.cfg, b, name)
            for key in ("noise", "event_times", "group_ids"):
                np.testing.assert_array_equal(p[key], b[key])
            for k in range(3):
                if j != k:
                    np.testing.assert_array_equal(p["latents"][:, k], b["latents"][:, k])
            keep = np.ones(b["x"].shape[:2], dtype=bool)
            keep[np.arange(24), b["event_times"][:, j]] = False
            np.testing.assert_array_equal(p["x"][keep], b["x"][keep])
            np.testing.assert_array_equal(p["x"][:, :, self.cfg.sensory_dim:],
                                          b["x"][:, :, self.cfg.sensory_dim:])
            self.assertFalse(np.array_equal(p["latents"][:, j], b["latents"][:, j]))
        np.testing.assert_array_equal(b["x"], untouched)

    def test_targets_recomputed_and_mirror_certification(self):
        b = generate(self.cfg, 24, "audit_test")
        for history in ("A", "B"):
            for changed in SOURCES:
                p = make_pair(self.cfg, b, changed)
                for task in TASKS:
                    same = np.array_equal(regression_target(b, task, history),
                                          regression_target(p, task, history))
                    self.assertEqual(same_target_certified(task, changed, history), same)
                    if same:
                        np.testing.assert_array_equal(
                            classification_target(self.cfg, b, task, history),
                            classification_target(self.cfg, p, task, history))

    def test_noiseless_observation_reference(self):
        cfg = FactorConfig(sigma=0)
        b = generate(cfg, 60)
        for j, source in enumerate(SOURCES):
            np.testing.assert_allclose(observed_sequence_reference(cfg, b, source),
                                       b["latents"][:, j], atol=2e-7)

    def test_gaussian_reference_mse(self):
        b = generate(self.cfg, 6000, "audit_test")
        pred = observed_sequence_reference(self.cfg, b, "A")
        actual = np.mean((pred-b["latents"][:, 0])**2)
        self.assertAlmostEqual(actual, self.cfg.sigma**2/(1+self.cfg.sigma**2), delta=.002)

    def test_invalid_inputs(self):
        for c in [FactorConfig(sensory_dim=2), FactorConfig(sigma=float("nan")),
                  FactorConfig(history_slots=(2, 2)), FactorConfig(current_time=15)]:
            with self.assertRaises(ValueError):
                generate(c, 6)
        with self.assertRaises(ValueError):
            generate(self.cfg, 2, sample_ids=[1, 1])
        with self.assertRaises(ValueError):
            make_pair(self.cfg, generate(self.cfg, 6), "A", pair_seed=-1)

    def test_measurement_noise_pair_has_no_content_or_target_change(self):
        b = generate(self.cfg, 24, "audit_test")
        p = make_noise_pair(self.cfg, b)
        for key in ("latents", "event_times", "group_ids"):
            np.testing.assert_array_equal(b[key], p[key])
        self.assertFalse(np.array_equal(b["noise"], p["noise"]))
        for task in TASKS:
            np.testing.assert_array_equal(regression_target(b, task), regression_target(p, task))
        np.testing.assert_array_equal(b["x"][:, :, self.cfg.sensory_dim:],
                                      p["x"][:, :, self.cfg.sensory_dim:])

class GeometryContracts(unittest.TestCase):
    def test_known_source_and_orthogonal_response(self):
        rng = np.random.default_rng(9)
        a, d, d2 = [rng.normal(size=20000) for _ in range(3)]
        r, rp = np.column_stack((a, 3*d)), np.column_stack((a, 3*d2))
        S = paired_source_moment(r, rp)
        self.assertAlmostEqual(S[1, 1], 9, delta=.3)
        self.assertEqual(aligned_power(S, np.array([[1.], [0.]])), 0.)
        self.assertGreaterEqual(np.linalg.eigvalsh(S).min(), -1e-10)

    def test_rank_deficiency_and_zero_basis(self):
        z = np.random.default_rng(10).normal(size=(200, 1))
        r = np.column_stack((z[:, 0], 2*z[:, 0], 3*z[:, 0]))
        U, _ = linear_target_basis(r, z)
        self.assertEqual(U.shape, (3, 1))
        zero, _ = linear_target_basis(np.zeros_like(r), z)
        self.assertEqual(zero.shape, (3, 0))
        out = paired_geometry_report(np.eye(3), np.eye(3), zero)
        self.assertIsNone(out["nuisance_to_target_response_ratio"])
        self.assertIsNone(out["target_aligned_nuisance_response"])

    def test_rotation_and_uniform_scale_controls(self):
        S, D, U = np.diag([2., 1.]), np.array([[1., .2], [.2, 4.]]), np.array([[1.], [0.]])
        t = .4
        Q = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        before = paired_geometry_report(S, D, U)
        after = paired_geometry_report(Q@S@Q.T, Q@D@Q.T, Q@U)
        scaled = paired_geometry_report(.01*S, .01*D, U)
        self.assertAlmostEqual(before["nuisance_to_target_response_ratio"], after["nuisance_to_target_response_ratio"])
        self.assertAlmostEqual(before["nuisance_to_target_response_ratio"], scaled["nuisance_to_target_response_ratio"])

if __name__ == "__main__":
    unittest.main()

