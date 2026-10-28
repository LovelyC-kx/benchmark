"""Behavioral controls for content-resolved frozen operations."""
from dataclasses import replace
import unittest
import numpy as np
from experiment_plan import RunConfig, CELLS
from load_data import LoadConfig, generate, make_source_pair, model_payload
from analyze_interventions import fit_natural_probe, apply_locked_probe, paired_mse_change
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from torch_adapter import LoadModel, collect_features
    from intervention_features import collect_operation_features, OBSERVATIONS


@unittest.skipIf(torch is None, 'PyTorch is required for frozen-content runtime checks')
class FrozenContentChecks(unittest.TestCase):
    def test_features_replay_batch_and_all_core_clear(self):
        cfg = LoadConfig(k=4, required_history=4)
        data = generate(cfg, 6, 'development_test')
        paired = make_source_pair(cfg, data, 'H1', pair_seed=17)
        for cell in CELLS:
            with self.subTest(cell=cell):
                torch.manual_seed(11)
                model = LoadModel(RunConfig(data=cfg, cell=cell, width=8))
                natural, _ = collect_features(model, model_payload(data)['x'], batch_size=3)
                ops = collect_operation_features(model, model_payload(data)['x'], batch_size=3)
                changed = collect_operation_features(model, model_payload(paired)['x'], batch_size=3)
                for key in ('prediction',)+OBSERVATIONS:
                    np.testing.assert_allclose(ops['baseline'][key], natural[key], atol=1e-6, rtol=1e-5)
                    np.testing.assert_array_equal(ops['noop'][key], ops['baseline'][key])
                    np.testing.assert_array_equal(ops['clear_core'][key], changed['clear_core'][key])

    def test_current_content_still_enters_after_clear(self):
        cfg = LoadConfig()
        data = generate(cfg, 6, 'development_test')
        paired = make_source_pair(cfg, data, 'C', pair_seed=17)
        torch.manual_seed(19)
        model = LoadModel(RunConfig(data=cfg, cell='continuous', width=8))
        first = collect_operation_features(model, model_payload(data)['x'])
        second = collect_operation_features(model, model_payload(paired)['x'])
        self.assertGreater(float(np.max(np.abs(first['clear_core']['last_post']-
                                              second['clear_core']['last_post']))), 1e-6)

    def test_probe_stays_fixed_when_state_signal_is_cleared(self):
        rng = np.random.default_rng(502)
        x, val, test = (rng.normal(size=(n, 3)) for n in (128, 64, 64))
        y, vy, ty = (a[:, :2] for a in (x, val, test))
        probe = fit_natural_probe(x, y, val, vy)
        original = apply_locked_probe(probe, test)
        cleared = apply_locked_probe(probe, np.zeros_like(test))
        no_op = paired_mse_change(original, original, ty)
        effect = paired_mse_change(original, cleared, ty)
        self.assertEqual(no_op['paired_delta_mse'], 0.)
        self.assertEqual(no_op['sequence_bootstrap_95'], [0., 0.])
        self.assertGreater(effect['paired_delta_mse'], .5)
        self.assertLess(np.mean((original-ty)**2), 1e-8)


if __name__ == '__main__':
    unittest.main()
