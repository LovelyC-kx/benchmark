import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from temporal_alignment import fit_transport,transport,compose_transports,cycle_diagnostics
from run_temporal_consistency import known_consistency
from phase_readers import phase_features,known_phase_control,clock_views
from temporal_capture import base_capture,named_temporal_bank
from bridge35 import BenchmarkModel,ModelConfig
from certify_delay_readers import certify
from run_event_scale_scan import quantize_observation
ROOT=Path(__file__).resolve().parents[1]

class ClosureScience(unittest.TestCase):
    def test_affine_composition_includes_offsets(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(100,3));y=x+[2,5,7];z=2*y+[-3,2,1]
        a=fit_transport(x,y,'center');b=fit_transport(y,z,'similarity');c=compose_transports(a,b)
        np.testing.assert_allclose(transport(c,x),z,atol=2e-14)
    def test_equal_linear_matrices_can_have_broken_offset_cycle(self):
        a={'w':np.eye(2),'destination_mean':np.zeros(2),'source_mean':np.ones(2)}
        b=dict(a);direct=dict(a)
        r=cycle_diagnostics(a,b,direct,np.ones((10,2)))
        self.assertEqual(r['matrix_disagreement'],0);self.assertEqual(r['offset_disagreement'],1)
        self.assertGreater(r['heldout_prediction_MSE'],0)
    def test_cycle_does_not_imply_time_translation(self):
        r=known_consistency()
        for row in r.values():self.assertLess(row['cycle']['heldout_prediction_MSE'],1e-25)
        self.assertLess(r['constant_speed']['equal_lag_matrix_difference'],1e-12)
        self.assertGreater(r['changing_speed']['equal_lag_matrix_difference'],.5)
    def test_phase_side_information_restores_rotated_content(self):
        r=known_phase_control();self.assertGreater(r['raw_MSE'],.8);self.assertLess(r['phase_conditioned_MSE'],.001)
    def test_phase_invalid_does_not_encode_arbitrary_zero_phase(self):
        x=np.ones((2,3,4));p=np.ones((2,3));v=np.zeros((2,3),bool)
        features=phase_features(x,p,v)
        np.testing.assert_array_equal(features[...,:4],x);self.assertEqual(np.abs(features[...,4:]).max(),0)
    def test_phase_uses_only_past_prefix(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(3,12,4));e=np.zeros((3,12));e[:,2]=1
        z=x.copy();z[:,8:]+=100;f=e.copy();f[:,9]=1
        a=clock_views(x,e,np.ones(3,dtype=int)*2);b=clock_views(z,f,np.ones(3,dtype=int)*2)
        for key in a:np.testing.assert_array_equal(a[key][:,:8],b[key][:,:8])
    def test_phase_only_features_have_no_state_amplitude(self):
        a=phase_features(np.ones((2,4,3)),np.ones((2,4)),phase_only=True)
        b=phase_features(np.ones((2,4,3))*999,np.ones((2,4)),phase_only=True)
        np.testing.assert_array_equal(a,b)
    def test_analogue_reset_cell_output_is_not_called_spike(self):
        torch.set_num_threads(1);torch.manual_seed(10)
        model=BenchmarkModel((4,),(2,),3,ModelConfig(neuron='analog_hard_ref',width=4))
        x=np.random.default_rng(10).normal(size=(2,3,4)).astype(np.float32)
        bank=base_capture(model,x)
        self.assertIn('first_analog_output',bank);self.assertIn('first_spike',bank)
        np.testing.assert_array_equal(bank['first_analog_output'],bank['first_pre'])
    def test_image_coordinates_survive_named_bank(self):
        torch.set_num_threads(1);model=BenchmarkModel((1,8,8),(2,),3,ModelConfig(backbone='cnn2d',width=2))
        x=np.random.default_rng(11).normal(size=(2,3,1,8,8)).astype(np.float32)
        result=named_temporal_bank(model,x);key='backbone.a.act/post'
        self.assertEqual(result['carriers'][key].shape,(2,3,128))
        self.assertEqual(result['view_specification'][key]['unflattened_feature_shape'],[2,8,8])
        with self.assertRaisesRegex(ValueError,'aliases'):base_capture(model,x)
    def test_wrong_certification_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'best.pt').write_bytes(b'wrong checkpoint')
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):certify(run_directory=path)
    def test_precision_is_observer_rounding_not_clipping(self):
        x=np.array([-99.,-.29,.29,99.]);y=quantize_observation(x,.5)
        np.testing.assert_array_equal(y,[-99.,-.5,.5,99.])
        np.testing.assert_array_equal(quantize_observation(x,None),x)
        with self.assertRaises(ValueError):quantize_observation(x,0)
    def test_finite_quantizer_clips_and_respects_level_budget(self):
        x=np.linspace(-99,99,10000);q=quantize_observation(x,2/3,1)
        self.assertEqual(len(np.unique(q)),4);self.assertEqual(q.min(),-1);self.assertEqual(q.max(),1)
    def test_suite_propagates_custom_checkpoint_to_every_model_analysis(self):
        from l36_suite import build_commands
        commands=build_commands('/tmp/example_l36','/tmp/my_checkpoint')
        for name,cmd in commands:
            if name not in ('known_controls','model_interfaces'):
                self.assertIn('--run-directory',cmd);self.assertIn('/tmp/my_checkpoint',cmd)
    def test_repeated_average_roundoff_not_mistaken_for_unbounded_loss(self):
        from temporal_probes import paired_bound
        r=paired_bound(np.array([1.+4e-16,.5]),np.array([.5,.5]),1)
        self.assertEqual(r['gain'],.25)
        with self.assertRaises(ValueError):paired_bound(np.array([1.+1e-6]),np.array([0.]),1)
    def test_bound_rejects_invalid_family_or_delta(self):
        from temporal_probes import paired_bound
        for family,delta in [(0,.05),(1,0),(1,1),(1.5,.05)]:
            with self.assertRaises(ValueError):paired_bound(np.array([.5]),np.array([.4]),family,delta)
    def test_three_cell_model_cannot_mislabel_second_as_last(self):
        model=BenchmarkModel((4,),(2,),3,ModelConfig(depth=3,width=4))
        x=np.ones((2,3,4),np.float32)
        with self.assertRaisesRegex(ValueError,'aliases'):base_capture(model,x)
    def test_mean_geometry_handles_one_coordinate_20_steps_and_analog_cell(self):
        from dataclasses import replace,asdict
        from bridge35 import load_model
        from torch_adapter import LoadModel
        from run_mean_geometry import run
        _,cfg,_=load_model(ROOT/'evidence/source_smoke_run')
        cfg=replace(cfg,data=replace(cfg.data,k=1,blank_delay=7),cell='continuous',width=8)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);torch.manual_seed(3659);model=LoadModel(cfg)
            torch.save({'config':asdict(cfg),'model_state':model.state_dict()},path/'best.pt')
            (path/'run.json').write_text(json.dumps({'config':asdict(cfg),'optimizer_updates':0}))
            result=run(path,n=2,repeats=2)
            self.assertEqual(result['steps'],20);self.assertEqual(result['visible_coordinates_per_source'],1)
            self.assertEqual(result['forward_captures'],8)
            self.assertTrue(all(g['event_crossing_fraction_by_time'] is None for g in result['finite_scale_groups']))

if __name__=='__main__':unittest.main()
