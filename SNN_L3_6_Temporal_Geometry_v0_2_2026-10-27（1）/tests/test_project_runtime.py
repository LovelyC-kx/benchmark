import unittest
import numpy as np
import torch
from pathlib import Path
from dataclasses import replace
from bridge35 import load_model,fixed_batch,Cell,NEURONS,BenchmarkModel,ModelConfig
from temporal_capture import base_capture,temporal_views
from capture35 import capture
from temporal_contracts import prefix_invariant
from bridge import _render
from run_temporal_review import target_bank

ROOT=Path(__file__).resolve().parents[1]

class RuntimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.model,cls.cfg,_=load_model(ROOT/'evidence/source_smoke_run')
    def test_future_source_never_changes_past_states(self):
        b=fixed_batch(self.cfg.data,8,'geometry_fit',7680000,0);z=b['latents'].copy();z[:,4]+=5
        x=_render(self.cfg.data,z,b['measurement_noise'],b['event_times'],b['source_present'])[:,:16]
        a,c=base_capture(self.model,b['x']),base_capture(self.model,x)
        for key in a:self.assertTrue(prefix_invariant(a[key],c[key],12),key)
    def test_extended_deadline_has_same_prefix(self):
        a=fixed_batch(self.cfg.data,8,'geometry_fit',7680100,0)
        b=fixed_batch(replace(self.cfg.data,blank_delay=11),8,'geometry_fit',7680100,0)
        self.assertTrue(prefix_invariant(a['x'],b['x'],16))
        x,y=base_capture(self.model,a['x']),base_capture(self.model,b['x'])
        for key in x:self.assertTrue(prefix_invariant(x[key],y[key],16),key)
    def test_schedule_pairs_preserve_content_and_noise(self):
        a=fixed_batch(self.cfg.data,8,'geometry_fit',7680200,0)
        for schedule in range(1,6):
            b=fixed_batch(self.cfg.data,8,'geometry_fit',7680200,schedule)
            np.testing.assert_array_equal(a['latents'],b['latents']);np.testing.assert_array_equal(a['measurement_noise'],b['measurement_noise'])
    def test_twelve_cells_have_explicit_temporal_exports(self):
        torch.manual_seed(3607)
        for cell in NEURONS:
            with self.subTest(cell=cell):
                cfg=ModelConfig(backbone='mlp',neuron=cell,width=8,depth=2,normalization='none')
                model=BenchmarkModel((4,),(3,),6,cfg);x=np.random.default_rng(36).normal(size=(4,6,4)).astype(np.float32)
                views=temporal_views(base_capture(model,x))
                self.assertEqual(views['first_post'].shape,(4,6,8))
                self.assertEqual(views['core_post'].shape,(4,6,16))
                if cell in ('continuous','tanh'):
                    self.assertNotIn('first_spike',views)
                else:self.assertIn('first_spike',views)
                if cell in ('alif_ref','adlif_ref'):
                    self.assertIn('first_a',views);self.assertIn('first_post_aux',views)
                if cell=='two_state_ref':self.assertIn('first_syn',views)
    def test_surrogate_gradient_is_not_forward_spike_derivative(self):
        cell=Cell(1,'lif_hard',beta=.8,threshold=1.,surrogate='atan',sg_scale=2.)
        x=torch.tensor([[.2]],requires_grad=True);out=cell(x);out.sum().backward()
        self.assertGreater(float(x.grad),0)
        predictions=[]
        for current in (.19,.21):
            cell.reset();predictions.append(float(cell(torch.tensor([[current]]))))
        self.assertEqual(predictions,[0.,0.])
    def test_target_blocks_follow_visible_dimension(self):
        for q in (1,2,4):
            z=np.random.default_rng(361).normal(size=(20,5*q));y,theta,groups=target_bank(z)
            self.assertEqual(y.shape,(20,5*q+80));self.assertEqual(theta.shape,(5,q,16))
            np.testing.assert_array_equal(y[:,:5*q],z)
            for j,g in enumerate(groups):
                expected=np.where(z[:,q*j:q*j+q]@theta[j]>=0,1.,-1.)
                np.testing.assert_array_equal(y[:,g['binary_columns']],expected)
    def test_invalid_source_block_dimension_rejected(self):
        with self.assertRaises(ValueError):target_bank(np.zeros((10,15)))

if __name__=='__main__':unittest.main()
