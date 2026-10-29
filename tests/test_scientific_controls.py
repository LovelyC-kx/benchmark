from dataclasses import replace
import unittest
import numpy as np
import torch
from snnbench.specs import *
from snnbench.catalogue import task_bank
from snnbench.data import generate
from snnbench.models import create_model
from snnbench.runtime import collect,replay_suffix
from snnbench.engine import native_losses
from snnbench.analysis import collect_store,realized_current_accounting,diagnostic_targets,source_schema
from snnbench.reference.neurons import Cell
from snnbench.kernels.latent_theory42 import gaussian_risk_chain,gaussian_risk

torch.set_num_threads(2)


class ScientificControls(unittest.TestCase):
    def test_coordinate_demand_is_not_whole_source_demand(self):
        task=TaskSpec(name='tagged_joint',k=2);b=generate(task,5);schema=source_schema(b['meta'])
        roles=[s['role'] if i in s['requested_coordinates'] else 'unrequested' for s in schema for i in range(s['stop']-s['start'])]
        self.assertEqual(roles.count('focal'),2);self.assertEqual(roles.count('cotarget'),2);self.assertEqual(roles.count('unrequested'),36)
        spatial=generate(TaskSpec(family='spatial',name='spatial_v2',T=6,objective='classification'),3)
        self.assertEqual(spatial['meta']['requested_coordinate_indices']['C'],[1])

    def test_symbolic_diagnostics_are_categorical(self):
        task=TaskSpec(family='mqar',name='multi_query_reference',objective='sequence_classification',T=32,params={'queries':4,'vocab_size':16});b=generate(task,5)
        self.assertEqual(b['targets'].shape,(5,64));self.assertEqual(b['meta']['source_dims'],[16]*4)
        self.assertTrue(np.all(b['targets'].reshape(5,4,16).sum(-1)==1));self.assertEqual(b['source_times'].shape,(5,4))
    def test_all_62_screen_task_native_loss_paths(self):
        tasks=task_bank('screen');self.assertEqual(len(tasks),62)
        for t in tasks:
            with self.subTest(family=t.family,name=t.name,objective=t.objective):
                b=generate(t,2);b['meta'].update(steps=t.T,input_shape=list(b['x'].shape[2:]));image=b['x'].ndim==5
                spec=RunSpec(task=t,model=ModelSpec(backbone='cnn2d' if image else 'mlp',width=4,encoder_width=8))
                model=create_model(spec,b['meta']);loss,_=native_losses(model(torch.from_numpy(b['x'])),b);loss.mean().backward();self.assertTrue(torch.isfinite(loss).all())

    def test_readout_state_replay_for_every_readout(self):
        b=generate(TaskSpec(),3);b['meta'].update(steps=16,input_shape=list(b['x'].shape[2:]))
        for readout in ('last','mean','sum','exponential','trajectory'):
            s=RunSpec(model=ModelSpec(backbone='temporal_conv',neuron='alif_ref',width=4,encoder_width=8,readout=readout));m=create_model(s,b['meta']);c=collect(m,b,snapshots=True)
            self.assertTrue(np.array_equal(c['prediction'],replay_suffix(m,b,c['snapshots'][6])),readout)

    def test_event_branches_are_independent(self):
        x=torch.ones(2,3);results={}
        for branch in ('emit','reset','aux'):
            c=Cell(3,kind='alif_ref');c.force_events(**{branch:torch.zeros_like(x)});out=c(x);results[branch]=(out,c.state)
        self.assertTrue(torch.all(results['emit'][0]==0));self.assertTrue(torch.all(results['emit'][1]['v']==0));self.assertTrue(torch.all(results['emit'][1]['a']>0))
        self.assertTrue(torch.all(results['reset'][0]==1));self.assertTrue(torch.all(results['reset'][1]['v']>0))
        self.assertTrue(torch.all(results['aux'][0]==1));self.assertTrue(torch.all(results['aux'][1]['a']==0))

    def test_realized_path_accounting_and_shared_unit_coordinates(self):
        b=generate(TaskSpec(),4);b['meta'].update(steps=16,input_shape=list(b['x'].shape[2:]))
        for kind in CELLS:
            if kind=='tanh':continue
            s=RunSpec(model=ModelSpec(neuron=kind,width=8,encoder_width=8));m=create_model(s,b['meta']);c=collect_store(m,b,2,3,'cpu')
            for row in realized_current_accounting(m,c):self.assertLess(row['max_abs_final_state_residual'],1e-4,kind)
            for site in m.temporal_owners():self.assertEqual(c['observation_contracts'][site+'::post']['selected_indices'],c['observation_contracts'][site+'::effective_current']['selected_indices'])

    def test_target_budget_preserves_native_targets_and_roles(self):
        task=TaskSpec();b=generate(task,4);data={'probe_train':b,'probe_val':generate(task,4,'probe_val')};reduced,c=diagnostic_targets(data,10)
        self.assertTrue(np.array_equal(reduced['probe_train']['y'],b['y']));self.assertEqual(sum(reduced['probe_train']['meta']['source_dims']),10)
        self.assertEqual(c['original_dimension'],40);self.assertEqual(reduced['probe_train']['meta']['requested_sources'],b['meta']['requested_sources'])

    def test_population_nested_risk_and_task_energy_counterexample(self):
        chain=gaussian_risk_chain(np.eye(2),np.eye(2),np.array([[1.,0.]]),np.eye(1),np.eye(1),np.array([[1.,1.]])/np.sqrt(2),np.array([[1.]])/np.sqrt(2))
        self.assertAlmostEqual(sum(chain['pieces'].values()),chain['native_risk']);self.assertAlmostEqual(chain['native_risk'],.5)
        code=np.diag([100.,.01]);target=np.array([[0.,1.]])
        self.assertAlmostEqual(gaussian_risk(np.eye(2),code,target),0.)
        self.assertAlmostEqual(gaussian_risk(np.eye(2),code[:1],target),1.)


if __name__=='__main__':unittest.main()
