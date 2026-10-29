"""CPU acceptance tests of scientific contracts, not paper benchmark results."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from snnbench.specs import *
from snnbench.data import generate,DataStore,source_pair
from snnbench.models import create_model
from snnbench.runtime import collect,replay_suffix,intervene,message_edit,tensor_input
from snnbench.engine import native_losses,train,load_run
from snnbench.analysis import analyze,collect_store,interval
from snnbench.utils import parameter_hash,file_hash

torch.set_num_threads(2)


def contract_batch(task,n=3):
    b=generate(task,n);b['meta'].update(steps=b['x'].shape[1],input_shape=list(b['x'].shape[2:]));return b


class DataContracts(unittest.TestCase):
    def test_all_original_native_targets(self):
        for name in VECTOR_TASKS+IMAGE_TASKS:
            for objective in ('classification','reconstruction' if name in IMAGE_TASKS else 'regression'):
                with self.subTest(name=name,objective=objective):
                    t=TaskSpec(family='legacy',name=name,objective=objective);b=generate(t,4)
                    self.assertEqual(len(b['x']),4);self.assertTrue(np.isfinite(b['x']).all());self.assertEqual(b['meta']['model_fields'],['x'])
                    if objective=='classification':self.assertTrue(np.issubdtype(b['y'].dtype,np.integer));self.assertLess(b['y'].max(),b['meta']['classes'])

    def test_all_protocols_and_prefix_invariance(self):
        for j in range(1,10):
            for obj in (('classification','reconstruction') if j in (7,8) else ('classification','regression')):
                t=TaskSpec(family='protocol',name=f'G{j:02d}',T=6 if j in (7,8) else 32,objective=obj)
                with self.subTest(name=t.name,objective=obj):
                    a=generate(t,5);b=generate(t,2,sample_ids=[3,4]);self.assertTrue(np.array_equal(a['x'][3:],b['x']));self.assertTrue(np.array_equal(a['y'][3:],b['y']))
                    self.assertEqual(a['targets'].shape[1],sum(a['meta']['source_dims']))

    def test_tagged_role_load_and_shared_world(self):
        t=TaskSpec(required_history=4,k=2);a=generate(t,8);b=generate(replace(t,k=8),8)
        self.assertTrue(np.array_equal(a['z'],b['z']));self.assertTrue(np.array_equal(a['noise'],b['noise']))
        self.assertTrue(np.all(a['loss_mask'].sum(1)==10));self.assertTrue(np.all(a['x'][:,-3:]==0))
        self.assertEqual(a['meta']['requested_sources'],['H1','H2','H3','H4','C'])
        pair=source_pair(t,a,'H3');self.assertTrue(np.array_equal(a['noise'],pair['noise']));self.assertTrue(np.array_equal(a['z'][:,:2],pair['z'][:,:2]));self.assertFalse(np.array_equal(a['z'][:,2],pair['z'][:,2]))
        with self.assertRaises(ValueError):generate(replace(t,T=32),2)

    def test_symbolic_native_masks(self):
        for family,obj in (('official_adding','classification'),('copy_memory','sequence_classification'),('mqar','classification')):
            t=TaskSpec(family=family,name=family,objective=obj,T=32);b=contract_batch(t,10);s=RunSpec(task=t,model=ModelSpec(width=4,encoder_width=8));m=create_model(s,b['meta']);p=m(torch.from_numpy(b['x']));loss,_=native_losses(p,b);loss.mean().backward()
            self.assertTrue(torch.isfinite(loss).all())
            if family=='official_adding':self.assertEqual(sorted(b['y'].tolist()),list(range(10)))
            if family=='copy_memory':self.assertTrue(np.all(b['loss_mask'].sum(1)==t.k))

    def test_g01_exact_age_and_g02_invalid(self):
        t=TaskSpec(family='protocol',name='G01',T=64,params={'items':4,'age':16,'writes':4});b=generate(t,5)
        self.assertTrue(np.all(b['source_times'][:,0]==47));self.assertTrue(np.array_equal(b['y'],b['targets'][:,:4]))
        t=replace(t,name='G02',objective='classification',params={'items':4,'age':16,'versions':2,'invalidate_probability':1.});b=generate(t,5);self.assertTrue(np.all(b['y']==2))

    def test_g04_not_an_order_only_label(self):
        t=TaskSpec(family='protocol',name='G04',T=32,objective='classification',params={'physical_time':True,'show_clock':True})
        b=generate(t,128);self.assertTrue(np.array_equal(b['y'],(b['targets'][:,0]>.25).astype(int)))
        self.assertEqual(set(b['y'].tolist()),{0,1})
        with self.assertRaises(ValueError):generate(replace(t,params={'physical_time':True,'show_clock':False}),3)

    def test_g09_acquisition_clock(self):
        t=TaskSpec(family='protocol',name='G09',T=16,params={'measurements':4,'acquisition':'delayed_linear'});b=generate(t,3)
        self.assertTrue(np.all(b['availability_times']>=b['timestamps']-1e-6));self.assertFalse(b['meta']['online_causal_at_nominal_grid'])
        c=generate(replace(t,params={'measurements':4,'acquisition':'causal_hold'}),3);self.assertTrue(np.array_equal(b['y'],c['y']))

    def test_store_chunk_stability_and_readonly(self):
        with tempfile.TemporaryDirectory() as root:
            t=TaskSpec();s=DataStore(t,root);b=s.ensure('train',67);a=generate(t,67)
            self.assertTrue(np.array_equal(a['x'],b['x']));self.assertFalse(b['x'].flags.writeable)
            v=s.ensure('val',8);self.assertFalse(set(v['group_ids'])&set(b['group_ids']))


class ModelContracts(unittest.TestCase):
    def test_all_original_native_loss_paths(self):
        for name in VECTOR_TASKS+IMAGE_TASKS:
            image=name in IMAGE_TASKS
            for objective in ('classification','reconstruction' if image else 'regression'):
                t=TaskSpec(family='legacy',name=name,objective=objective,image_size=8);b=contract_batch(t,2)
                m=create_model(RunSpec(task=t,model=ModelSpec(backbone='cnn2d' if image else 'mlp',width=4,encoder_width=8)),b['meta'])
                p=m(torch.from_numpy(b['x']));loss,_=native_losses(p,b);loss.mean().backward();self.assertTrue(torch.isfinite(loss).all())

    def test_multi_query_mqar_loss(self):
        t=TaskSpec(family='mqar',name='multi_query_reference',objective='sequence_classification',T=32,params={'queries':4,'vocab_size':16})
        b=contract_batch(t,3);self.assertTrue(np.all(b['loss_mask'].sum(1)==4));m=create_model(RunSpec(task=t,model=ModelSpec(width=4,encoder_width=8)),b['meta']);loss,_=native_losses(m(torch.from_numpy(b['x'])),b);loss.mean().backward()

    def test_credit_detach_preserves_forward_and_changes_gradient(self):
        s=RunSpec(model=ModelSpec(neuron='continuous',width=4,encoder_width=8));b=contract_batch(s.task,3)
        models=[create_model(replace(s,model=replace(s.model,gradient_mode=mode)),b['meta']) for mode in ('full','detach','notd')]
        predictions=[];grad=[]
        for m in models:
            p=m(torch.from_numpy(b['x']));predictions.append(p.detach());native_losses(p,b)[0].mean().backward();grad.append(m.encoder[0].weight.grad)
        self.assertTrue(torch.equal(predictions[0],predictions[1]));self.assertFalse(torch.equal(grad[0],grad[1]));self.assertFalse(torch.equal(predictions[0],predictions[2]))

    def test_96_backbone_cell_pairs_backward_and_replay(self):
        for bb in REFERENCE_BACKBONES:
            image=bb in ('cnn2d','sew_resnet18','spike_attention','unet')
            t=TaskSpec(family='spatial',name='spatial_v2',objective='classification',T=6,image_size=8) if image else TaskSpec()
            b=contract_batch(t,2)
            for cell in CELLS:
                with self.subTest(backbone=bb,cell=cell):
                    s=RunSpec(task=t,model=ModelSpec(backbone=bb,neuron=cell,width=4,encoder_width=8,observation='output'));m=create_model(s,b['meta'])
                    p=m(torch.from_numpy(b['x']));loss,_=native_losses(p,b);loss.mean().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                    capture=collect(m,b,snapshots=True);replay=replay_suffix(m,b,capture['snapshots'][2]);self.assertTrue(np.array_equal(replay,capture['prediction']))
                    thin=collect_store(m,b,2,4,'cpu');self.assertTrue(np.array_equal(thin['prediction'],capture['prediction']))

    def test_continuous_baselines_and_native_spatial_reconstruction(self):
        for bb in BASELINES+('cnn2d','unet'):
            image=bb in ('convgru','cnn2d','unet');t=TaskSpec(family='spatial',name='spatial_v2',T=6,image_size=8,objective='reconstruction') if image else TaskSpec()
            b=contract_batch(t,2);s=RunSpec(task=t,model=ModelSpec(backbone=bb,width=4,encoder_width=8,observation='output'));m=create_model(s,b['meta']);p=m(torch.from_numpy(b['x']));loss,_=native_losses(p,b);loss.mean().backward()
            self.assertTrue(np.array_equal(collect(m,b)['prediction'],collect_store(m,b,2,4,'cpu')['prediction']))

    def test_phase_and_noop_hooks(self):
        s=RunSpec(model=ModelSpec(width=4,encoder_width=8,neuron='continuous',observation='output'));b=contract_batch(s.task,4);m=create_model(s,b['meta']);a=collect(m,b)['prediction'];last=list(m.temporal_owners())[-1]
        z=intervene(m,b,last,15,edit_state=True);self.assertTrue(np.array_equal(a,z))
        z=intervene(m,b,last,15,edit_state=False,edit_message=True);self.assertFalse(np.array_equal(a,z))
        with message_edit(m,last,4,lambda x:x):
            with torch.no_grad():z=m(tensor_input(b,'cpu')).numpy()
        self.assertTrue(np.array_equal(a,z))

    def test_initialization_pairing(self):
        s=RunSpec(model=ModelSpec(width=4,encoder_width=8));b=contract_batch(s.task);a=create_model(s,b['meta']);other=replace(s,model=replace(s.model,neuron='alif_ref'));c=create_model(other,b['meta'])
        for name,p in a.named_parameters():
            if name in dict(c.named_parameters()):self.assertTrue(torch.equal(p,dict(c.named_parameters())[name]),name)


class ExecutionContracts(unittest.TestCase):
    def tiny(self,objective='regression'):
        return RunSpec(task=TaskSpec(objective=objective),model=ModelSpec(width=4,encoder_width=8),
            train=TrainSpec(epochs=2,train_scenes=32,val_scenes=16,batch_size=8,save_every_updates=2),
            analysis=AnalysisSpec(fit_scenes=24,select_scenes=16,eval_scenes=16,batch_size=8,max_features=8,interval_resamples=20,mlp_steps=3,anova_scenes=8))

    def test_exact_mid_epoch_and_boundary_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=self.tiny();full=train(s,root/'full',root/'data');_,m,ck,_=load_run(full,checkpoint='last.pt');h=parameter_hash(m)
            for cut in (3,4):
                paused=train(s,root/('resume'+str(cut)),root/'data',max_updates=cut);resumed=train(s,root/('resume'+str(cut)),root/'data')
                _,other,oc,_=load_run(resumed,checkpoint='last.pt');self.assertEqual(h,parameter_hash(other));self.assertEqual(ck['progress']['history'],oc['progress']['history'])

    def test_train_analysis_full_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=self.tiny();run=train(s,root/'runs',root/'data');out=analyze(run,root/'data');r=json.loads((out/'report.json').read_text())
            self.assertEqual(set(r['modules']),set(s.analysis.modules));self.assertEqual(r['modules']['recovery']['status'],'computed');self.assertTrue(r['modules']['selective']['rows'])
            self.assertFalse((DataStore(s.task,root/'data').root/'audit').exists());self.assertGreater(r['resources']['parameters'],0)
            status=json.loads((run/'status.json').read_text());self.assertTrue(status['parameters_changed']);self.assertEqual(status['updates'],8)

    def test_group_bootstrap(self):
        x=np.arange(6.);a=interval(x,20,.05,groups=np.array(['one']*6));self.assertIsNone(a['interval'])
        b=interval(x,20,.05,groups=np.array(['one']*3+['two']*3));self.assertEqual(b['independent_groups'],2)

    def test_all_migrated_kernels_import(self):
        for p in (Path(__file__).resolve().parents[1]/'snnbench/kernels').glob('*.py'):
            if p.stem!='__init__':importlib.import_module('snnbench.kernels.'+p.stem)


if __name__=='__main__':unittest.main()
