"""Fixtures check protocol boundaries, not full-corpus benchmark performance."""
import csv
import gzip
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch
import numpy as np
import torch
from scipy.io import savemat
from snnbench.real_data import prepare_mnist,prepare_forecast_csv,prepare_wisdm,prepare_openbmi,prepare_esc50
from snnbench.prepare import verify_dataset,TRAIN_ROLES,TEST_ROLES
from snnbench.catalogue import model_bank,task_bank
from snnbench.specs import *
from snnbench.models import create_model
from snnbench.engine import native_losses,forecast_metrics
from snnbench.runtime import collect,replay_suffix
from snnbench.data import DataStore
from snnbench.utils import file_hash

torch.set_num_threads(2)


class ExpandedProtocols(unittest.TestCase):
    def test_mnist_scan_permutation_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for prefix,n in (('train',16),('t10k',4)):
                for suffix,shape in (('images-idx3-ubyte',(n,28,28)),('labels-idx1-ubyte',(n,))):
                    array=(np.arange(np.prod(shape))%(256 if len(shape)==3 else 10)).astype('uint8').reshape(shape)
                    with gzip.open(root/(prefix+'-'+suffix+'.gz'),'wb') as f:f.write(bytes([0,0,8,len(shape)])+struct.pack('>'+'I'*len(shape),*shape)+array.tobytes())
            for mode in ('sequential','permuted','repeat'):
                out=root/mode;prepare_mnist(root,out,mode,steps=3);verify_dataset(out)
                x=np.load(out/'train/x.npy');self.assertEqual(x.shape[1:],(3,1,28,28) if mode=='repeat' else (784,1))
                c=json.loads((out/'dataset.json').read_text())['contract'];self.assertFalse(c['physical_time'])
                if mode=='permuted':self.assertEqual(sorted(c['permutation']),list(range(784)))
                if mode=='repeat':self.assertTrue(np.array_equal(x[:,0],x[:,2]))

    def test_forecast_no_overlap_train_only_scale_and_native_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'ETT_fixture.csv';n=2000
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['date','a','b'])
                for i in range(n):w.writerow([str(np.datetime64('2020-01-01T00')+np.timedelta64(i,'h')),i,2*i+(10000 if i>=1200 else 0)])
            out=root/'prepared';prepare_forecast_csv(path,out,context=4,horizon=3,stride=2,dataset_id='fixture',embargo=2,block_size=30);verify_dataset(out)
            manifest=json.loads((out/'dataset.json').read_text());c=manifest['contract'];self.assertEqual(c['normalization']['mean'],[599.5,1199.])
            used={}
            for role in TRAIN_ROLES+TEST_ROLES:
                ix=np.load(out/role/'input_indices.npy');iy=np.load(out/role/'target_indices.npy');observed=set(ix.reshape(-1))|set(iy.reshape(-1))
                self.assertTrue(np.all(iy[:,0]>ix[:,-1]));self.assertTrue(np.all(np.load(out/role/'target_timestamps.npy')[:,0]>np.load(out/role/'timestamps.npy')[:,-1]))
                for other,values in used.items():self.assertFalse(observed&values,(role,other))
                used[role]=observed
            t=TaskSpec(family='archive',name='fixture',objective='regression',T=4,archive_root=str(out),archive_manifest_sha256=file_hash(out/'dataset.json'))
            batch=DataStore(t,root/'cache').ensure('train',3)
            for name in ('mlp','gru','transformer','s4d_ref','lru_ref','mamba_ref'):
                s=RunSpec(task=t,model=ModelSpec(backbone=name,observation='output',width=4,encoder_width=8));model=create_model(s,batch['meta']);p=model(torch.from_numpy(np.array(batch['x'])))
                self.assertEqual(p.shape,(3,3,2));native_losses(p,batch)[0].mean().backward()
                cap=collect(model,batch,snapshots=True);self.assertTrue(np.array_equal(cap['prediction'],replay_suffix(model,batch,cap['snapshots'][1])))
            metrics=forecast_metrics(np.array(batch['y'])+1,batch);self.assertTrue(np.allclose(metrics['MSE_standardized'],1))
            self.assertTrue(np.allclose(metrics['MSE_original_units'],np.square(c['normalization']['std']).mean()))
            with self.assertRaisesRegex(ValueError,r'context\+horizon'):prepare_forecast_csv(path,root/'invalid',context=100,horizon=100)
            long=root/'long';prepare_forecast_csv(path,long,context=100,horizon=50,stride=2,role_allocation='minimum_first',embargo=2);verify_dataset(long)
            lc=json.loads((long/'dataset.json').read_text());self.assertEqual(lc['contract']['role_allocation'],'minimum_first')
            self.assertTrue(all(v['samples']>=2 for v in lc['splits'].values()))

    def test_wisdm_subjects_and_activity_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw/watch/gyro';raw.mkdir(parents=True)
            for sid in range(20):
                lines=[]
                for i in range(20):lines.append(f'{1600+sid},{"A" if i<10 else "B"},{i*50000000},{sid},{i},{sid+i};')
                (raw/f'{sid}.txt').write_text('\n'.join(lines))
            out=root/'canonical';prepare_wisdm(root,out,window=6,stride=3);verify_dataset(out);c=json.loads((out/'dataset.json').read_text())['contract']
            self.assertEqual(c['normalization']['fit_role'],'train');self.assertEqual(np.load(out/'train/x.npy').shape[1:],(6,3))
            self.assertEqual(sum(np.load(out/r/'x.npy').shape[0] for r in TRAIN_ROLES+TEST_ROLES),80)

    def test_openbmi_subjects_and_right_boundary_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for sid in range(20):
                savemat(root/f'data_{sid}.mat',{f'data_{sid}':np.full((2,20,3),sid,dtype='float32')});savemat(root/f'label_{sid}.mat',{f'label_{sid}':np.array([1,2])})
            out=root/'canonical';prepare_openbmi(root,out,1000,downsample=5);verify_dataset(out)
            self.assertEqual(np.load(out/'train/x.npy').shape[1:],(4,3));self.assertAlmostEqual(float(np.load(out/'train/timestamps.npy')[0,0]),.005)

    def test_esc50_original_recording_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'audio').mkdir();(root/'meta').mkdir()
            with (root/'meta/esc50.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['filename','fold','target','src_file'])
                for i in range(14):
                    filename=f'{i}.wav';w.writerow([filename,5 if i>=12 else 1,0,str(i)])
                    with wave.open(str(root/'audio'/filename),'wb') as audio:audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(44100);audio.writeframes(np.zeros(5*44100,dtype='<i2').tobytes())
            out=root/'canonical';prepare_esc50(root,out);verify_dataset(out);self.assertEqual(np.load(out/'train/x.npy').shape[1:],(498,64))

    def test_expanded_catalogue_compatible_and_native_output(self):
        tasks=task_bank('expanded');seen=set()
        for t in tasks:
            for m in model_bank(t,'expanded'):
                RunSpec(task=t,model=m).validate();seen.add(m.backbone);self.assertEqual(m.observation,'output')
        self.assertEqual(seen,set(REFERENCE_BACKBONES+BASELINES));self.assertEqual(len(seen),32)
        names={m.backbone for m in model_bank(TaskSpec(),'full',roster='expanded')}
        self.assertIn('mamba_ref',names);self.assertIn('dilated_spiking_tcn',names)

    def test_specialized_dynamics_and_normalization_survive_initialization(self):
        c={'steps':4,'input_shape':[3],'output_shape':[2],'target_mode':'continuous'}
        for name in ('s4d_ref','lru_ref','mamba_ref','transformer'):
            model=create_model(RunSpec(model=ModelSpec(backbone=name,width=4,observation='output')),c)
            for mod in model.modules():
                if isinstance(mod,(torch.nn.LayerNorm,torch.nn.GroupNorm)):self.assertTrue(torch.equal(mod.weight,torch.ones_like(mod.weight)))
            if name=='s4d_ref':self.assertTrue(torch.allclose(model.backbone.log_a,torch.full((4,),np.log(.5))))
            if name=='mamba_ref':self.assertTrue(torch.equal(model.backbone.D,torch.ones(8)))

    def test_notd_transformer_keeps_exogenous_position_and_replays(self):
        c={'steps':4,'input_shape':[3],'output_shape':[2],'target_mode':'continuous'}
        s=RunSpec(model=ModelSpec(backbone='transformer',width=4,observation='output',gradient_mode='notd'))
        model=create_model(s,c);x=torch.ones(2,4,3);_,sequence=model(x,return_sequence=True)
        self.assertEqual(model.backbone.position_t,3);self.assertEqual(model.backbone.state['l0_keys'].shape[2],1)
        self.assertFalse(torch.equal(sequence[:,0],sequence[:,3]))
        model(x,after_step=lambda m,t:setattr(self,'clock_snapshot',m.snapshot()) if t==1 else None)
        expected=model.prediction().detach().numpy();model.reset();model.restore(self.clock_snapshot)
        for t in range(model.time,4):model.step(x[:,t])
        self.assertTrue(np.array_equal(model.prediction().detach().numpy(),expected))

    def test_author_runner_executes_pinned_source_and_rejects_edits(self):
        from snnbench.upstream import run_suite
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'run.py').write_text('import sys\nprint(sys.argv[1])\n')
            for argv in (['init'],['config','user.name','Fixture'],['config','user.email','fixture@example.invalid'],['add','run.py'],['commit','-m','fixture']):subprocess.run(['git','-C',str(source),*argv],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
            config=root/'plan.json';data=root/'data.json';data.write_text('{"fixture":true}')
            config.write_text(json.dumps({'argv':['run.py','native-ok'],'input_files':[str(data)]}))
            with patch('snnbench.upstream.suites',return_value={'fixture':{'commit':commit,'repository':'local fixture','entrypoints':[['run.py']]}}):
                receipt=run_suite('fixture',source,config,root/'receipt');self.assertEqual(receipt['status'],'completed');self.assertFalse(receipt['official_reproduction_verified'])
                self.assertEqual((root/'receipt/native.log').read_text().strip(),'native-ok')
                (source/'run.py').write_text('print("changed")\n')
                with self.assertRaisesRegex(ValueError,'Tracked author'):run_suite('fixture',source,config,root/'bad')

    def test_optional_author_archives_do_not_download_by_default(self):
        from snnbench.download import main
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'registry.json';path.write_text(json.dumps({'datasets':[{'id':'D1','name':'fixture','priority':'P0','files':[{'filename':'raw','url':'fixture://raw'},{'filename':'pooled','url':'fixture://pooled','optional':True}]}]}))
            with patch('snnbench.download.download_file',return_value={'sha256':'fixture'}) as download:
                main(['--manifest',str(path),'--out',tmp]);self.assertEqual(download.call_count,1)
                download.reset_mock();main(['--manifest',str(path),'--out',tmp,'--include-native']);self.assertEqual(download.call_count,2)


if __name__=='__main__':unittest.main()
