"""Artificial format fixtures, not validation of complete official corpora."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from snnbench.prepare import prepare_har,prepare_spikes,prepare_gsc,prepare_gesture,verify_dataset
from snnbench.oracles import task_reference
from snnbench.specs import TaskSpec
from snnbench.data import generate


class RawAdapters(unittest.TestCase):
    def test_har_raw_inertial_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw';out=Path(tmp)/'canonical'
            signals=['body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z']
            for split,count,start in (('train',12,1),('test',4,101)):
                folder=root/split;inertial=folder/'Inertial Signals';inertial.mkdir(parents=True);subjects=np.repeat(np.arange(start,start+count),3);n=len(subjects)
                np.savetxt(folder/f'subject_{split}.txt',subjects,fmt='%d');np.savetxt(folder/f'y_{split}.txt',1+np.arange(n)%6,fmt='%d')
                for channel in signals:np.savetxt(inertial/f'{channel}_{split}.txt',np.ones((n,128)),fmt='%.1f')
            r=prepare_har(root,out);verify_dataset(out);x=np.load(out/'train/x.npy');self.assertEqual(x.shape[1:],(128,9));self.assertEqual(sum(r['samples'].values()),48)

    @unittest.skipUnless(importlib.util.find_spec('h5py'),'optional h5py not installed')
    def test_shd_ssc_hdf5_fixture(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files={}
            for split,n in (('train',64),('val',16),('test',16)):
                path=root/(split+'.h5');files[split]=str(path)
                with h5py.File(path,'w') as f:
                    f['labels']=np.arange(n)%20;group=f.create_group('spikes');times=group.create_dataset('times',(n,),dtype=h5py.vlen_dtype(np.float64));units=group.create_dataset('units',(n,),dtype=h5py.vlen_dtype(np.int64))
                    for i in range(n):times[i]=[0.,.1,1.3];units[i]=[0,1,699]
            r=prepare_spikes(files,root/'canonical',T=5,window=1.4);verify_dataset(root/'canonical')
            self.assertEqual(sum(r['samples'].values()),96);self.assertEqual(np.load(root/'canonical/train/x.npy').shape[1:],(5,700))

    @unittest.skipUnless(importlib.util.find_spec('h5py'),'optional h5py not installed')
    def test_earthnet_author_hdf5_fixture(self):
        import h5py
        from snnbench.earthnet import prepare_earthnet
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,count,start in (('train',12,0),('test',4,100)):
                with h5py.File(root/(split+'.h5'),'w') as f:
                    for tile in range(start,start+count):
                        for window in range(2):
                            g=f.create_group(f'{split}/tile{tile:03d}/cube_2017-01-01_2017-12-31_{window}')
                            g['highresdynamic']=np.full((8,8,7,6),.5,'float32');g['idx_good_frames']=np.arange(6)
            out=root/'canonical';r=prepare_earthnet(root/'train.h5',root/'test.h5',out,T=6,size=8);verify_dataset(out);self.assertEqual(sum(r['samples'].values()),32)
            self.assertEqual(np.load(out/'train/x.npy').shape[1:],(6,5,8,8));self.assertEqual(np.load(out/'train/y.npy').shape[1:],(4,8,8))

    def test_tonic_gesture_npy_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,users in (('ibmGestureTrain',range(1,13)),('ibmGestureTest',range(101,105))):
                for user in users:
                    folder=root/split/(f'user{user}_fixture');folder.mkdir(parents=True)
                    for label in (0,1):np.save(folder/(str(label)+'.npy'),np.array([[0,0,0,0],[127,127,1,100],[64,64,0,300]],dtype='int64'))
            r=prepare_gesture(root,root/'canonical',T=5,window=1.,size=8);verify_dataset(root/'canonical');self.assertEqual(sum(r['samples'].values()),32)
            self.assertEqual(np.load(root/'canonical/train/x.npy').shape[1:],(5,2,8,8))

    def test_gsc_wav_and_speaker_lists_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw';root.mkdir();val=[];test=[]
            for j in range(35):(root/f'word{j:02d}').mkdir()
            for speaker in range(18):
                for take in range(2):
                    path=root/'word00'/f'speaker{speaker}_nohash_{take}.wav';rel=path.relative_to(root).as_posix()
                    if speaker in (14,15):val.append(rel)
                    if speaker in (16,17):test.append(rel)
                    with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(np.zeros(800,dtype='<i2').tobytes())
            (root/'validation_list.txt').write_text('\n'.join(val));(root/'testing_list.txt').write_text('\n'.join(test))
            out=Path(tmp)/'canonical';r=prepare_gsc(root,out);verify_dataset(out);self.assertEqual(sum(r['samples'].values()),36);self.assertEqual(np.load(out/'train/x.npy').shape[1:],(98,40))


class MathematicalControls(unittest.TestCase):
    def test_observation_floors(self):
        t=TaskSpec();ref=task_reference(t);self.assertAlmostEqual(ref['coordinate_MSE'],.04/1.04)
        t=TaskSpec(family='protocol',name='G05',T=32,params={'measurements':8,'acquisition':'repeat'});a=task_reference(t)
        b=task_reference(TaskSpec(family='protocol',name='G05',T=32,params={'measurements':8,'acquisition':'independent'}));self.assertGreater(a['coordinate_MSE'],b['coordinate_MSE'])
        same=task_reference(TaskSpec(family='protocol',name='G06',T=32,sigma=0,params={'target_frequency':2.,'distractor_frequency':2.}));distinct=task_reference(TaskSpec(family='protocol',name='G06',T=32,sigma=0,params={'target_frequency':2.,'distractor_frequency':5.}));self.assertGreater(same['continuous_target_MSE'],0);self.assertEqual(distinct['continuous_target_MSE'],0)

    def test_demand_only_input_is_fixed(self):
        a=generate(TaskSpec(content_mode='demand_only',k=2),4);b=generate(TaskSpec(content_mode='demand_only',k=8),4)
        self.assertTrue(np.array_equal(a['x'],b['x']));self.assertFalse(np.array_equal(a['loss_mask'],b['loss_mask']))


if __name__=='__main__':unittest.main()
