"""Complete state retention, actual queue work and recoverable training evidence."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from snnbench.archive import TensorArchive,ArchiveReader,capture_batch,archive_run
from snnbench.campaign import campaign_plan,stage_postprocessing,post_worker,_claim_post
from snnbench.coverage import plan_inventory
from snnbench.data import generate
from snnbench.specs import *
from snnbench.models import create_model
from snnbench.runtime import replay_suffix
from snnbench.engine import train
from snnbench.queue import connect
from snnbench.telemetry import UpdateJournal

torch.set_num_threads(2)


class CompleteArchive(unittest.TestCase):
    def test_original_bits_binary_signed_zero_bfloat_and_nested_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'values.zip';w=TensorArchive(p)
            binary=np.tile(np.array([0.,1.],dtype='float32'),64);signed=np.array([-0.,0.,1.],dtype='float32');bf=torch.tensor([.125,1.5],dtype=torch.bfloat16)
            w.tree('tree',{'binary':binary,'signed':signed,'bf':bf,'tuple':(None,torch.tensor(3)),'empty':np.empty((3,0)),'infinite':float('inf')});w.tree('again',binary);index=w.close()
            self.assertEqual(len(index['arrays']),5);self.assertTrue(any(v['encoding']=='packbits-little' for v in index['arrays'].values()))
            r=ArchiveReader(p);tree=r.tree('tree',tensors=True)
            self.assertEqual(tree['binary'].tobytes(),binary.tobytes());self.assertEqual(tree['signed'].tobytes(),signed.tobytes());self.assertTrue(torch.equal(tree['bf'],bf));self.assertEqual(tree['tuple'][1].ndim,0);r.close()
            self.assertEqual(tree['empty'].shape,(3,0));self.assertEqual(tree['infinite'],float('inf'))

    def test_full_coordinates_and_complete_replay_for_multiple_state_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            for backbone in ('mlp','transformer','mamba_ref','conv_lif_rnn'):
                task=TaskSpec(family='spatial',name='spatial_v2',T=6,image_size=8,objective='reconstruction') if backbone=='conv_lif_rnn' else TaskSpec()
                batch=generate(task,3);batch['meta'].update(input_shape=list(batch['x'].shape[2:]),steps=task.T)
                width=160 if backbone=='mlp' else 4;s=RunSpec(task=task,model=ModelSpec(backbone=backbone,width=width,encoder_width=8,observation='output'))
                model=create_model(s,batch['meta']);p=Path(tmp)/(backbone+'.zip');info=capture_batch(model,batch,p);r=ArchiveReader(p)
                snapshot=r.tree('step/00000003/snapshot',tensors=True);prediction=r.tree('prediction');self.assertTrue(np.array_equal(prediction,replay_suffix(model,batch,snapshot)))
                if backbone=='mlp':self.assertEqual(r.tree('step/00000000/owner/backbone.cells.0/last')['pre'].shape[-1],160)
                if backbone=='transformer':
                    self.assertEqual(snapshot['owners']['backbone']['l0_keys'].shape[2],4)
                    self.assertTrue(any('prefix_base' in v for v in r.index['arrays'].values()))
                self.assertTrue(info['complete_coordinates']);self.assertTrue(any('/leaf/' in key for key in r.index['trees']));r.close()

    def test_archive_failure_keeps_partial_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=TaskSpec();batch=generate(task,2);batch['meta'].update(input_shape=list(batch['x'].shape[2:]),steps=task.T)
            model=create_model(RunSpec(),batch['meta']);batch['y']=np.full_like(batch['y'],np.nan);p=Path(tmp)/'failed.zip'
            with self.assertRaises(FloatingPointError):capture_batch(model,batch,p)
            self.assertFalse(p.exists());failed=list(Path(tmp).glob('*.failed-*.zip'));self.assertEqual(len(failed),1)
            r=ArchiveReader(failed[0]);self.assertEqual(r.tree('capture_exception')['type'],'FloatingPointError');r.close()


class CampaignExecution(unittest.TestCase):
    def tiny(self):return RunSpec(model=ModelSpec(width=4,encoder_width=8),train=TrainSpec(epochs=1,train_scenes=16,val_scenes=8,batch_size=4,save_every_updates=1),analysis=AnalysisSpec(fit_scenes=8,select_scenes=4,eval_scenes=8,batch_size=4,max_features=8,mlp_steps=1,interval_resamples=5,anova_scenes=4))

    def test_training_journal_and_complete_archive_sealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=self.tiny();run=train(s,root/'runs',root/'data')
            rows=[json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()];self.assertEqual([r['update'] for r in rows],[1,2,3,4]);self.assertEqual(len(rows[0]['native_losses']),4);self.assertTrue(rows[0]['gradient_statistics_after_clip'])
            out=archive_run(run,root/'data',roles=('development',),scenes=3,batch_size=2);r=json.loads((out/'receipt.json').read_text());self.assertEqual(len(r['shards']),2)
            self.assertEqual(out,archive_run(run,root/'data',roles=('development',),scenes=3,batch_size=2));self.assertTrue(r['complete_coordinates'])

    def test_failed_training_keeps_update_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=replace(self.tiny(),model=ModelSpec(neuron='continuous',gain=1e30,width=4,encoder_width=8))
            with self.assertRaises(FloatingPointError):train(s,root/'runs',root/'data')
            run=root/'runs'/s.run_id;r=json.loads((run/'status.json').read_text());self.assertEqual(r['status'],'failed')
            evidence=torch.load(run/r['failure_evidence'],weights_only=True);self.assertEqual(evidence['batch']['x'].shape[0],4)

    def test_uncommitted_update_rows_are_preserved_before_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);j=UpdateJournal(root,0,'r');j.append({'update':1});j.append({'update':2});j=UpdateJournal(root,1,'r');j.append({'update':2})
            self.assertEqual(len((root/'updates.jsonl').read_text().splitlines()),2);self.assertEqual(len(list(root.glob('updates_uncommitted_*'))),1)

    def test_original_six_layer_inventory_and_materialized_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=campaign_plan(root,seeds=(11,),profiles=('smoke',),materialize=True)
            self.assertEqual(plan['training_queue']['pending'],8);inventory=json.loads((root/'all_layer_coverage.json').read_text());ids={r['source_module_id'] for r in inventory['master_cards']}
            self.assertEqual(inventory['counts']['master_cards'],40);self.assertIn('L5.4',ids);self.assertIn('L6.5',ids);self.assertFalse(inventory['all_planned_analysis_complete'])

    def test_post_queue_executes_actual_analysis_and_serializes_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'external_data_cache';campaign_plan(root,seeds=(11,),profiles=('smoke',),data_root=cache);s=self.tiny();run=train(s,root/'runs',cache);r=stage_postprocessing(root,checkpoints='terminal',archive_scenes=2,archive_batch=2)
            self.assertEqual(r['new_jobs'],4)
            with connect(r['database']) as con:
                first=_claim_post(con,'owner1',300,2);self.assertIsNotNone(first);self.assertIsNone(_claim_post(con,'owner2',300,2));con.execute("UPDATE jobs SET state='pending',owner=NULL,attempts=0");con.commit()
                # Run one genuine analysis subprocess, leaving archive jobs for
                # the separate sealing test rather than duplicating large traces.
                con.execute("UPDATE jobs SET state='done' WHERE spec LIKE '%\"kind\": \"archive\"%'");con.commit()
            status=post_worker(r['database'],'cpu',one=True);self.assertEqual(status.get('failed',0),0)
            self.assertTrue(list(run.glob('analysis/*/receipt.json')))


if __name__=='__main__':unittest.main()
