"""Actual catalogue design and dependency compilation; no GPU result claims."""
from collections import Counter
from contextlib import closing
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compile_stage1 as stage
from test_efficient_campaign import archives_at, tiny_campaign
from snnbench.catalogue import task_bank, model_bank
from snnbench.specs import RunSpec


def write_master(root):
    ids=sorted({c for _,cards in stage.GROUP_INFO.values() for c in cards})
    text='# Original fixture master\n'+''.join(
        '## '+c+' Question\n**执行逻辑。** Original controls.\n' for c in ids)
    for c in ('X.PAPER','X.STATS','X.COMP','X.CODE'):
        text+='## '+c+' Governance\n**执行逻辑。** Evidence.\n'
    (root/'SNN_Research_Master_Index_2026-10-24.md').write_text(text)
    (root/'SNN_L4_3_Deep_Review.md').write_text('## 8. 4.3.8：Frozen design\n')


def old_membership(row, archives):
    spec=RunSpec.from_dict(row['spec'])
    task_ids={stage.ec.task_key(t) for t in task_bank('full')}
    if spec.task.family!='archive' and stage.ec.task_key(spec.task) not in task_ids:return False
    if spec.train.lr!=.001:return False
    manifest=next((a for t,a in archives if t.identity()==spec.task.identity()),None)
    wanted=stage.ec.canonical(spec.to_dict()['model'])
    return any(stage.ec.canonical(stage.ec.make_spec(spec.task,m,11,manifest).to_dict()['model'])==wanted
        for profile in ('full','paired','learning') for m in stage.ec.models_for(spec.task,manifest,profile))


class Stage1Design(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.archives=archives_at(cls.root)
        cls.rows,cls.groups,cls.counts=stage.design(cls.archives)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_exact_count_and_six_layers(self):
        self.assertEqual(len(self.rows),492)
        cards={c for g in self.groups.values() for c in g['cards']}
        self.assertEqual(len(cards),36)
        self.assertEqual({c[:2] for c in cards},{'L1','L2','L3','L4','L5','L6'})
        self.assertTrue({'L3.'+str(i) for i in range(1,8)}<=cards)

    def test_three_seeds_complete_budget_and_explicit_physical_width(self):
        independent={}
        for row in self.rows.values():
            s=RunSpec.from_dict(row['spec']);s.validate()
            self.assertEqual(s.train.epochs,60);self.assertEqual(s.train.batch_size,128)
            self.assertEqual(s.train.train_scenes,6144);self.assertEqual(s.train.save_every_updates,100)
            d=s.to_dict();seed=d['train'].pop('seed');d['train'].pop('order_seed')
            independent.setdefault(stage.ec.canonical(d),set()).add(seed)
        self.assertTrue(all(seeds=={11,22,33} for seeds in independent.values()))
        widths={self.rows[r]['spec']['model']['width'] for r in self.groups['capacity']['run_ids']}
        self.assertEqual(widths,{16,32})

    def test_classification_reconstruction_real_and_forward_factorial(self):
        specs=[RunSpec.from_dict(r['spec']) for r in self.rows.values()]
        self.assertTrue({'classification','regression','sequence_classification','reconstruction'}<={s.task.objective for s in specs})
        self.assertEqual({Path(s.task.archive_root).name for s in specs if s.task.family=='archive'},{'shd','har'})
        cells={self.rows[r]['spec']['model']['neuron'] for r in self.groups['mechanisms']['run_ids']}
        self.assertTrue({'continuous','threshold','analog_hard_ref','lif_hard','alif_ref','two_state_ref'}<=cells)

    def test_learning_has_matched_gain_forward_and_declared_rule_changes(self):
        rows=[RunSpec.from_dict(self.rows[r]['spec']) for r in self.groups['credit']['run_ids']]
        self.assertEqual({s.model.gradient_mode for s in rows},{'full','detach','notd'})
        self.assertEqual({s.model.gain for s in rows},{.5})
        self.assertEqual({s.model.observation for s in rows},{'output'})
        rows=[RunSpec.from_dict(self.rows[r]['spec']) for r in self.groups['surrogate']['run_ids']]
        self.assertEqual({(s.model.surrogate,s.model.detach_reset) for s in rows},
                         {(s,d) for s in ('atan','triangle') for d in (False,True)})

    def test_expected_analysis_jobs_and_formation_preserve_roles(self):
        jobs=[j for r in self.rows.values() for j in stage.checkpoint_recipes(RunSpec.from_dict(r['spec']),r)]
        self.assertEqual(len(jobs),732)
        self.assertEqual(sum(j['purpose']=='same_model_formation' for j in jobs),240)
        self.assertEqual({j['analysis']['eval_split'] for j in jobs},{'development'})
        self.assertEqual({j['analysis']['eval_scenes'] for j in jobs},{512})

    def test_requirement_omission_fails_instead_of_dropping(self):
        inv={'master_cards':[{'id':'master:L6.99','source_module_id':'L6.99'}],
             'specialized_subsections':[]}
        with self.assertRaisesRegex(ValueError,'Unrouted'):
            stage.requirement_dependencies(inv,self.groups)

    def test_missing_real_archive_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'shd and har'):
            stage.design([(t,a) for t,a in self.archives if Path(t.archive_root).name!='har'])


class CompileAndMigration(unittest.TestCase):
    def fixture(self,root):
        source=root/'original';source.mkdir();code=root/'code';code.mkdir();write_master(code)
        adir=root/'archives';adir.mkdir();archives=archives_at(adir)
        rows,groups,counts=stage.design(archives)
        # Reconstruct old membership using actual original task/model factories.
        with sqlite3.connect(source/'train.sqlite') as con:
            stage.ec.init_queue(con)
            for key,row in rows.items():
                if old_membership(row,archives):con.execute('INSERT INTO jobs(id,spec) VALUES(?,?)',
                    (key,stage.ec.canonical(row['spec'])))
        stage.ec.atomic(source/'all_layer_coverage.json',{'master_cards':[],'specialized_subsections':[],'counts':{}})
        stage.ec.atomic(source/'campaign.json',{'materialized':True,'profiles':['full','paired','learning'],
            'seeds':[11,22,33,44,55],'archives':[t.archive_root for t,_ in archives],
            'training_output':str(source/'runs'),'data_output':str(source/'data'),
            'training_implementation_sha256':'fixture'})
        return source,code,rows

    def test_compile_keeps_old_readonly_and_routes_every_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,code,rows=self.fixture(root)
            before=stage.ec.sha(source/'train.sqlite')
            with patch('builtins.print'):
                result=stage.compile_plan(source,root/'stage1',code)
            self.assertEqual(before,stage.ec.sha(source/'train.sqlite'))
            self.assertEqual(result['selected_runs'],492)
            self.assertEqual(result['shared_analysis_jobs'],732)
            self.assertEqual(result['master_cards'],40)
            self.assertFalse(result['all_original_analysis_complete'])
            self.assertGreater(result['origin_counts']['new_explicit_contract'],0)
            self.assertGreater(result['origin_counts']['existing_legacy_contract'],0)
            self.assertEqual(result['origin_counts'],{'existing_legacy_contract':429,'new_explicit_contract':63})
            deps=json.loads((root/'stage1/requirement_dependencies.json').read_text())
            self.assertTrue(all(r['first_stage_required'] and not r['complete_section_claim'] for r in deps['requirements']))
            self.assertIn('section:4.3.8',{r['id'] for r in deps['requirements']})
            jobs=[json.loads(l) for l in (root/'stage1/analysis_jobs.jsonl').read_text().splitlines()]
            gated=[j for j in jobs if j['gate']]
            self.assertEqual(len(gated),18)
            budget=json.loads((root/'stage1/runtime_budget.json').read_text())
            self.assertIsNone(budget['full_stage_walltime'])
            stage.ec.check_manifest(root/'stage1',result)

    def test_budget_failure_does_not_modify_source_or_emit_partial_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,code,_=self.fixture(root);before=stage.ec.sha(source/'train.sqlite')
            with self.assertRaisesRegex(ValueError,'no automatic subsampling'):
                stage.compile_plan(source,root/'stage1',code,max_runs=10)
            self.assertFalse((root/'stage1').exists())
            self.assertEqual(before,stage.ec.sha(source/'train.sqlite'))

    def test_missing_original_identity_does_not_become_a_new_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,code,rows=self.fixture(root)
            with sqlite3.connect(source/'train.sqlite') as con:
                key=con.execute('SELECT id FROM jobs LIMIT 1').fetchone()[0]
                con.execute('DELETE FROM jobs WHERE id=?',(key,))
            with self.assertRaisesRegex(ValueError,'Expected original identity is missing'):
                stage.compile_plan(source,root/'stage1',code)
            self.assertFalse((root/'stage1').exists())

    def test_wrong_source_campaign_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'smoke';source.mkdir()
            stage.ec.atomic(source/'campaign.json',{'materialized':True,'profiles':['smoke'],'seeds':[11]})
            with self.assertRaisesRegex(ValueError,'full/paired/learning'):
                stage.compile_plan(source,root/'stage1')
            self.assertFalse((root/'stage1').exists())

    def test_new_contract_absent_from_old_database_stays_pending_on_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';p,rows=tiny_campaign(root)
            old=Path(p['source_root'])/'train.sqlite'
            with sqlite3.connect(old) as con:con.execute('DELETE FROM jobs WHERE id=?',(rows[0].run_id,))
            with patch.object(stage.ec,'source_check'),patch.object(stage.ec,'legacy_processes',return_value=[]),\
                 patch.object(stage.ec,'pause_related_campaigns',return_value=[]),patch('builtins.print'):
                self.assertTrue(stage.ec.activate(root))
            with stage.ec.connect(root/'train.sqlite',True) as con:
                self.assertEqual(con.execute('SELECT state FROM jobs WHERE id=?',(rows[0].run_id,)).fetchone()[0],'pending')

    def test_new_contract_previous_failure_is_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';p,rows=tiny_campaign(root)
            with sqlite3.connect(Path(p['source_root'])/'train.sqlite') as con:
                con.execute('DELETE FROM jobs WHERE id=?',(rows[0].run_id,))
            run=Path(p['training_output'])/rows[0].run_id;run.mkdir()
            stage.ec.atomic(run/'status.json',{'status':'failed','run_id':rows[0].run_id,'error':'retained failure'})
            with patch.object(stage.ec,'source_check'),patch.object(stage.ec,'legacy_processes',return_value=[]),\
                 patch.object(stage.ec,'pause_related_campaigns',return_value=[]),patch('builtins.print'):
                stage.ec.activate(root)
            with stage.ec.connect(root/'train.sqlite',True) as con:
                self.assertEqual(con.execute('SELECT state,error FROM jobs WHERE id=?',(rows[0].run_id,)).fetchone(),
                                 ('failed','retained failure'))


class RelatedCampaignSafety(unittest.TestCase):
    def test_only_related_foreign_supervisor_is_paused_and_own_children_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);own=base/'stage1';other=base/'v05';foreign=base/'unrelated'
            for root in (own,other,foreign):root.mkdir()
            runs=str(base/'original/runs')
            for root,out in ((own,runs),(other,runs),(foreign,str(base/'different/runs'))):
                stage.ec.atomic(root/'campaign.json',{'training_output':out})
            helper=b'/project/scripts/efficient_campaign.py'
            args={
                101:[b'python',helper,b'run',b'--root',str(own).encode()],
                102:[b'python',helper,b'run',b'--root',str(other).encode()],
                103:[b'python',helper,b'run',b'--root',str(foreign).encode()],
                104:[b'python',helper,b'_train',b'--out',runs.encode(),b'--spec',str(own/'queue_specs/a.json').encode()],
                105:[b'python',helper,b'_train',b'--out',runs.encode(),b'--spec',str(other/'queue_specs/b.json').encode()],
                106:[b'python',b'/different/run_campaign.py',b'--out',str(base/'different/runs').encode()],
            }
            original=Path.iterdir
            def listing(path):
                return iter(Path('/proc')/str(pid) for pid in args) if path==Path('/proc') else original(path)
            p={'execution_root':str(own),'source_root':str(base/'original'),'training_output':runs}
            with patch.object(Path,'iterdir',listing),patch.object(stage.ec,'proc_args',side_effect=lambda pid:args[int(pid)]):
                self.assertEqual(stage.ec.pause_related_campaigns(p),[str(other)])
                self.assertEqual(stage.ec.legacy_processes(p),[102,105])
            self.assertTrue((other/'pause.request').is_file())
            self.assertFalse((own/'pause.request').exists())
            self.assertFalse((foreign/'pause.request').exists())


if __name__=='__main__':unittest.main()
