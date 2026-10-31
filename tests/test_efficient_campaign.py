"""Planner/control/scheduler tests use actual contracts and fake GPU execution.

No GPU speed, model convergence, or specialized scientific analysis claim.
"""
from collections import Counter
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import patch

from snnbench.catalogue import task_bank
from snnbench.specs import TaskSpec, RunSpec

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/efficient_campaign.py'
spec = importlib.util.spec_from_file_location('efficient_campaign', SCRIPT)
ec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ec)


def archive_items():
    rows = [('esc50_f' + str(i), 'D41', 498, [40]) for i in range(1, 6)]
    rows += [(f'{n}_L96_H{h}', f'{d}_L96_H{h}', 96, [7])
             for n, d in [('ETTh1', 'D96'), ('ETTh2', 'D97')]
             for h in [24, 96, 192, 336, 720]]
    rows += [('gesture', 'D35', 100, [2, 32, 32]), ('gsc35', 'D33', 98, [40]),
             ('har', 'D46', 128, [9]), ('mnist_permuted', 'D89_permuted', 784, [1]),
             ('mnist_repeat', 'D89_repeat', 16, [1, 28, 28]),
             ('mnist_sequential', 'D89_sequential', 784, [1]),
             ('shd', 'SHD', 100, [700]), ('ssc', 'SSC', 100, [700])]
    rows += [(f'wisdm_{d}_{s}', f'D47_{d}_{s}', 200, [3])
             for d in ['phone', 'watch'] for s in ['accel', 'gyro']]
    return rows


def archives_at(root):
    archives = []
    for name, dataset, T, shape in archive_items():
        folder = root / name
        folder.mkdir()
        contract = {'dataset_id': dataset, 'steps': T, 'input_shape': shape,
                    'output_shape': [10], 'target_mode': 'class_index'}
        if dataset.startswith(('D96', 'D97')):
            horizon = int(name.rsplit('H', 1)[1])
            contract.update(objective='regression', target_mode='continuous', output_shape=[horizon, 7])
        manifest = {'fixture_identity': name, 'contract': contract, 'splits': {
            role: {'samples': 10000} for role in ('train', 'val', 'probe_train', 'probe_val',
            'calibration', 'verification_train', 'verification_val', 'development', 'confirmation', 'audit')}}
        ec.atomic(folder / 'dataset.json', manifest)
        task = TaskSpec(family='archive', name=dataset, T=T,
            objective=contract.get('objective', 'classification'), archive_root=str(folder),
            archive_manifest_sha256=ec.sha(folder / 'dataset.json'))
        archives.append((task, manifest))
    return archives


class FullDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.archives = archives_at(cls.root)
        cls.rows = {}
        cls.counts = Counter()

        def add(task, model, seed, manifest, reason, priority):
            s = ec.make_spec(task, model, seed, manifest)
            cls.rows.setdefault(s.run_id, {'spec': s, 'reasons': set()})['reasons'].add(reason)
            cls.counts[reason] += 1

        ec.design(task_bank('expanded'), task_bank('full'), cls.archives, add)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_design_count_and_all_conditions(self):
        self.assertEqual(len(self.rows), 43885)
        self.assertEqual(dict(self.counts), {
            'core': 21705, 'full_condition_axes': 6300,
            'paired_local_factorial': 7776, 'learning_local_factorial': 2304,
            'cross_backbone_boundary': 1992, 'five_seed_sentinels': 3886})
        covered = {ec.task_key(r['spec'].task) for r in self.rows.values()}
        expected = {ec.task_key(t) for t in task_bank('full')} | {ec.task_key(t) for t, _ in self.archives}
        self.assertEqual(covered, expected)
        self.assertEqual(len(covered), 789)

    def test_every_base_model_has_three_seeds_and_sentinels_have_five(self):
        for task, manifest in [(t, None) for t in task_bank('expanded')] + self.archives:
            bank = ec.models_for(task, manifest)
            anchor = 'cnn2d' if any(m.backbone == 'cnn2d' for m in bank) else 'mlp'
            from snnbench.specs import BASELINES
            for model in bank:
                seeds = (11, 22, 33, 44, 55) if model.backbone == anchor or model.backbone in BASELINES else (11, 22, 33)
                for seed in seeds:
                    self.assertIn(ec.make_spec(task, model, seed, manifest).run_id, self.rows)

    def test_training_budget_and_evidence_not_shortened(self):
        for r in self.rows.values():
            s = r['spec']
            self.assertEqual(s.train.epochs, 60)
            self.assertEqual(s.train.batch_size, 128)
            self.assertEqual(s.train.save_every_updates, 100)
            self.assertEqual(len(s.train.checkpoint_fractions), 8)
            self.assertEqual(len(s.analysis.modules), 14)

    def test_full_local_factorials_not_posthoc_substitutes(self):
        paired = [r['spec'] for r in self.rows.values() if 'paired_local_factorial' in r['reasons']]
        learning = [r['spec'] for r in self.rows.values() if 'learning_local_factorial' in r['reasons']]
        self.assertEqual({s.model.observation for s in paired}, {'pre', 'post', 'output'})
        self.assertEqual({s.model.readout for s in paired}, {'last', 'mean', 'exponential'})
        self.assertEqual({s.model.gradient_mode for s in learning}, {'full', 'detach', 'notd'})
        self.assertEqual({s.model.surrogate for s in learning}, {'atan', 'triangle', 'sigmoid', 'rectangle'})
        self.assertEqual({s.model.encoder for s in learning}, {'none', 'auto'})

    def test_plan_matches_legacy_ids_and_keeps_source_readonly(self):
        source = self.root / 'legacy'
        source.mkdir()
        ec.atomic(source / 'campaign.json', {
            'materialized': True, 'profiles': ['full', 'paired', 'learning'],
            'seeds': [11, 22, 33, 44, 55], 'archives': [t.archive_root for t, _ in self.archives],
            'training_output': str(source / 'runs'), 'data_output': str(source / 'data'),
            'training_implementation_sha256': 'fixture'})
        ec.atomic(source / 'all_layer_coverage.json', {'all_planned_analysis_complete': False,
            'master_cards': [{'id': 'master:L' + str(i)} for i in range(1, 7)],
            'specialized_subsections': [{'id': 'section:3.3.2'}]})
        with ec.connect(source / 'train.sqlite') as con:
            ec.init_queue(con)
            con.executemany('INSERT INTO jobs(id,spec) VALUES(?,?)',
                [(run_id, ec.canonical(r['spec'].to_dict())) for run_id, r in self.rows.items()])
        original_hash = ec.sha(source / 'train.sqlite')
        out = self.root / 'optimized'
        with patch('builtins.print'):
            result = ec.plan(source, out)
        self.assertEqual(result['selected_runs'], 43885)
        self.assertEqual(ec.sha(source / 'train.sqlite'), original_hash)
        self.assertEqual((source / 'all_layer_coverage.json').read_bytes(), (out / 'all_layer_coverage.json').read_bytes())
        coverage = json.loads((out / 'training_coverage.json').read_text())
        self.assertEqual(coverage['task_contracts'], 789)
        ec.check_manifest(out, result)
        with ec.connect(out / 'train.sqlite') as con:
            run_id = next(iter(self.rows))
            con.execute('UPDATE jobs SET spec=? WHERE id=?', ('{}', run_id))
        with self.assertRaisesRegex(ValueError, 'Queue scientific'):
            ec.check_manifest(out, result)
        with self.assertRaises(FileExistsError):
            ec.plan(source, out)

    def test_budget_is_an_error_not_silent_subsampling(self):
        source = self.root / 'budget_source'
        source.mkdir()
        ec.atomic(source / 'campaign.json', {'materialized': True, 'profiles': ['full', 'paired', 'learning'],
            'seeds': [11, 22, 33, 44, 55], 'archives': [t.archive_root for t, _ in self.archives]})
        ec.atomic(source / 'all_layer_coverage.json', {})
        out = self.root / 'budget_out'
        with self.assertRaisesRegex(ValueError, 'no requirements silently dropped'):
            ec.plan(source, out, max_runs=10)
        self.assertFalse(out.exists())


def tiny_campaign(root, count=1):
    root.mkdir()
    legacy = root.parent / (root.name + '_legacy')
    legacy.mkdir()
    runs = legacy / 'runs'
    runs.mkdir()
    rows = []
    for i in range(count):
        s = replace(RunSpec(), train=replace(RunSpec().train, seed=11+i, order_seed=1011+i))
        rows.append(s)
    with ec.connect(legacy / 'train.sqlite') as con:
        ec.init_queue(con)
        for s in rows:
            con.execute('INSERT INTO jobs(id,spec) VALUES(?,?)', (s.run_id, ec.canonical(s.to_dict())))
        con.execute('INSERT INTO jobs(id,spec) VALUES(?,?)', ('outside_design', '{}'))
    with ec.connect(root / 'train.sqlite') as con:
        ec.init_queue(con)
        for i, s in enumerate(rows):
            con.execute('INSERT INTO jobs(id,spec) VALUES(?,?)', (s.run_id, ec.canonical(s.to_dict())))
            con.execute('INSERT INTO design VALUES(?,?,?,?,?,?,?)', (s.run_id, 0, i, '[]', 'key', 1, 2*ec.GIB))
    (root / 'manifest.jsonl').write_text(''.join(ec.canonical(s.to_dict()) + '\n' for s in rows))
    p = {'status': 'planned_not_activated', 'source_root': str(legacy), 'training_output': str(runs),
         'data_output': str(legacy / 'data'), 'training_implementation_sha256': 'fixture',
         'manifest_sha256': ec.sha(root / 'manifest.jsonl')}
    ec.atomic(root / 'campaign.json', p)
    return p, rows


class Migration(unittest.TestCase):
    def test_drain_defers_pending_without_touching_running_or_other_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'optimized'
            p, rows = tiny_campaign(root)
            old = Path(p['source_root']) / 'train.sqlite'
            with ec.connect(old) as con:
                con.execute("UPDATE jobs SET state='running',owner=? WHERE id=?",
                    (ec.socket.gethostname() + ':123:x', rows[0].run_id))
            with patch.object(ec, 'source_check'), patch.object(ec, 'legacy_processes', return_value=[123]), patch('builtins.print'):
                self.assertFalse(ec.activate(root, wait=False))
            with ec.connect(old, True) as con:
                states = dict(con.execute('SELECT id,state FROM jobs'))
            self.assertEqual(states[rows[0].run_id], 'running')
            self.assertEqual(states['outside_design'], 'deferred_v05')
            self.assertTrue((root / 'legacy_before_activation.sqlite').is_file())

    def test_reuse_complete_and_failed_results_and_keep_outside_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'optimized'
            p, rows = tiny_campaign(root, 2)
            old = Path(p['source_root']) / 'train.sqlite'
            run = Path(p['training_output']) / rows[0].run_id
            run.mkdir()
            ec.atomic(run / 'spec.json', rows[0].to_dict())
            ec.atomic(run / 'status.json', {'status': 'trained', 'run_id': rows[0].run_id})
            (run / 'last.pt').write_bytes(b'fixture checkpoint')
            (run / 'best.pt').write_bytes(b'fixture checkpoint')
            with ec.connect(old) as con:
                con.execute("UPDATE jobs SET state='done' WHERE id=?", (rows[0].run_id,))
                con.execute("UPDATE jobs SET state='failed',error='Nonfinite native loss' WHERE id=?", (rows[1].run_id,))
                con.execute("UPDATE jobs SET state='failed',error='outside result' WHERE id='outside_design'")
            with patch.object(ec, 'source_check'), patch.object(ec, 'legacy_processes', return_value=[]), \
                 patch.object(ec, 'validate_completed'), patch('builtins.print'):
                self.assertTrue(ec.activate(root))
                self.assertTrue(ec.activate(root))  # idempotent, not a destructive resynchronization
            with ec.connect(root / 'train.sqlite', True) as con:
                self.assertEqual(dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')), {'done': 1, 'failed': 1})
            self.assertIn('outside_design', (root / 'legacy_evidence.jsonl').read_text())
            self.assertTrue((run / 'last.pt').exists())

    def test_missing_checkpoint_prevents_false_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'optimized'
            p, rows = tiny_campaign(root)
            run = Path(p['training_output']) / rows[0].run_id
            run.mkdir()
            ec.atomic(run / 'spec.json', rows[0].to_dict())
            ec.atomic(run / 'status.json', {'status': 'trained'})
            fake = types.SimpleNamespace(load=lambda *a, **k: {})
            with patch.object(ec, 'source_check'), patch.object(ec, 'legacy_processes', return_value=[]), \
                 patch.dict('sys.modules', {'torch': fake}), patch('builtins.print'):
                with self.assertRaisesRegex(ValueError, 'Incomplete completed evidence'):
                    ec.activate(root)

    def test_reuse_requires_checkpoint_source_and_complete_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            spec = RunSpec()
            ec.atomic(run / 'spec.json', spec.to_dict())
            (run / 'last.pt').write_bytes(b'fixture')
            (run / 'best.pt').write_bytes(b'fixture')
            ck = {'schema': 'snnbench_checkpoint_v2', 'run_id': spec.run_id,
                  'implementation_sha256': 'correct', 'spec': spec.to_dict(),
                  'progress': {'epoch': 60}}
            fake = types.SimpleNamespace(load=lambda *a, **k: ck)
            with patch.dict('sys.modules', {'torch': fake}):
                ec.validate_completed(run, spec.run_id, 'correct')
                with self.assertRaisesRegex(ValueError, 'identity/source'):
                    ec.validate_completed(run, spec.run_id, 'wrong')
                ck['progress']['epoch'] = 10
                with self.assertRaisesRegex(ValueError, 'epoch budget'):
                    ec.validate_completed(run, spec.run_id, 'correct')


class GPUAdmission(unittest.TestCase):
    def setUp(self):
        self.gpu = {'total': 84*ec.GIB, 'free': 84*ec.GIB}

    def test_four_explicit_gpus_and_no_duplicate(self):
        inventory = {f'GPU-{i}': dict(index=i, uuid=f'GPU-{i}', **self.gpu) for i in range(5)}
        self.assertEqual(ec.resolve_gpus(['0', '1', '3', '4'], inventory), ['GPU-0', 'GPU-1', 'GPU-3', 'GPU-4'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ec.resolve_gpus(['1', 'GPU-1'], inventory)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            ec.resolve_gpus(['7'], inventory)

    def test_slots_reserve_and_per_device_budget(self):
        small = {'weight': 1, 'estimate': 2*ec.GIB}
        self.assertTrue(ec.affordable(self.gpu, [small]*3, 1, 2*ec.GIB, 4, .8, 12*ec.GIB))
        self.assertFalse(ec.affordable(self.gpu, [small]*4, 1, 2*ec.GIB, 4, .8, 12*ec.GIB))
        low_free = dict(self.gpu, free=13*ec.GIB)
        self.assertFalse(ec.affordable(low_free, [], 1, 2*ec.GIB, 4, .8, 12*ec.GIB))
        self.assertFalse(ec.affordable(self.gpu, [], 1, 70*ec.GIB, 4, .8, 0))

    def test_large_job_stays_exclusive_even_with_higher_slot_cap(self):
        large = {'weight': 4, 'estimate': 24*ec.GIB}
        self.assertTrue(ec.affordable(self.gpu, [], 4, 24*ec.GIB, 1, .8, 12*ec.GIB))
        self.assertFalse(ec.affordable(self.gpu, [large], 1, 2*ec.GIB, 8, .8, 0))
        self.assertFalse(ec.affordable(self.gpu, [{'weight': 1, 'estimate': 2*ec.GIB}], 4, 24*ec.GIB, 8, .8, 0))

    def test_scheduler_runs_each_job_once_across_four_mock_gpus_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'optimized'
            p, rows = tiny_campaign(root, 7)
            p['status'] = 'activated'
            ec.atomic(root / 'campaign.json', p)
            inventory = {f'GPU-{i}': dict(index=i, uuid=f'GPU-{i}', **self.gpu) for i in (0,1,3,4)}
            launched = []

            class Process:
                def __init__(self, command, **kwargs):
                    run_id = Path(command[command.index('--spec')+1]).stem
                    launched.append((run_id, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                    failed = run_id == rows[0].run_id
                    self.pid = 1000000 + len(launched)
                    self.returncode = int(failed)
                    receipt = Path(command[command.index('--receipt')+1])
                    ec.atomic(receipt, {'state': 'failed' if failed else 'done', 'seconds': 3,
                        'peak_reserved': ec.GIB, 'error': 'Nonfinite native loss' if failed else None})
                    run = Path(p['training_output']) / run_id
                    run.mkdir()
                    ec.atomic(run / 'status.json', {'status': 'failed' if failed else 'trained'})
                def poll(self):
                    return self.returncode

            with patch.object(ec, 'source_check'), patch.object(ec, 'legacy_processes', return_value=[]), \
                 patch.object(ec, 'gpu_inventory', return_value=inventory), \
                 patch.object(ec, 'gpu_compute_processes', return_value=[]), \
                 patch.object(ec.subprocess, 'Popen', Process), patch('builtins.print'):
                ec.schedule(root, ['0','1','3','4'], poll=0, cpu_budget=32)
            self.assertEqual(len(launched), 7)
            self.assertEqual(len(set(r for r, _ in launched)), 7)
            self.assertEqual({g for _, g in launched}, set(inventory))
            with ec.connect(root / 'train.sqlite', True) as con:
                self.assertEqual(dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')), {'done': 6, 'failed': 1})
                self.assertEqual(con.execute('SELECT COUNT(*) FROM allocations').fetchone()[0], 0)
                self.assertEqual(con.execute('SELECT observations FROM memory_profiles').fetchone()[0], 6)

    def test_foreign_gpu_process_is_rejected_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'optimized'
            p, _ = tiny_campaign(root)
            p['status'] = 'activated'
            ec.atomic(root / 'campaign.json', p)
            inventory = {'GPU-2': dict(index=2, uuid='GPU-2', **self.gpu)}
            with patch.object(ec, 'source_check'), patch.object(ec, 'legacy_processes', return_value=[]), \
                 patch.object(ec, 'gpu_inventory', return_value=inventory), \
                 patch.object(ec, 'gpu_compute_processes', return_value=[('GPU-2', 332210)]), \
                 patch.object(ec.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'other compute processes'):
                    ec.schedule(root, ['2'])
                spawn.assert_not_called()

    def test_orphan_in_launch_record_gap_is_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'optimized'
            p, rows = tiny_campaign(root)
            (root / 'queue_specs').mkdir()
            run_id = rows[0].run_id
            spec_path = root / 'queue_specs' / (run_id + '.json')
            receipt = root / 'logs' / (run_id + '.attempt1.resources.json')
            current = str(os.getpid())
            args = [b'python', str(SCRIPT).encode(), b'_train', b'--spec', str(spec_path).encode(),
                    b'--receipt', str(receipt).encode()]
            with ec.connect(root / 'train.sqlite') as con, \
                 patch.object(ec, 'proc_args', side_effect=lambda pid: args if str(pid) == current else []), \
                 patch.object(ec, 'proc_gpu', return_value='GPU-3'):
                ec.recover_children(root, p, con)
                self.assertEqual(con.execute('SELECT pid,gpu FROM allocations').fetchone(), (os.getpid(), 'GPU-3'))
                self.assertEqual(con.execute('SELECT state,attempts FROM jobs').fetchone(), ('running', 1))

    def test_limited_deployment_then_pause_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'optimized'
            p, rows = tiny_campaign(root, 2)
            p['status'] = 'activated'
            ec.atomic(root / 'campaign.json', p)
            inventory = {'GPU-1': dict(index=1, uuid='GPU-1', **self.gpu)}
            launched = []

            class Process:
                def __init__(self, command, **kwargs):
                    run_id = Path(command[command.index('--spec')+1]).stem
                    launched.append(run_id)
                    self.pid = 2000000 + len(launched)
                    self.returncode = 0
                    ec.atomic(command[command.index('--receipt')+1], {'state': 'done'})
                    run = Path(p['training_output']) / run_id
                    run.mkdir()
                    ec.atomic(run / 'status.json', {'status': 'trained'})
                def poll(self):
                    return 0

            with patch.object(ec, 'source_check'), patch.object(ec, 'legacy_processes', return_value=[]), \
                 patch.object(ec, 'gpu_inventory', return_value=inventory), \
                 patch.object(ec, 'gpu_compute_processes', return_value=[]), \
                 patch.object(ec.subprocess, 'Popen', Process), patch('builtins.print'):
                ec.schedule(root, ['1'], poll=0, cpu_budget=4, max_new_jobs=1)
                with ec.connect(root / 'train.sqlite', True) as con:
                    self.assertEqual(dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')), {'done': 1, 'pending': 1})
                ec.atomic(root / 'pause.request', {})
                with self.assertRaisesRegex(RuntimeError, 'Campaign paused'):
                    ec.schedule(root, ['1'])
                ec.schedule(root, ['1'], poll=0, cpu_budget=4, max_new_jobs=1, resume=True)
            self.assertEqual(len(launched), 2)
            self.assertEqual(len(set(launched)), 2)

    def test_checkpoint_requires_exact_unambiguous_gpu_affinity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(ec.checkpoint_affinity(root))
            (root / 'last.pt').write_bytes(b'fixture')
            for device, mask, expected in [('cuda:0', 'GPU-one', 'GPU-one'),
                                           ('cuda:0', '0', None),
                                           ('cuda:0', 'GPU-one,GPU-two', None),
                                           ('cpu', '', None)]:
                fake = types.SimpleNamespace(load=lambda *a, d=device, m=mask, **k: {
                    'environment': {'device': d, 'CUDA_VISIBLE_DEVICES': m}})
                with patch.dict('sys.modules', {'torch': fake}):
                    if expected:
                        self.assertEqual(ec.checkpoint_affinity(root), expected)
                    else:
                        with self.assertRaises(RuntimeError):
                            ec.checkpoint_affinity(root)

    def test_duplicate_coordinator_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / 'scheduler.lock').open('a') as stream:
                ec.fcntl.flock(stream, ec.fcntl.LOCK_EX | ec.fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'no duplicate launched'):
                    ec.schedule(root, ['1'])


if __name__ == '__main__':
    unittest.main()
