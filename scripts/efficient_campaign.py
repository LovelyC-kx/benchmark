#!/usr/bin/env python3
"""Coverage-first campaign planning and GPU scheduling, outside training identity.

Uses the existing training engine without changing its equations, budget, batches,
seeds, checkpoint policy, or strict resume rules. Plan is read-only with respect
to the legacy campaign. Activation is explicit, backed up, and drains old work.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack, closing
from dataclasses import replace
import fcntl
from functools import wraps
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import traceback

_guess = Path(__file__).resolve().parents[1]
if (_guess / 'src/benchmark/snnbench').is_dir():
    _guess = _guess / 'src/benchmark'
CODE = Path(os.environ.get('SNN_CODE', _guess)).resolve()
sys.path.insert(0, str(CODE))

GIB = 2**30
PRIMARY_SEEDS = (11, 22, 33)
EXTRA_SEEDS = (44, 55)
MODULES = ('task_controls', 'working_point', 'mechanism', 'probes', 'geometry',
           'temporal', 'communication', 'access', 'selective', 'interventions',
           'recovery', 'reconstruction', 'anova', 'theory')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def connect(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        return sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=60)
    con = sqlite3.connect(path, timeout=60)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=60000')
    return con


def locked(*names):
    def decorator(function):
        @wraps(function)
        def wrapper(root, *args, **kwargs):
            with ExitStack() as stack:
                for name in names:
                    stream = stack.enter_context((Path(root) / name).open('a'))
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise RuntimeError('Another controller holds ' + name + '; no duplicate launched')
                return function(root, *args, **kwargs)
        return wrapper
    return decorator


def init_queue(con):
    con.executescript('''
      CREATE TABLE jobs(id TEXT PRIMARY KEY, spec TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending', owner TEXT, lease REAL,
        attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
      CREATE TABLE design(id TEXT PRIMARY KEY, priority INTEGER, ordinal INTEGER,
        reasons TEXT, resource_key TEXT, weight INTEGER, estimate INTEGER);
      CREATE INDEX scheduling ON design(priority,ordinal);
      CREATE TABLE allocations(id TEXT PRIMARY KEY, pid INTEGER, host TEXT,
        gpu TEXT, started REAL, receipt TEXT, log TEXT);
      CREATE TABLE memory_profiles(key TEXT PRIMARY KEY, peak INTEGER, seconds REAL,
        observations INTEGER);
    ''')


def task_key(task):
    return canonical(task.identity())


def models_for(task, manifest, profile='expanded'):
    from snnbench.catalogue import model_bank
    proxy = task if manifest is None else replace(
        task, family='spatial' if len(manifest['contract']['input_shape']) == 3 else 'tagged')
    return model_bank(proxy, profile, 'expanded')


def anchor_models(task, manifest):
    from snnbench.specs import BASELINES
    bank = models_for(task, manifest)
    spatial = any(m.backbone == 'cnn2d' for m in bank)
    names = ('cnn2d', 'convgru') if spatial else ('mlp', 'gru')
    chosen = [m for m in bank if
              (m.backbone == names[0] and m.neuron in ('lif_hard', 'continuous'))
              or m.backbone == names[1]]
    assert len(chosen) == 3
    return chosen


def make_spec(task, model, seed, manifest):
    from snnbench.specs import TrainSpec, AnalysisSpec, RunSpec
    train = replace(TrainSpec(), seed=seed, order_seed=seed + 1000)
    analysis = AnalysisSpec()
    if manifest is not None:
        n = {k: v['samples'] for k, v in manifest['splits'].items()}
        train = replace(train, train_scenes=min(train.train_scenes, n['train']),
                        val_scenes=min(train.val_scenes, n['val']))
        analysis = replace(analysis,
            fit_scenes=min(analysis.fit_scenes, n['probe_train'], n['verification_train']),
            select_scenes=min(analysis.select_scenes, n['probe_val'], n['calibration'], n['verification_val']),
            eval_scenes=min(analysis.eval_scenes, n['development'], n['confirmation'], n['audit']))
    return RunSpec(task=task, model=model, train=train, analysis=analysis).validate()


def resource_hint(spec, manifest):
    task, model, train = spec.task, spec.model, spec.train
    if manifest is not None:
        shape = manifest['contract']['input_shape']
    else:
        from snnbench.catalogue import _is_image
        shape = [3, task.image_size, task.image_size] if _is_image(task) else [task.sensory_dim]
    image = len(shape) == 3
    attention = any(x in model.backbone for x in ('attention', 'transformer', 'conformer', 'vit'))
    if (image and (task.T > 16 or model.backbone not in ('cnn2d', 'convgru'))
            or attention and task.T > 256):
        weight, estimate = 4, 24 * GIB
    elif image or task.T > 128 or attention:
        weight, estimate = 2, 8 * GIB
    else:
        weight, estimate = 1, 2 * GIB
    key = hashlib.sha256(canonical({
        'shape': shape, 'T': task.T, 'objective': task.objective,
        'model': spec.to_dict()['model'], 'batch': train.batch_size,
    }).encode()).hexdigest()
    return key, weight, estimate


def boundary_tasks(tasks):
    """Eight predeclared contrasts, not selected using model performance."""
    selected = []
    for name in ('G01', 'G02', 'G05', 'G06'):
        pool = [t for t in tasks if t.family == 'protocol' and t.name == name
                and t.objective == 'regression' and t.params]
        if not pool:
            raise ValueError('Missing boundary protocol ' + name)
        if name == 'G01':
            score = lambda t: (t.params['items'], t.params['age'], t.params['writes'])
        elif name == 'G02':
            score = lambda t: (t.params['versions'], t.params['invalidate_probability'], t.params['query'])
        elif name == 'G05':
            # Same high measurement count: independent versus repeated acquisition.
            independent = [t for t in pool if t.params['acquisition'] == 'independent']
            repeat = [t for t in pool if t.params['acquisition'] == 'repeat']
            selected += [max(independent, key=lambda t: t.params['measurements']),
                         max(repeat, key=lambda t: t.params['measurements'])]
            continue
        else:
            score = lambda t: (abs(t.params['target_frequency'] - t.params['distractor_frequency']),
                               t.params['target_frequency'], t.params['distractor_frequency'])
        selected += [min(pool, key=score), max(pool, key=score)]
    return selected


def design(core, all_tasks, archives, add):
    """All base models x core tasks x 3 seeds; full axes on explicit anchors."""
    from snnbench.specs import BASELINES
    all_core = [(t, None) for t in core] + archives
    banks = [(t, m, models_for(t, m)) for t, m in all_core]
    # Round-robin task coverage, rather than lexicographically sorted run hashes.
    for seed in PRIMARY_SEEDS:
        for index in range(max(len(b) for _, _, b in banks)):
            for task, manifest, bank in banks:
                if index < len(bank):
                    add(task, bank[index], seed, manifest, 'core', 10 if seed == 11 else 40)

    core_ids = {task_key(t) for t in core}
    for task in all_tasks:
        if task_key(task) not in core_ids:
            for model in anchor_models(task, None):
                for seed in PRIMARY_SEEDS:
                    add(task, model, seed, None, 'full_condition_axes', 20 if seed == 11 else 50)

    paired = [('tagged', 'tagged_history', 'regression')]
    paired += [('protocol', name, 'reconstruction' if name in ('G07', 'G08') else 'regression')
               for name in ('G01', 'G02', 'G05', 'G06', 'G07', 'G08', 'G09')]
    for family, name, objective in paired:
        matches = [t for t in core if (t.family, t.name, t.objective) == (family, name, objective)]
        if len(matches) != 1:
            raise ValueError('Paired anchor is not unique: ' + name)
        task = matches[0]
        base = {canonical(make_spec(task, m, 11, None).to_dict()['model']) for m in models_for(task, None)}
        for model in models_for(task, None, 'paired'):
            if canonical(make_spec(task, model, 11, None).to_dict()['model']) not in base:
                for seed in PRIMARY_SEEDS:
                    add(task, model, seed, None, 'paired_local_factorial', 20 if seed == 11 else 50)

    learning = []
    for family, name, objective in [('copy_memory', 'copy_memory_reference', 'sequence_classification'),
                                   ('protocol', 'G02', 'classification'),
                                   ('protocol', 'G07', 'reconstruction')]:
        candidates = [t for t in core if (t.family, t.name, t.objective) == (family, name, objective)]
        if len(candidates) != 1:
            raise ValueError('Learning anchor is not unique: ' + name)
        learning.append((candidates[0], None))
    ssc = [(t, m) for t, m in archives if Path(t.archive_root).name == 'ssc']
    if len(ssc) != 1:
        raise ValueError('A real SSC canonical archive named ssc is required for the learning anchor')
    learning += ssc
    for task, manifest in learning:
        base = {canonical(make_spec(task, m, 11, manifest).to_dict()['model']) for m in models_for(task, manifest)}
        for model in models_for(task, manifest, 'learning'):
            if canonical(make_spec(task, model, 11, manifest).to_dict()['model']) not in base:
                for seed in PRIMARY_SEEDS:
                    add(task, model, seed, manifest, 'learning_local_factorial', 20 if seed == 11 else 50)

    for task in boundary_tasks(all_tasks):
        for model in models_for(task, None):
            for seed in PRIMARY_SEEDS:
                add(task, model, seed, None, 'cross_backbone_boundary', 30 if seed == 11 else 50)

    # Five-seed controls cover every base task, every cell on one backbone,
    # and every compatible continuous baseline, irrespective of observed score.
    for task, manifest, bank in banks:
        anchor = 'cnn2d' if any(m.backbone == 'cnn2d' for m in bank) else 'mlp'
        for model in bank:
            if model.backbone == anchor or model.backbone in BASELINES:
                for seed in EXTRA_SEEDS:
                    add(task, model, seed, manifest, 'five_seed_sentinels', 60)


def plan(source_root, out, max_runs=45000):
    from snnbench.catalogue import task_bank
    from snnbench.specs import TaskSpec, RunSpec
    source_root, out = Path(source_root).resolve(), Path(out).resolve()
    if source_root == out or out.exists():
        raise FileExistsError('Use a fresh, distinct optimized campaign directory')
    legacy = json.loads((source_root / 'campaign.json').read_text())
    if not legacy['materialized']:
        raise ValueError('Legacy queue must already exist')
    required = {'full', 'paired', 'learning'}
    if not required.issubset(legacy['profiles']) or not set(PRIMARY_SEEDS + EXTRA_SEEDS).issubset(legacy['seeds']):
        raise ValueError('Source campaign must contain all profiles and five declared seeds')
    inventory = source_root / 'all_layer_coverage.json'
    if not inventory.is_file():
        raise FileNotFoundError('Original six-layer inventory required: ' + str(inventory))
    archives = []
    for path in legacy['archives']:
        path = Path(path).resolve()
        manifest = json.loads((path / 'dataset.json').read_text())
        c = manifest['contract']
        objective = c.get('objective') or ('classification' if c['target_mode'] == 'class_index'
                    else 'reconstruction' if len(c['output_shape']) == 3 else 'regression')
        task = TaskSpec(family='archive', name=c['dataset_id'], objective=objective, T=c['steps'],
                        archive_root=str(path), archive_manifest_sha256=sha(path / 'dataset.json'))
        archives.append((task, manifest))
    core, all_tasks = task_bank('expanded'), task_bank('full')
    selected = {}
    intended = {}

    def add(task, model, seed, manifest, reason, priority):
        spec = make_spec(task, model, seed, manifest)
        run_id = spec.run_id
        intended[reason] = intended.get(reason, 0) + 1
        if run_id not in selected:
            key, weight, estimate = resource_hint(spec, manifest)
            selected[run_id] = {'spec': spec.to_dict(), 'reasons': set(), 'priority': priority,
                                'ordinal': len(selected), 'resource_key': key,
                                'weight': weight, 'estimate': estimate}
        selected[run_id]['reasons'].add(reason)
        selected[run_id]['priority'] = min(priority, selected[run_id]['priority'])

    design(core, all_tasks, archives, add)
    if len(selected) > max_runs:
        raise ValueError(f'Design requires {len(selected)} runs, budget is {max_runs}; no requirements silently dropped')
    # Every selected training identity must already exist in the original plan.
    with closing(connect(source_root / 'train.sqlite', True)) as old:
        legacy_counts = dict(old.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        for run_id, row in selected.items():
            found = old.execute('SELECT spec FROM jobs WHERE id=?', (run_id,)).fetchone()
            if found is None:
                raise ValueError('Requested identity not in legacy queue: ' + run_id)
            original = json.loads(found[0])
            if canonical(RunSpec.from_dict(original).identity()) != canonical(RunSpec.from_dict(row['spec']).identity()):
                raise ValueError('Legacy scientific identity mismatch: ' + run_id)
            row['spec'] = original  # preserve original analysis defaults and operational paths too

    out.mkdir(parents=True)
    reasons = {}
    coverage = {}
    update_budget = 0
    con = connect(out / 'train.sqlite')
    init_queue(con)
    with con, (out / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
        for run_id, row in selected.items():
            spec = row['spec']
            payload = canonical(spec)
            manifest.write(payload + '\n')
            con.execute('INSERT INTO jobs(id,spec) VALUES(?,?)', (run_id, payload))
            con.execute('INSERT INTO design VALUES(?,?,?,?,?,?,?)', (
                run_id, row['priority'], row['ordinal'], canonical(sorted(row['reasons'])),
                row['resource_key'], row['weight'], row['estimate']))
            for reason in row['reasons']:
                reasons[reason] = reasons.get(reason, 0) + 1
            spec_obj = RunSpec.from_dict(spec)
            key = task_key(spec_obj.task)
            coverage.setdefault(key, {'task': spec['task'], 'seeds': set(), 'runs': 0,
                                      'backbones': set(), 'neurons': set(), 'packages': set()})
            c = coverage[key]
            c['seeds'].add(spec['train']['seed'])
            c['runs'] += 1
            c['backbones'].add(spec['model']['backbone'])
            c['neurons'].add(spec['model']['neuron'])
            c['packages'].update(row['reasons'])
            update_budget += spec['train']['epochs'] * math.ceil(spec['train']['train_scenes'] / spec['train']['batch_size'])
    con.close()
    expected_keys = {task_key(t) for t in all_tasks} | {task_key(t) for t, _ in archives}
    if set(coverage) != expected_keys:
        raise AssertionError('Task-condition coverage mismatch')
    serialized = []
    for value in coverage.values():
        serialized.append({k: sorted(v) if isinstance(v, set) else v for k, v in value.items()})
    atomic(out / 'training_coverage.json', {'all_task_contracts_covered': True,
        'task_contracts': len(coverage), 'core_contracts': len(core), 'real_contracts': len(archives),
        'rows': serialized,
        'scope': 'Every original task contract, core compatible model, local factorial and declared seed is generated explicitly; not all global interactions are trained.'})
    # Preserve the complete original ledger. No heading or shared module is relabelled completed.
    (out / 'all_layer_coverage.json').write_bytes(inventory.read_bytes())
    original_ledger = json.loads(inventory.read_text())
    requirement_routes = []
    for row in original_ledger.get('master_cards', []) + original_ledger.get('specialized_subsections', []):
        requirement_routes.append({
            'id': row['id'], 'original_requirement': row,
            'candidate_training_packages': sorted(reasons),
            'shared_analysis_components': row.get('shared_runner_components', []),
            'execution_state': 'pending', 'complete_section_claim': False,
            'closure_requires': 'Original controls, source-specific adapters, outputs, and applicability decisions; training membership or shared-module output alone does not close this requirement.',
        })
    atomic(out / 'requirement_routes.json', {
        'scope': 'Requirement retention and candidate evidence routes, not scientific completion',
        'requirements': requirement_routes, 'original_counts': original_ledger.get('counts'),
        'untrained_global_interactions': 'All original source-queue rows remain recoverable as deferred work; request targeted follow-ups explicitly.',
    })
    atomic(out / 'analysis_plan.json', {
        'shared_modules': MODULES, 'checkpoints': 'all saved checkpoints',
        'scene_banks': 'original declared complete-coordinate banks, original precision',
        'training_readouts': 'independent training retained in paired_local_factorial',
        'posthoc_readouts': 'fit independent readers on frozen representations with original role separation',
        'specialized_ledger': 'all_layer_coverage.json', 'all_layer_analysis_complete': False,
        'author_native_experiments': 'registered separately; not implemented by this scheduler',
        'cross_module_reuse': 'Share raw traces and valid identical fits; no discarded coordinates or merged intervention branches',
    })
    roster = source_root / 'source_roster.json'
    if roster.exists():
        (out / 'source_roster.json').write_bytes(roster.read_bytes())
    plan_value = {
        'schema': 'efficient_campaign_v05', 'status': 'planned_not_activated',
        'planner_sha256': sha(Path(__file__).resolve()),
        'source_root': str(source_root), 'training_output': legacy['training_output'],
        'data_output': legacy['data_output'],
        'training_implementation_sha256': legacy['training_implementation_sha256'],
        'source_counts_at_plan': legacy_counts, 'legacy_runs': sum(legacy_counts.values()),
        'selected_runs': len(selected), 'reduction_factor': sum(legacy_counts.values()) / len(selected),
        'optimizer_updates_upper_budget': update_budget, 'package_memberships': reasons,
        'package_additions_before_dedup': intended, 'main_seeds': PRIMARY_SEEDS,
        'sentinel_seeds': PRIMARY_SEEDS + EXTRA_SEEDS,
        'manifest_sha256': sha(out / 'manifest.jsonl'),
        'original_coverage_sha256': sha(inventory),
        'source_roster_sha256': sha(roster) if roster.exists() else None,
        'activation_policy': 'Back up, defer old pending jobs, drain active work, then reuse completed/failed evidence; do not delete the legacy campaign',
        'gpu_policy': {'ids': [], 'expected_cards': 4, 'declared_memory_gib_per_card': 84,
                       'budget_fraction': .8, 'reserve_gib': 12, 'max_slots': 4,
                       'unknown_small_gib': 2, 'unknown_medium_gib': 8,
                       'unknown_large_gib': 24, 'large_job_exclusive': True},
        'changes_to_model_or_training_budget': False,
    }
    atomic(out / 'campaign.json', plan_value)
    print(json.dumps({k: v for k, v in plan_value.items() if k != 'gpu_policy'}, ensure_ascii=False, indent=2))
    return plan_value


def proc_args(pid):
    try:
        return Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
    except (OSError, ValueError):
        return []


def legacy_processes(plan_value):
    db = str(Path(plan_value['source_root']) / 'train.sqlite').encode()
    runs = plan_value['training_output'].encode()
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        args = proc_args(path.name)
        if b'snnbench' in args and ((b'worker' in args and db in args)
                                    or (b'train' in args and runs in args)):
            found.append(int(path.name))
    return found


def source_check(p):
    from snnbench.engine import implementation_hash
    if implementation_hash() != p['training_implementation_sha256']:
        raise ValueError('Training source fingerprint changed; do not mix implementations')


def check_manifest(root, p):
    if sha(root / 'manifest.jsonl') != p['manifest_sha256']:
        raise ValueError('Optimized manifest changed')
    digest = hashlib.sha256()
    with closing(connect(root / 'train.sqlite', True)) as con:
        for payload, in con.execute('SELECT j.spec FROM jobs j JOIN design d USING(id) ORDER BY d.ordinal'):
            digest.update(payload.encode() + b'\n')
    if digest.hexdigest() != p['manifest_sha256']:
        raise ValueError('Queue scientific specifications differ from the frozen manifest')


def validate_completed(run, run_id, implementation):
    import torch
    from snnbench.specs import RunSpec
    saved = RunSpec.load(run / 'spec.json')
    if saved.run_id != run_id or not all((run / name).is_file() for name in ('last.pt', 'best.pt')):
        raise ValueError('Incomplete completed evidence: ' + str(run))
    for name in ('last.pt', 'best.pt'):
        ck = torch.load(run / name, map_location='cpu', weights_only=True)
        if (ck.get('schema') != 'snnbench_checkpoint_v2' or ck.get('run_id') != run_id
                or ck.get('implementation_sha256') != implementation
                or RunSpec.from_dict(ck['spec']).run_id != run_id):
            raise ValueError('Completed checkpoint identity/source mismatch: ' + str(run / name))
        if name == 'last.pt' and ck['progress']['epoch'] != saved.train.epochs:
            raise ValueError('Completed checkpoint has not finished the declared epoch budget')


@locked('control.lock')
def activate(root, wait=False, poll=5):
    root = Path(root).resolve()
    p = json.loads((root / 'campaign.json').read_text())
    source_check(p)
    check_manifest(root, p)
    if p['status'] == 'activated':
        status_report(root)
        return True
    old_path = Path(p['source_root']) / 'train.sqlite'
    old = connect(old_path)
    backup = root / 'legacy_before_activation.sqlite'
    if not backup.exists():
        temporary = backup.with_name(backup.name + f'.tmp-{os.getpid()}')
        with sqlite3.connect(temporary) as target:
            old.backup(target)
            if target.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Legacy backup integrity check failed')
        os.replace(temporary, backup)
    while True:
        with old:
            old.execute("UPDATE jobs SET state='deferred_v05' WHERE state='pending'")
        processes = legacy_processes(p)
        running = old.execute("SELECT COUNT(*) FROM jobs WHERE state='running'").fetchone()[0]
        print(json.dumps({'legacy_running': running, 'legacy_processes': processes,
                          'policy': 'finish current tasks; no signals sent'}), flush=True)
        if not processes:
            # All registered execution processes are gone. Any leftover leases
            # stay deferred, with their partial checkpoints retained for affinity-aware resume.
            owners = old.execute("SELECT owner FROM jobs WHERE state='running'").fetchall()
            if any(owner and not owner.startswith(socket.gethostname() + ':') for owner, in owners):
                raise RuntimeError('Nonlocal owner found; cannot declare a drained local campaign')
            with old:
                old.execute("UPDATE jobs SET state='deferred_v05',owner=NULL,lease=NULL WHERE state='running'")
            break
        if not wait:
            old.close()
            print('Old pending work is deferred. Re-run activate --wait after active work finishes.')
            return False
        time.sleep(poll)
    new = connect(root / 'train.sqlite')
    reused = []
    with new:
        for run_id, payload, state in new.execute('SELECT id,spec,state FROM jobs').fetchall():
            original = old.execute('SELECT state,error FROM jobs WHERE id=?', (run_id,)).fetchone()
            run = Path(p['training_output']) / run_id
            status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
            if original[0] == 'done' or status.get('status') == 'trained':
                validate_completed(run, run_id, p['training_implementation_sha256'])
                new.execute("UPDATE jobs SET state='done',error=NULL WHERE id=?", (run_id,))
                reused.append({'run_id': run_id, 'state': 'done', 'path': str(run),
                               'spec_sha256': sha(run / 'spec.json'), 'status_sha256': sha(run / 'status.json'),
                               'last_checkpoint_sha256': sha(run / 'last.pt'),
                               'best_checkpoint_sha256': sha(run / 'best.pt')})
            elif original[0] == 'failed':
                new.execute("UPDATE jobs SET state='failed',error=? WHERE id=?", (original[1], run_id))
                reused.append({'run_id': run_id, 'state': 'failed', 'path': str(run), 'error': original[1]})
    # Preserve old completed evidence even where it is outside the optimized training design.
    with (root / 'legacy_evidence.jsonl').open('w') as stream:
        for run_id, state, error in old.execute("SELECT id,state,error FROM jobs WHERE state IN ('done','failed')"):
            stream.write(canonical({'run_id': run_id, 'state': state, 'error': error,
                'path': str(Path(p['training_output']) / run_id), 'retain_for_discovery': True}) + '\n')
    atomic(root / 'reused_evidence.json', reused)
    p['status'] = 'activated'
    p['activation_time'] = time.time()
    atomic(root / 'campaign.json', p)
    old.close()
    new.close()
    status_report(root)
    return True


def status_report(root):
    root = Path(root)
    with closing(connect(root / 'train.sqlite', True)) as con:
        counts = dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        allocations = [dict(zip(('run_id', 'pid', 'host', 'gpu'), row)) for row in con.execute(
            'SELECT id,pid,host,gpu FROM allocations')]
    result = {'root': str(root.resolve()), 'counts': counts, 'active_allocations': allocations}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def gpu_inventory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    result = {}
    for row in csv.reader(io.StringIO(output)):
        index, uuid, total, free = [v.strip() for v in row]
        result[uuid] = {'index': int(index), 'uuid': uuid, 'total': int(total) * 2**20,
                        'free': int(free) * 2**20}
    return result


def resolve_gpus(tokens, inventory):
    selected = []
    for token in tokens:
        matches = [g for g in inventory.values() if token == g['uuid'] or token == str(g['index'])]
        if len(matches) != 1:
            raise ValueError('Unknown GPU: ' + token)
        selected.append(matches[0]['uuid'])
    if len(selected) != len(set(selected)):
        raise ValueError('Duplicate GPU')
    return selected


def affordable(device, active, weight, estimate, max_slots, fraction, reserve):
    if sum(min(a['weight'], max_slots) for a in active) + min(weight, max_slots) > max_slots:
        return False
    # A class-4 job remains exclusive even if --max-slots is raised.
    if active and (weight == 4 or any(a['weight'] == 4 for a in active)):
        return False
    if sum(a['estimate'] for a in active) + estimate > device['total'] * fraction:
        return False
    return device['free'] >= estimate + reserve


def checkpoint_affinity(run):
    path = Path(run) / 'last.pt'
    if not path.exists():
        return None
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    env = checkpoint['environment']
    if not str(env['device']).startswith('cuda'):
        raise RuntimeError('CPU checkpoint cannot be exact-resumed on CUDA: ' + str(run))
    uuid = env.get('CUDA_VISIBLE_DEVICES')
    if not uuid or not uuid.startswith('GPU-') or ',' in uuid:
        raise RuntimeError('Checkpoint GPU affinity is not an unambiguous UUID: ' + str(run))
    return uuid


def gpu_compute_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True)
    return [(row[0].strip(), int(row[1].strip())) for row in csv.reader(io.StringIO(output)) if row]


def proc_gpu(pid):
    env = Path(f'/proc/{int(pid)}/environ').read_bytes().split(b'\0')
    return next(os.fsdecode(x).split('=', 1)[1] for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))


def recover_children(root, p, con):
    """Recover also the Popen-to-allocation crash window from scoped /proc argv."""
    helper = str(Path(__file__).resolve()).encode()
    registered = {row[0] for row in con.execute('SELECT id FROM allocations')}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        args = proc_args(proc.name)
        if helper not in args or b'_train' not in args or b'--spec' not in args:
            continue
        try:
            spec_path = Path(os.fsdecode(args[args.index(b'--spec') + 1]))
            receipt = Path(os.fsdecode(args[args.index(b'--receipt') + 1]))
            if spec_path.parent != root / 'queue_specs':
                continue
            run_id = spec_path.stem
            if run_id in registered:
                continue
            row = con.execute('SELECT state FROM jobs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                raise RuntimeError('Unregistered live child: ' + run_id)
            gpu = proc_gpu(proc.name)
            logfile = str(receipt).replace('.resources.json', '.log')
            attempt = int(receipt.name.split('.attempt')[1].split('.')[0])
            with con:
                con.execute("UPDATE jobs SET state='running',owner=?,attempts=MAX(attempts,?) WHERE id=?",
                            (socket.gethostname() + ':' + proc.name, attempt, run_id))
                con.execute('INSERT INTO allocations VALUES(?,?,?,?,?,?,?)',
                    (run_id, int(proc.name), socket.gethostname(), gpu, time.time(), str(receipt), logfile))
            registered.add(run_id)
        except (FileNotFoundError, ProcessLookupError):
            continue


def train_one(spec_path, out, data, receipt):
    import torch
    from snnbench.engine import train
    from snnbench.specs import RunSpec
    started = time.monotonic()
    result = {'state': 'failed', 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'launcher_sha256': sha(Path(__file__).resolve())}
    try:
        spec = RunSpec.load(spec_path)
        result['run_id'] = spec.run_id
        train(spec, out_root=out, data_root=data, device='cuda:0')
        result['state'] = 'done'
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        result['seconds'] = time.monotonic() - started
        if torch.cuda.is_initialized():
            try:
                result['peak_reserved'] = torch.cuda.max_memory_reserved(0)
                result['peak_allocated'] = torch.cuda.max_memory_allocated(0)
            except Exception as exc:
                result['resource_measurement_error'] = type(exc).__name__ + ': ' + str(exc)
        atomic(receipt, result)


@locked('scheduler.lock', 'control.lock')
def schedule(root, gpu_tokens, max_slots=4, fraction=.8, reserve_gib=12, poll=1, cpu_budget=None, resume=False, max_new_jobs=None):
    root = Path(root).resolve()
    pause = root / 'pause.request'
    if resume:
        pause.unlink(missing_ok=True)
    elif pause.exists():
        raise RuntimeError('Campaign paused; use run --resume to continue')
    p = json.loads((root / 'campaign.json').read_text())
    if p['status'] != 'activated':
        raise RuntimeError('Activate and drain the legacy queue first')
    source_check(p)
    check_manifest(root, p)
    if legacy_processes(p):
        raise RuntimeError('Legacy execution process found; drain it before scheduling')
    inventory = gpu_inventory()
    selected = resolve_gpus(gpu_tokens, inventory)
    if not selected:
        raise ValueError('An explicit GPU list is required')
    atomic(root / 'gpu_allocation.json', {'gpus': [inventory[g] for g in selected],
        'scheduler_sha256': sha(Path(__file__).resolve()),
        'max_slots': max_slots, 'budget_fraction': fraction, 'reserve_gib': reserve_gib,
        'individual_gpu_memory_not_pooled': True, 'training_source_unchanged': True})
    con = connect(root / 'train.sqlite')
    recover_children(root, p, con)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    specs = root / 'queue_specs'
    specs.mkdir(exist_ok=True)
    active = {}
    affinity = {}
    # Recover child processes from a previous scheduler without launching duplicates.
    for run_id, pid, host, gpu, started, receipt, logfile in con.execute('SELECT * FROM allocations').fetchall():
        if host != socket.gethostname():
            raise RuntimeError('Nonlocal active allocation')
        if gpu not in selected:
            raise RuntimeError('A live/recoverable allocation requires GPU ' + gpu + '; include it explicitly')
        meta = con.execute('SELECT d.resource_key,d.weight,d.estimate,j.spec FROM design d JOIN jobs j USING(id) WHERE id=?', (run_id,)).fetchone()
        active[run_id] = {'pid': pid, 'gpu': gpu, 'receipt': receipt, 'log': logfile,
            'key': meta[0], 'weight': meta[1], 'estimate': meta[2], 'process': None,
            'threads': json.loads(meta[3])['train']['threads']}

    owned = {a['pid'] for run_id, a in active.items()
             if str(root / 'queue_specs' / (run_id + '.json')).encode() in proc_args(a['pid'])
             and str(Path(__file__).resolve()).encode() in proc_args(a['pid'])}
    foreign = [(g, pid) for g, pid in gpu_compute_processes() if g in selected and pid not in owned]
    if foreign:
        raise RuntimeError('Selected GPUs have other compute processes; no process stopped: ' + repr(foreign))
    affinity_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    cpu_budget = max(2, int(affinity_cpus * .8)) if cpu_budget is None else cpu_budget
    if cpu_budget < 1:
        raise ValueError('Positive CPU budget required')
    print(canonical({'cpu_thread_budget': cpu_budget, 'gpus': selected,
                     'max_small_jobs_per_gpu': max_slots, 'large_jobs_exclusive': True}), flush=True)

    idle_since = None
    started_jobs = 0
    try:
        while True:
            # Receipt + exit + trained status are needed to declare completion.
            for run_id, a in list(active.items()):
                proc = a['process']
                alive = proc.poll() is None if proc is not None else (
                    b'_train' in proc_args(a['pid'])
                    and str(Path(__file__).resolve()).encode() in proc_args(a['pid'])
                    and str(root / 'queue_specs' / (run_id + '.json')).encode() in proc_args(a['pid']))
                if alive:
                    continue
                receipt = json.loads(Path(a['receipt']).read_text()) if Path(a['receipt']).exists() else {}
                trained = Path(p['training_output']) / run_id / 'status.json'
                state = json.loads(trained.read_text()).get('status') if trained.exists() else None
                ok = receipt.get('state') == 'done' and state == 'trained' and (proc is None or proc.returncode == 0)
                error = None if ok else receipt.get('error', 'Execution failed; see ' + a['log'])
                with con:
                    con.execute('UPDATE jobs SET state=?,error=?,owner=NULL,lease=NULL WHERE id=?',
                                ('done' if ok else 'failed', error, run_id))
                    con.execute('DELETE FROM allocations WHERE id=?', (run_id,))
                    if ok and receipt.get('peak_reserved'):
                        peak = int(receipt['peak_reserved'] * 1.3 + GIB)
                        old = con.execute('SELECT peak,seconds,observations FROM memory_profiles WHERE key=?', (a['key'],)).fetchone()
                        con.execute('INSERT OR REPLACE INTO memory_profiles VALUES(?,?,?,?)', (
                            a['key'], max(peak, old[0] if old else 0), receipt.get('seconds', 0.), (old[2] if old else 0) + 1))
                print(canonical({'run_id': run_id, 'state': 'done' if ok else 'failed', 'error': error,
                                 'seconds': receipt.get('seconds'), 'gpu': a['gpu']}), flush=True)
                del active[run_id]

            if pause.exists():
                if not active:
                    print('PAUSED: active tasks drained; pending jobs preserved.', flush=True)
                    break
                time.sleep(poll)
                continue
            if max_new_jobs is not None and started_jobs >= max_new_jobs:
                if not active:
                    print('Invocation job limit reached; remaining jobs stay pending.', flush=True)
                    break
                time.sleep(poll)
                continue
            inventory = gpu_inventory()
            candidates = con.execute('''SELECT j.id,j.spec,d.resource_key,d.weight,d.estimate
                FROM jobs j JOIN design d USING(id) WHERE j.state='pending'
                ORDER BY d.priority,d.ordinal LIMIT 512''').fetchall()
            launched = False
            for run_id, payload, key, weight, estimate in candidates:
                if max_new_jobs is not None and started_jobs >= max_new_jobs:
                    break
                threads = json.loads(payload)['train']['threads']
                if sum(a['threads'] for a in active.values()) + threads > cpu_budget:
                    continue
                if run_id not in affinity:
                    affinity[run_id] = checkpoint_affinity(Path(p['training_output']) / run_id)
                learned = con.execute('SELECT peak FROM memory_profiles WHERE key=?', (key,)).fetchone()
                estimate = max(GIB, learned[0]) if learned else estimate
                choices = [g for g in selected if affinity[run_id] in (None, g)
                           and affordable(inventory[g], [a for a in active.values() if a['gpu'] == g],
                                          weight, estimate, max_slots, fraction, reserve_gib * GIB)]
                if not choices:
                    continue
                gpu = min(choices, key=lambda g: (sum(a['weight'] for a in active.values() if a['gpu'] == g), -inventory[g]['free']))
                spec_path = specs / (run_id + '.json')
                spec_path.write_text(payload)
                attempt = con.execute('SELECT attempts FROM jobs WHERE id=?', (run_id,)).fetchone()[0] + 1
                receipt = logs / f'{run_id}.attempt{attempt}.resources.json'
                logfile = logs / f'{run_id}.attempt{attempt}.log'
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1',
                           PYTHONPATH=str(CODE) + os.pathsep + env.get('PYTHONPATH', ''),
                           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                command = [sys.executable, str(Path(__file__).resolve()), '_train',
                    '--spec', str(spec_path), '--out', p['training_output'], '--data', p['data_output'],
                    '--receipt', str(receipt)]
                with logfile.open('a') as stream:
                    process = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                with con:
                    con.execute("UPDATE jobs SET state='running',owner=?,attempts=? WHERE id=?",
                                (socket.gethostname() + ':' + str(process.pid), attempt, run_id))
                    con.execute('INSERT INTO allocations VALUES(?,?,?,?,?,?,?)',
                        (run_id, process.pid, socket.gethostname(), gpu, time.time(), str(receipt), str(logfile)))
                active[run_id] = {'pid': process.pid, 'gpu': gpu, 'process': process, 'receipt': str(receipt),
                    'log': str(logfile), 'key': key, 'weight': weight, 'estimate': estimate, 'threads': threads}
                inventory[gpu]['free'] -= estimate  # reserve launches whose CUDA context is not yet visible
                launched = True
                started_jobs += 1
            remaining = con.execute("SELECT COUNT(*) FROM jobs WHERE state='pending'").fetchone()[0]
            if not active and not remaining:
                break
            if not active and not launched:
                if idle_since is None:
                    idle_since = time.monotonic()
                if time.monotonic() - idle_since > 60:
                    blocked = [{'run_id': r[0], 'required_gpu': affinity.get(r[0]), 'estimate_bytes': r[4]}
                               for r in candidates[:20]]
                    atomic(root / 'admission_wait.json', {'pending': remaining, 'examples': blocked,
                        'reason': 'No admissible GPU; check affinity, memory reserve, slot capacity, and competing processes'})
                    raise RuntimeError('No admissible work for 60s; inspect admission_wait.json')
            else:
                idle_since = None
            time.sleep(poll)
    finally:
        # Child training processes keep running after coordinator exit. Their
        # durable allocations and receipts are adopted by the next coordinator.
        con.close()
    status_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    q = commands.add_parser('plan')
    q.add_argument('--source-root', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--max-runs', type=int, default=45000)
    q = commands.add_parser('activate')
    q.add_argument('--root', required=True)
    q.add_argument('--wait', action='store_true')
    q = commands.add_parser('status')
    q.add_argument('--root', required=True)
    q = commands.add_parser('pause')
    q.add_argument('--root', required=True)
    q = commands.add_parser('run')
    q.add_argument('--root', required=True)
    q.add_argument('--gpus', nargs='+', required=True)
    q.add_argument('--max-slots', type=int, default=4)
    q.add_argument('--budget-fraction', type=float, default=.8)
    q.add_argument('--reserve-gib', type=float, default=12)
    q.add_argument('--cpu-budget', type=int)
    q.add_argument('--resume', action='store_true')
    q.add_argument('--max-new-jobs', type=int)
    q = commands.add_parser('_train')
    for flag in ('spec', 'out', 'data', 'receipt'):
        q.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        plan(args.source_root, args.out, args.max_runs)
    elif args.command == 'activate':
        activate(args.root, args.wait)
    elif args.command == 'status':
        status_report(args.root)
    elif args.command == 'pause':
        root = Path(args.root).resolve()
        if not (root / 'campaign.json').is_file():
            parser.error('Unknown campaign')
        atomic(root / 'pause.request', {'time': time.time(), 'policy': 'drain current tasks, retain pending work'})
        print('Pause requested; current tasks finish normally.')
    elif args.command == 'run':
        if (not 0 < args.budget_fraction <= 1 or not math.isfinite(args.reserve_gib)
                or args.reserve_gib < 0 or args.max_slots < 1
                or args.max_new_jobs is not None and args.max_new_jobs < 1):
            parser.error('Invalid resource budget')
        schedule(args.root, args.gpus, args.max_slots, args.budget_fraction, args.reserve_gib,
                 cpu_budget=args.cpu_budget, resume=args.resume, max_new_jobs=args.max_new_jobs)
    else:
        train_one(args.spec, args.out, args.data, args.receipt)


if __name__ == '__main__':
    main()
