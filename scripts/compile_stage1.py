#!/usr/bin/env python3
"""Compile the all-layer first stage without touching running work or old results.

Emits training specifications, shared analysis recipes, complete-capture requests,
and a requirement dependency ledger. A source file or recipe is not execution.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict, replace
from itertools import product
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import efficient_campaign as ec
from snnbench.specs import TaskSpec, RunSpec, ModelSpec, AnalysisSpec

SEEDS = (11, 22, 33)
ALL_MODULES = tuple(AnalysisSpec().modules)
CONTENT = ('working_point', 'probes', 'geometry', 'temporal', 'access')
GROUP_INFO = {
 'mechanisms': ('同一输入下的内容保留、可读、利用与发放/重置作用', ('L1.C','L1.D','L2.1','L2.2','L2.4','L2.5','L2.7','L3.1','L3.3','L3.4','L3.5','L3.6','L3.7','L4.1','L4.2','L4.3','L4.4','L6.5')),
 'classification': ('同源分类目标与连续恢复需求的差异', ('L1.C','L1.F','L4.2')),
 'history_structure': ('历史必要性、关联查询、次序与间隔需求', ('L1.1','L1.2','L1.G')),
 'age_writes': ('空白年龄与额外非目标写入分离', ('L2.3','L3.2','L3.6','L5.1','L5.2')),
 'update_protection': ('最新值更新与未更新项目的保护', ('L1.D','L5.2')),
 'measurements': ('独立新观测、重复与保持的区分', ('L1.E','L2.3','L5.1')),
 'physical_time': ('物理时间、可用时刻与内部步的边界', ('L1.E','L2.3','L3.6','L5.1')),
 'frequency': ('可辨识频率与重叠控制', ('L2.3','L3.3')),
 'spatial': ('同源分类/重建与有效、过期历史的空间调用', ('L1.F','L2.6','L3.5','L4.5')),
 'spatial_aux': ('重建中膜状态和辅助状态的作用范围', ('L2.5','L4.5')),
 'real': ('已准备真实输入上的实测分析', ('L1.G','L5.3','L5.5')),
 'capacity': ('内容维数K、请求数L、物理宽度N的匹配对照', ('L2.2','L3.2','L3.3','L3.4','L5.1')),
 'energy': ('原始与固定RMS写入尺度的对照', ('L2.2','L3.2','L6.3')),
 'dc_normalization': ('原始写入与DC归一化的匹配参照', ('L2.2','L2.3')),
 'backbone': ('显式循环/延迟路径与局部状态替代', ('L2.1','L2.6','L3.7','L5.5')),
 'credit': ('full/detach同前向与notd前向变化', ('L6.1','L6.4','L6.5')),
 'surrogate': ('同前向的SG形状和reset梯度规则', ('L6.2','L6.5')),
 'encoder': ('共同编码器与编码先验', ('L2.2','L6.3')),
 'normalization': ('样本内空间归一化及工作点', ('L2.2','L6.3')),
 'learned_decay': ('可学习衰减与异质初始化的形成过程', ('L2.5','L6.5')),
 'optimization': ('固定完整预算的有限学习率对照', ('L6.4',)),
 'observation': ('相同内部权限与额外状态权限的读出对照', ('L4.1','L4.2')),
 'heldout_design': ('冻结规则后的留出条件与设计效用', ('L5.4','L4.3','L4.4')),
}


def m(neuron='lif_hard', backbone='mlp', **changes):
    # Matches the expanded roster. Continuous baseline defaults stay identical.
    value = ModelSpec(backbone=backbone, neuron=neuron, observation='output',
                      gain=2. if backbone == 'gru' else .5)
    return replace(value, **changes)


def design(archives):
    rows = {}
    memberships = Counter()
    groups = {}

    def add(group, tasks, models, modules=CONTENT, priority=20, train_changes=None):
        title, cards = GROUP_INFO[group]
        g = groups.setdefault(group, {'title':title, 'cards':list(cards), 'run_ids':set()})
        for item, model, seed in product(tasks, models, SEEDS):
            task, manifest = item if isinstance(item, tuple) else (item, None)
            spec = ec.make_spec(task, model, seed, manifest)
            if train_changes:
                spec = replace(spec, train=replace(spec.train, **train_changes)).validate()
            key = spec.run_id
            memberships[group] += 1
            if key not in rows:
                resource, weight, estimate = ec.resource_hint(spec, manifest)
                rows[key] = {'spec':spec.to_dict(), 'groups':set(), 'modules':set(),
                    'priority':priority, 'ordinal':len(rows), 'resource_key':resource,
                    'weight':weight, 'estimate':estimate}
            row = rows[key]
            row['groups'].add(group); row['modules'].update(modules)
            row['priority'] = min(priority, row['priority']); g['run_ids'].add(key)

    current = TaskSpec(name='tagged_current')
    history = TaskSpec(name='tagged_history')
    mechanistic = [m(cell) for cell in ('lif_hard','threshold','analog_hard_ref',
                    'continuous','lif_soft','alif_ref','two_state_ref')] + [m(backbone='gru')]
    anchors = [m(), m('continuous'), m(backbone='gru')]
    two = [m(), m('continuous')]
    add('mechanisms', [current, history], mechanistic, ALL_MODULES, 0)
    add('classification', [replace(t, objective='classification') for t in (current,history)],
        anchors, ('task_controls',)+CONTENT, 10)
    add('history_structure', [
        TaskSpec(family='copy_memory',name='copy_memory_reference',objective='sequence_classification',T=64),
        TaskSpec(family='mqar',name='multi_query_reference',objective='sequence_classification',T=64,
                 params={'queries':4,'vocab_size':16}),
        TaskSpec(family='protocol',name='G03',objective='classification',T=32),
        TaskSpec(family='protocol',name='G04',objective='regression',T=32)],
        anchors, ('task_controls',)+CONTENT, 20)
    add('age_writes', [replace(history, blank_delay=b, T=13+b, extra_writes=w)
        for b,w in product((3,64),(0,3))], anchors, ALL_MODULES, 5)
    add('update_protection', [TaskSpec(family='protocol',name='G02',T=64,
        params={'versions':4,'query':q,'invalidate_probability':.25,'age':16})
        for q in ('latest','protected')], anchors, ('task_controls',)+CONTENT, 20)
    add('measurements', [TaskSpec(family='protocol',name='G05',T=32,
        params={'measurements':n,'acquisition':a})
        for n,a in product((1,8,32),('independent','repeat','causal_hold'))],
        anchors, ('task_controls','working_point','probes','access'), 20)
    add('physical_time', [TaskSpec(family='protocol',name='G09',T=t,
        params={'measurements':4,'acquisition':a})
        for t,a in product((16,64),('new','causal_hold'))], two,
        ('task_controls','working_point','probes','temporal','access'), 20)
    # G04 irregular intervals require an observable clock. This is not G09 resampling.
    add('physical_time', [TaskSpec(family='protocol',name='G04',T=64,
        params={'physical_time':p,'show_clock':True}) for p in (False,True)], two,
        ('task_controls',)+CONTENT, 20)
    add('frequency', [TaskSpec(family='protocol',name='G06',T=64,
        params={'target_frequency':2.,'distractor_frequency':d}) for d in (2.,5.)],
        two, ('task_controls','working_point','probes','access'), 20)
    images = [TaskSpec(family='protocol',name=n,T=6,objective=o)
        for n,o in product(('G07','G08'),('classification','reconstruction'))]
    image_models = [m(backbone='cnn2d'),m('continuous',backbone='cnn2d'),m(backbone='convgru',gain=2.)]
    spatial_modules = ('task_controls','working_point','probes','geometry','access','interventions','reconstruction')
    add('spatial', images, image_models, spatial_modules, 10)
    recon = [t for t in images if t.objective=='reconstruction']
    add('spatial_aux', recon, [m(c,backbone='cnn2d') for c in ('alif_ref','two_state_ref')],
        spatial_modules, 20)
    real = [(t,a) for t,a in archives if Path(t.archive_root).name in ('shd','har')]
    if {Path(t.archive_root).name for t,_ in real} != {'shd','har'}:
        raise ValueError('First-stage real input evidence requires prepared canonical shd and har archives')
    add('real', real, anchors, ('task_controls','working_point','access'), 10)
    for width in (16,32):
        add('capacity', [TaskSpec(k=k,required_history=l) for k,l in product((2,8),(1,4))],
            [replace(x,width=width) for x in two], ALL_MODULES, 20)
    add('energy', [TaskSpec(k=8,required_history=4,energy_mode='raw')], two, ALL_MODULES, 20)
    add('dc_normalization',[history],[replace(x,dc_normalize=d) for x,d in product(two,(False,True))],
        ALL_MODULES,20)
    add('backbone', [current,history], [m(c,backbone=b)
        for c,b in product(('lif_hard','continuous'),('recurrent_mlp','temporal_conv'))],
        ('working_point','probes','geometry','communication','interventions','access'), 10)
    add('credit', [history], [m(gradient_mode=g) for g in ('full','detach','notd')], ALL_MODULES, 5)
    add('surrogate', [history], [m(surrogate=s,detach_reset=d)
        for s,d in product(('atan','triangle'),(False,True))], ALL_MODULES, 5)
    add('encoder', [history], [m(encoder=e) for e in ('none','auto')], ALL_MODULES, 20)
    add('normalization', [recon[0]], [m(backbone='cnn2d',normalization=n) for n in ('none','group')],
        spatial_modules, 20)
    add('learned_decay', [history], [m(c) for c in ('plif_ref','hetero_lif')], ALL_MODULES, 10)
    for lr in (.0003,.001,.003):
        add('optimization',[history],[m()],ALL_MODULES,20,{'lr':lr})
    add('observation',[history],[m(observation=o) for o in ('pre','post','output')],ALL_MODULES,10)
    # Outcomes cannot pick these before the discovery rule is frozen. No audit role is opened.
    add('heldout_design',[replace(history,blank_delay=16,T=29,extra_writes=w) for w in (0,3)],
        anchors, CONTENT, 90)
    return rows, groups, memberships


def load_archives(legacy):
    result=[]
    for raw in legacy['archives']:
        root=Path(raw).resolve();path=root/'dataset.json';manifest=json.loads(path.read_text())
        c=manifest['contract'];objective=c.get('objective') or ('classification' if c['target_mode']=='class_index'
                 else 'reconstruction' if len(c['output_shape'])==3 else 'regression')
        task=TaskSpec(family='archive',name=c['dataset_id'],objective=objective,T=c['steps'],
            archive_root=str(root),archive_manifest_sha256=ec.sha(path))
        result.append((task,manifest))
    return result


def module_id(row):
    if row['id'].startswith('master:'):return row['source_module_id']
    return 'L'+'.'.join(row['source_section_id'].split('.')[:2])


def requirement_dependencies(inventory, groups):
    from snnbench.coverage import ROUTES
    requirements=[]
    for original in inventory['master_cards']+inventory['specialized_subsections']:
        module=module_id(original)
        matched=[k for k,g in groups.items() if module in g['cards']]
        science=module.startswith('L')
        if science and not matched:
            raise ValueError('Unrouted analytical requirement: '+original['id'])
        code=original.get('code_candidates',[])
        special=original['id'].startswith('section:')
        requirements.append({'id':original['id'],'module':module,'original_requirement':original,
            'first_stage_required':True,'training_groups':matched,
            'shared_components':original.get('shared_runner_components',ROUTES.get(module,[])),
            'shared_route_is_partial':True,
            'specialized_source_code_present':any(c.get('exists',False) for c in code) if code else None,
            'declared_code_candidates':len(code),
            'adapter_execution_verified':False,
            'execution_status':'specialized_adapter_requires_verification' if special else 'planned',
            'result_artifacts':[],'complete_section_claim':False,
            'closure':'Execute original applicable controls and validate outputs; a routed training group, source file, or shared module does not close this requirement.'})
    return requirements


def checkpoint_recipes(spec, row):
    a=asdict(spec.analysis)
    a['modules']=[x for x in ALL_MODULES if x in row['modules']]
    # This is exploratory development. Confirmation needs a separate frozen recipe.
    a['eval_split']='development'
    yield {'checkpoint':'best.pt','analysis':a,'purpose':'main_shared_analysis'}
    if spec.task.family=='tagged' and spec.task.name=='tagged_history' and not spec.task.params \
            and spec.task.blank_delay==3 and spec.task.extra_writes==-1 \
            and spec.model.backbone=='mlp' and spec.model.neuron in \
                ('lif_hard','continuous','alif_ref','two_state_ref','plif_ref','hetero_lif'):
        total=spec.train.epochs*math.ceil(spec.train.train_scenes/spec.train.batch_size)
        b=dict(a,modules=list(CONTENT))
        for fraction in (0.,.1,.5):
            yield {'checkpoint':f'step_{math.ceil(fraction*total):08d}.pt',
                   'analysis':b,'purpose':'same_model_formation'}
        yield {'checkpoint':'last.pt','analysis':b,'purpose':'same_model_formation'}


def audit_blockers():
    # These are explicit implementation/validation dependencies, not assertions
    # that the archived standalone protocols are absent or scientifically wrong.
    return [
      {'id':'nonrequested_content','modules':['L3.3','L4.3'],'requirement':'Add nonrequested-source erasure, independent hit/collateral verification, and matched random/no-op controls. Current shared selective primarily targets the focal source.'},
      {'id':'specialized_protocol_adapters','modules':['L3.2','L3.3','L3.4','L3.6','L3.7','L4.1','L4.2','L4.3','L4.4','L4.5'],'requirement':'Verify each original subsection adapter on trained research-model checkpoints. Shared component receipts are partial evidence.'},
      {'id':'known_answer_calibration','modules':['L2.3','L3.2','L3.4','L4.4'],'requirement':'Attach current known-answer algebra/reference calibration and replay/identity controls before interpreting empirical effects.'},
      {'id':'capacity_anchor_loss','modules':['L3.2'],'requirement':'The native objective averages over requested coordinates. Verify fixed-anchor loss weighting and scoring controls before treating K/L/N comparisons as a capacity law; post-hoc focal scoring alone does not equal matched training weights.'},
      {'id':'same_forward_learning','modules':['L6.1','L6.2'],'requirement':'Check same-weight full/detach and SG/reset-gradient forward equivalence separately from retraining; notd changes forward.'},
      {'id':'path_restoration','modules':['L3.7','L4.3','L4.4'],'requirement':'Validate actual binary edges, donor/time/dose controls, joint owners and restoration followed by reblocking.'},
      {'id':'spatial_history','modules':['L1.F','L4.5'],'requirement':'Validate original spatial intervention/restoration protocol, C/H/U regions, alignment and frequency outputs. Shared reconstruction metrics alone are insufficient.'},
      {'id':'design_rule_freeze','modules':['L5.4','L4.3','L4.4'],'requirement':'Freeze a diagnostic-to-modification rule and matched selection-cost controls before opening held-out results. No automatic audit/confirmation access.'},
      {'id':'real_reconstruction','modules':['L4.5','L5.3'],'requirement':'Prepared SHD/HAR support real classification only. Real reconstruction source/adaptation remains separately blocked; synthetic G07/G08 are not its substitute.'},
      {'id':'cost_calibration','modules':['L5.5','L6.4'],'requirement':'Measure training, archive and analysis runtimes/storage separately on representative vector/long/image jobs. GPU memory is not a throughput forecast.'},
    ]


def declared_changes(spec):
    """Only these predeclared scientific additions may lack an old identity."""
    model=spec.model;changes=[]
    if model.width==16:changes.append('physical_width_16')
    if model.dc_normalize:changes.append('DC_normalized_write')
    if model.normalization=='group':changes.append('sample_local_group_normalization')
    if spec.train.lr!=.001:changes.append('learning_rate')
    if model.encoder=='none':changes.append('encoder_none')
    if model.gain==.5:
        if model.gradient_mode!='full':changes.append('matched_gain_credit_rule')
        if model.surrogate!='atan' or not model.detach_reset:changes.append('matched_gain_surrogate_or_reset_gradient')
        if model.observation!='output':changes.append('matched_gain_observation')
    return changes


def compile_plan(source_root, out, source_code=None, max_runs=2000):
    source_root=Path(source_root).resolve();out=Path(out).resolve()
    if out.exists() or out==source_root:raise FileExistsError('Use a fresh first-stage directory; old results are never replaced')
    legacy=json.loads((source_root/'campaign.json').read_text())
    if not legacy.get('materialized'):raise ValueError('Original materialized campaign required')
    if not {'full','paired','learning'}.issubset(legacy.get('profiles',[])) or not set(SEEDS).issubset(legacy.get('seeds',[])):
        raise ValueError('Source must be the original full/paired/learning campaign with all three primary seeds')
    code=Path(source_code or ec.CODE).resolve()
    from snnbench.coverage import plan_inventory
    inventory=plan_inventory(code)
    if 'explicit_protocol_heading_index' not in inventory:
        raise ValueError('Update snnbench/coverage.py to reconcile the original explicit subsection headings')
    science={r['source_module_id'] for r in inventory['master_cards'] if r['source_module_id'].startswith('L')}
    expected={c for _,cards in GROUP_INFO.values() for c in cards}
    if science!=expected:raise ValueError('Master requirement set changed; reconcile before compiling: '+repr(science^expected))
    rows,groups,memberships=design(load_archives(legacy))
    if len(rows)>max_runs:raise ValueError(f'{len(rows)} declared runs exceed {max_runs}; no automatic subsampling')
    dependencies=requirement_dependencies(inventory,groups)
    origins=Counter();reuse=Counter();durations=[]
    with closing(ec.connect(source_root/'train.sqlite',True)) as old:
        source_counts=dict(old.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        for run_id,row in rows.items():
            found=old.execute('SELECT spec,state,error FROM jobs WHERE id=?',(run_id,)).fetchone()
            row['origin']='new_explicit_contract'
            row['declared_changes']=declared_changes(RunSpec.from_dict(row['spec']))
            if found:
                original=json.loads(found[0])
                if ec.canonical(RunSpec.from_dict(original).identity())!=ec.canonical(RunSpec.from_dict(row['spec']).identity()):
                    raise ValueError('Legacy identity mismatch: '+run_id)
                row['spec']=original;row['origin']='existing_legacy_contract';row['legacy_state']=found[1]
            elif not row['declared_changes']:
                raise ValueError('Expected original identity is missing; do not relabel it as a new experiment: '+run_id)
            origins[row['origin']]+=1
            run=Path(legacy['training_output'])/run_id
            status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
            if status.get('status')=='trained' and all((run/n).is_file() for n in ('spec.json','last.pt','best.pt')):
                row['reuse']='candidate_requires_checkpoint_validation'
                reuse['completed_candidates']+=1
                if isinstance(status.get('elapsed_seconds_this_invocation'),(int,float)):
                    durations.append({'run_id':run_id,'seconds_this_invocation':status['elapsed_seconds_this_invocation'],
                        'scope':'May exclude earlier resumed work; not an unbiased complete-job duration'})
            elif status.get('status')=='failed' or (found and found[1]=='failed'):
                row['reuse']='failed_evidence_retained';reuse['legacy_failed']+=1
            elif found and found[1]=='running':
                row['reuse']='currently_running';reuse['legacy_running']+=1
            else:row['reuse']='training_required';reuse['training_required']+=1
    out.mkdir(parents=True)
    con=ec.connect(out/'train.sqlite');ec.init_queue(con)
    recipes=[];requests=[];formation=0;updates=0;provenance=[]
    with con,(out/'manifest.jsonl').open('w') as stream:
        for run_id,row in rows.items():
            payload=ec.canonical(row['spec']);stream.write(payload+'\n')
            con.execute('INSERT INTO jobs(id,spec) VALUES(?,?)',(run_id,payload))
            con.execute('INSERT INTO design VALUES(?,?,?,?,?,?,?)',(run_id,row['priority'],row['ordinal'],
                ec.canonical(sorted(row['groups'])),row['resource_key'],row['weight'],row['estimate']))
            spec=RunSpec.from_dict(row['spec'])
            updates+=spec.train.epochs*math.ceil(spec.train.train_scenes/spec.train.batch_size)
            provenance.append({'run_id':run_id,'origin':row['origin'],'reuse':row['reuse'],
                'groups':sorted(row['groups']),'declared_scientific_changes':row['declared_changes'],'training_spec':row['spec']})
            for recipe in checkpoint_recipes(spec,row):
                job={'kind':'analyze','run_id':run_id,'run':str(Path(legacy['training_output'])/run_id),
                    'data':legacy['data_output'],'entry':'snnbench.analysis:analyze',**recipe,
                    'dependency':'trained_checkpoint_and_source_hash_validation',
                    'gate':'frozen_rule_before_heldout_analysis' if row['groups']=={'heldout_design'} else None,
                    'state':'planned_not_executed','specialized_sections_complete':False}
                job['id']=__import__('hashlib').sha256(ec.canonical(job).encode()).hexdigest()[:24]
                recipes.append(job);formation+=recipe['purpose']=='same_model_formation'
            requests.append({'kind':'archive','run_id':run_id,'run':str(Path(legacy['training_output'])/run_id),
                'data':legacy['data_output'],'checkpoints':'all saved checkpoints',
                'roles':['calibration','probe_train','probe_val','verification_train','verification_val','development'],
                'scenes':None,'coordinate_policy':'all original coordinates and tensor bits',
                'checkpoint_expansion':'after training, fingerprint each actual file; do not replace unknown counts with completed jobs',
                'state':'planned_not_executed'})
    con.close()
    for name,values in [('analysis_jobs.jsonl',recipes),('capture_requests.jsonl',requests),('training_provenance.jsonl',provenance)]:
        with (out/name).open('w') as stream:
            for value in values:stream.write(ec.canonical(value)+'\n')
    old_inventory=source_root/'all_layer_coverage.json'
    (out/'original_all_layer_coverage.json').write_bytes(old_inventory.read_bytes())
    ec.atomic(out/'all_layer_coverage.json',inventory)
    ec.atomic(out/'requirement_dependencies.json',{'requirements':dependencies,'all_sections_complete':False})
    ec.atomic(out/'implementation_audit.json',{'unclosed_dependencies':audit_blockers(),
        'scope':'Needs current checkpoint adaptation/validation; source presence is not an execution receipt',
        'original_specialized_requirements':len(inventory['specialized_subsections'])})
    group_json={k:{**g,'run_ids':sorted(g['run_ids']),'unique_runs':len(g['run_ids'])} for k,g in groups.items()}
    ec.atomic(out/'scientific_groups.json',group_json)
    main=len(rows);optimistic=main-reuse['completed_candidates']-reuse['legacy_failed']
    estimate={'training_jobs_upper':main,'training_jobs_after_candidate_reuse':optimistic,
        'reuse_candidates_are_not_validated_completions':True,
        'scenarios':[{'hypothetical_aggregate_runs_per_minute':rate,'training_hours_upper':main/rate/60,
            'training_hours_after_candidate_reuse':optimistic/rate/60} for rate in (1.5,3.,4.)],
        'interpretation':'Count-only scenarios. 1.5/min came from prior mixed six-worker user observation; 3/4 are unmeasured scenarios, not claimed four-GPU throughput.',
        'archive_hours':None,'analysis_hours':None,'specialized_adapter_hours':None,
        'full_stage_walltime':None,'reason':'Measure heterogeneous analysis/capture and unresolved adapter work before a deadline forecast',
        'completed_invocation_durations':durations}
    ec.atomic(out/'runtime_budget.json',estimate)
    plan={'schema':'all_layer_stage1_v1','status':'planned_not_activated','execution_root':str(out),
        'source_root':str(source_root),'training_output':legacy['training_output'],'data_output':legacy['data_output'],
        'training_implementation_sha256':legacy['training_implementation_sha256'],
        'planner_sha256':ec.sha(Path(__file__).resolve()),'scheduler_sha256':ec.sha(Path(ec.__file__)),
        'manifest_sha256':ec.sha(out/'manifest.jsonl'),'source_counts_at_plan':source_counts,
        'legacy_runs':sum(source_counts.values()),'selected_runs':main,'origin_counts':dict(origins),
        'reuse_candidates':dict(reuse),'primary_seeds':SEEDS,'optimizer_updates_upper_budget':updates,
        'shared_analysis_jobs':len(recipes),'same_model_formation_jobs':formation,
        'complete_capture_requests':len(requests),'capture_checkpoint_count':'expand actual saved files after training',
        'capture_scene_policy_unchanged':True,'training_engine_unchanged':True,
        'all_layers':['L1','L2','L3','L4','L5','L6'],'master_cards':len(inventory['master_cards']),
        'specialized_sections':len(inventory['specialized_subsections']),
        'all_original_analysis_complete':False,'shared_analysis_is_not_specialized_completion':True,
        'original_coverage_sha256':ec.sha(old_inventory),'stage1_coverage_sha256':ec.sha(out/'all_layer_coverage.json'),
        'declared_group_additions':dict(memberships),'original_training_budget_reduced':False,
        'independent_new_contracts_are_explicit':True,'plan_mutates_legacy_queue':False}
    ec.atomic(out/'campaign.json',plan)
    write_report(out,plan,groups,dependencies,estimate)
    print(json.dumps(plan,ensure_ascii=False,indent=2))
    return plan


def write_report(out,plan,groups,dependencies,budget):
    lines=['# 第一阶段：训练与全部分析层级的依赖清单','',
        '这是编译产物，不是已完成实验。编译不修改正在运行的旧队列。','',
        f"独立训练：{plan['selected_runs']}；共同分析作业：{plan['shared_analysis_jobs']}；完整轨迹归档请求：{plan['complete_capture_requests']}。",
        f"主卡：{plan['master_cards']}；专项编号：{plan['specialized_sections']}；全部六层已登记。",
        '新增科学配置包含物理宽度与匹配学习规则等，单独列入training_provenance.jsonl。','',
        '## 科学实验组','', '| 组 | 目的 | 独立配置 |','|---|---|---:|']
    for key,g in groups.items():lines.append(f"| {key} | {g['title']} | {len(g['run_ids'])} |")
    lines+=['','组间重叠按run_id去重；每组使用三个训练种子，共用数据生成协议，不声称三个独立数据世界。','',
        '## 每节的依赖','', '| 原编号 | 对应训练组 | 状态 |','|---|---|---|']
    for r in dependencies:lines.append(f"| {r['id']} | {', '.join(r['training_groups']) or '统计/来源/交付核验'} | {r['execution_status']} |")
    lines+=['','## 耗时','',
        'runtime_budget.json给出计数换算场景。完整分析、归档、专项适配耗时尚未测量，因此不把训练小时数当作整阶段完成时间。','',
        '## 执行状态','',
        '所有共同分析仅请求development角色。heldout_design组的结果需等诊断规则和对照成本冻结后再打开；confirmation/audit未被自动访问。',
        'implementation_audit.json保留专项适配与验证依赖。源码存在、路由存在或共同模块完成均不代表原小节闭环。',
        '已有完成结果须经检查点/源代码/完整训练预算核验才复用，已有失败保持失败证据。']
    (out/'STAGE1_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--source-code');p.add_argument('--max-runs',type=int,default=2000)
    args=p.parse_args();compile_plan(args.source_root,args.out,args.source_code,args.max_runs)


if __name__=='__main__':main()
