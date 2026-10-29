"""Executable experiment manifests, with explicit blocked external sources.

No design row is counted as a trained model. Factorials are unions of declared
scientific comparisons; incompatible backbones/targets are never silently run.
"""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from .specs import *
from .utils import atomic_json,file_hash,digest

VECTOR_BACKBONES=('mlp','recurrent_mlp','temporal_conv','cnn1d')
IMAGE_BACKBONES=('cnn2d','sew_resnet18','spike_attention','unet')


def task_bank(profile):
    tasks=[]
    if profile=='smoke':
        return [TaskSpec(),TaskSpec(objective='classification'),TaskSpec(family='spatial',name='spatial_v2',T=6,objective='classification'),TaskSpec(family='spatial',name='spatial_v2',T=6,objective='reconstruction')]
    # Preserve all historical tasks with their native regression/classification alternatives.
    for name in VECTOR_TASKS:
        for objective in ('classification','regression'):tasks.append(TaskSpec(family='legacy',name=name,objective=objective))
    for name in IMAGE_TASKS:
        for objective in ('classification','reconstruction'):tasks.append(TaskSpec(family='legacy',name=name,objective=objective))
    for name in ('tagged_current','tagged_history','tagged_joint','tagged_load'):
        for objective in ('classification','regression'):tasks.append(TaskSpec(name=name,objective=objective))
    tasks.extend([TaskSpec(family='official_adding',name='nine_marker_balanced_reference',objective='classification',T=64),
                  TaskSpec(family='copy_memory',name='copy_memory_reference',objective='sequence_classification',T=64),
                  TaskSpec(family='mqar',name='single_query_reference',objective='classification',T=64)])
    tasks.append(TaskSpec(family='mqar',name='multi_query_reference',objective='sequence_classification',T=64,params={'queries':4,'vocab_size':16}))
    for j in range(1,10):
        for objective in (('classification','reconstruction') if j in (7,8) else ('classification','regression')):
            tasks.append(TaskSpec(family='protocol',name=f'G{j:02d}',T=6 if j in (7,8) else 32,objective=objective))
    if profile=='full':
        # Orthogonal tagged axes: blank age, requested coordinates, requested histories,
        # nonrequested writes, physical content vs demand, and energy convention.
        for k,L,blank,content,energy,objective in product((2,4,8),(0,1,2,4),(0,3,16,64),('physical_k','demand_only'),('raw','fixed_rms'),('classification','regression')):
            tasks.append(TaskSpec(name='tagged_load',k=k,required_history=L,blank_delay=blank,T=13+blank,content_mode=content,energy_mode=energy,objective=objective))
        for extra,blank,objective in product((0,1,2,3),(0,3,16,64),('classification','regression')):
            tasks.append(TaskSpec(name='tagged_history',required_history=1,extra_writes=extra,blank_delay=blank,T=13+blank,objective=objective))
        for K,age,writes,objective in product((2,4,8,16),(8,16,32,64),(0,4,16,32),('classification','regression')):
            t=TaskSpec(family='protocol',name='G01',T=128,objective=objective,params={'items':K,'age':age,'writes':writes})
            try:t.validate();tasks.append(t)
            except ValueError:pass
        for versions,query,invalid,obj in product((0,1,2,4),('latest','protected'),(0.,.25,1.),('classification','regression')):
            tasks.append(TaskSpec(family='protocol',name='G02',T=64,objective=obj,params={'versions':versions,'query':query,'invalidate_probability':invalid,'age':16}))
        for items,obj in product((4,8,16),('classification','regression')):tasks.append(TaskSpec(family='protocol',name='G03',T=64,objective=obj,params={'items':items}))
        for physical,obj in product((False,True),('classification','regression')):tasks.append(TaskSpec(family='protocol',name='G04',T=64,objective=obj,params={'physical_time':physical,'show_clock':True}))
        for count,mode,obj in product((1,2,4,8,16,32),('independent','repeat','causal_hold'),('classification','regression')):
            tasks.append(TaskSpec(family='protocol',name='G05',T=32,objective=obj,params={'measurements':count,'acquisition':mode}))
        for f,d,obj in product((1.,2.,5.,10.),(1.,2.,5.,10.),('classification','regression')):
            tasks.append(TaskSpec(family='protocol',name='G06',T=64,objective=obj,params={'target_frequency':f,'distractor_frequency':d}))
        for T,count,mode,obj in product((16,32,64),(4,8,16),('new','repeat','causal_hold','delayed_linear'),('classification','regression')):
            tasks.append(TaskSpec(family='protocol',name='G09',T=T,objective=obj,params={'measurements':count,'acquisition':mode}))
    return list({digest(t.identity()):t for t in tasks}.values())


def _is_image(t):return t.family=='spatial' or t.name in IMAGE_TASKS or (t.family=='protocol' and t.name in ('G07','G08'))


def model_bank(task,profile):
    image=_is_image(task);recon=task.objective=='reconstruction'
    if profile=='smoke':return [ModelSpec(backbone='cnn2d' if image else 'mlp',width=4,encoder_width=8),ModelSpec(backbone='convgru' if image else 'gru',width=4,encoder_width=8,observation='output')]
    backbones=('cnn2d','unet') if recon else IMAGE_BACKBONES if image else VECTOR_BACKBONES
    cells=CELLS
    for_backbones=[ModelSpec(backbone=b,neuron=n) for b,n in product(backbones,cells)]
    baseline=('convgru',) if image else ('gru','lstm','diagonal_ssm','causal_attention')
    for_backbones.extend(ModelSpec(backbone=b,observation='output') for b in baseline)
    if profile=='paired':
        # Dynamics/readout/observation changes live on the same scientific scene bank.
        for neuron,beta,reset,observation,readout in product(('lif_hard','lif_soft','threshold','continuous'),(.5,.8,.95),(0.,.5,1.),('pre','post','output'),('last','mean','exponential')):
            for_backbones.append(ModelSpec(backbone='cnn2d' if image else 'mlp',neuron=neuron,beta=beta,reset_strength=reset,observation=observation,readout=readout))
    if profile=='learning':
        for neuron,gradient,surrogate,detach,encoder in product(('lif_hard','lif_soft','alif_ref','two_state_ref'),('full','detach','notd'),('atan','triangle','sigmoid','rectangle'),(False,True),('none','auto')):
            for_backbones.append(ModelSpec(backbone='cnn2d' if image else 'mlp',neuron=neuron,gradient_mode=gradient,surrogate=surrogate,detach_reset=detach,encoder=encoder))
    return list({digest(m):m for m in for_backbones}.values())


def compile_manifest(out,profile='screen',seeds=(11,22,33),archives=(),limit=None):
    if profile not in ('smoke','screen','full','paired','learning'):raise ValueError('Profile')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);count=0;seen=set();counts={};training=TrainSpec()
    if profile=='smoke':training=TrainSpec(epochs=2,train_scenes=32,val_scenes=16,batch_size=8,save_every_updates=2)
    tasks=task_bank(profile);external=[]
    for root in archives:
        p=Path(root);d=json.loads((p/'dataset.json').read_text());c=d['contract'];objective='classification' if c['target_mode']=='class_index' else 'reconstruction' if len(c['output_shape'])==3 else 'regression'
        t=TaskSpec(family='archive',name=c['dataset_id'],objective=objective,T=c['steps'],archive_root=str(p.resolve()),archive_manifest_sha256=file_hash(p/'dataset.json'))
        external.append((t,d))
    with out.open('w',encoding='utf-8') as f:
        for task,manifest in [(t,None) for t in tasks]+external:
            for model in model_bank(task,profile) if manifest is None else model_bank(replace(task,family='spatial' if len(manifest['contract']['input_shape'])==3 else 'tagged'),profile):
                for seed in seeds:
                    train=replace(training,seed=int(seed),order_seed=int(seed)+1000);analysis=AnalysisSpec()
                    if profile=='smoke':analysis=AnalysisSpec(fit_scenes=24,select_scenes=16,eval_scenes=16,batch_size=8,max_features=8,interval_resamples=50,mlp_steps=5,anova_scenes=8)
                    if manifest is not None:
                        sizes={k:v['samples'] for k,v in manifest['splits'].items()};train=replace(train,train_scenes=min(train.train_scenes,sizes['train']),val_scenes=min(train.val_scenes,sizes['val']))
                        analysis=replace(analysis,fit_scenes=min(analysis.fit_scenes,sizes['probe_train'],sizes['verification_train']),select_scenes=min(analysis.select_scenes,sizes['probe_val'],sizes['calibration'],sizes['verification_val']),eval_scenes=min(analysis.eval_scenes,sizes['development'],sizes['confirmation'],sizes['audit']))
                    spec=RunSpec(task=task,model=model,train=train,analysis=analysis).validate()
                    if spec.run_id in seen:continue
                    seen.add(spec.run_id);f.write(json.dumps(spec.to_dict(),ensure_ascii=False,separators=(',',':'))+'\n');count+=1
                    key=task.family+'/'+task.objective;counts[key]=counts.get(key,0)+1
                    if limit is not None and count>=limit:break
                if limit is not None and count>=limit:break
            if limit is not None and count>=limit:break
    receipt={'schema':'executable_manifest_v2','profile':profile,'seeds':list(seeds),'runs':count,'counts':counts,'manifest_sha256':file_hash(out),
        'meaning':'validated runnable specifications; no trained-model claim','external_archives':list(archives),
        'external_without_preparation':'registry candidates are excluded, never synthesized or silently substituted'}
    atomic_json(out.with_suffix(out.suffix+'.receipt.json'),receipt);return receipt
