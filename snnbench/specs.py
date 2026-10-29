"""Versioned task, model, training and analysis contracts. Unknown fields fail."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import math
from .utils import digest

CELLS=('lif_hard','lif_soft','threshold','if_hard','plif_ref','alif_ref','adlif_ref',
       'hetero_lif','two_state_ref','continuous','tanh','analog_hard_ref')
REFERENCE_BACKBONES=('mlp','recurrent_mlp','temporal_conv','cnn1d','cnn2d','sew_resnet18','spike_attention','unet')
BASELINES=('gru','lstm','diagonal_ssm','causal_attention','convgru')
VECTOR_TASKS=('repeat','stale','complement','order','interval','delay','xor','crosslag','drift','binary_adding')
IMAGE_TASKS=('image_repeat','image_complement','image_stale','image_misaligned','image_rain','image_cloud')


@dataclass
class TaskSpec:
    family:str='tagged'
    name:str='tagged_load'
    objective:str='regression'
    T:int=16
    k:int=4
    sensory_dim:int=8
    history_sources:int=4
    required_history:int=1
    blank_delay:int=3
    extra_writes:int=-1
    sigma:float=.2
    data_seed:int=13400
    projection_seed:int=901
    image_size:int=16
    content_mode:str='physical_k'
    energy_mode:str='fixed_rms'
    repeated_noise:bool=False
    new_steps:int=4
    delay:int=3
    interval:int=3
    shift:int=2
    archive_root:str=''
    archive_manifest_sha256:str=''
    version:str='counter_scene_v2'
    params:dict=field(default_factory=dict)

    def validate(self):
        if self.family not in ('legacy','tagged','spatial','archive','official_adding','copy_memory','mqar','protocol'):
            raise ValueError('Unknown task family: '+self.family)
        if self.objective not in ('classification','regression','reconstruction','sequence_classification'):
            raise ValueError('Unknown objective')
        if min(self.T,self.k,self.sensory_dim,self.image_size)<1 or self.sigma<0:
            raise ValueError('Positive dimensions and nonnegative noise required')
        if self.data_seed<0 or self.projection_seed<0:raise ValueError('Nonnegative seeds')
        if self.family=='legacy' and self.name not in VECTOR_TASKS+IMAGE_TASKS:raise ValueError('Unknown legacy task')
        if self.family=='tagged':
            if self.name not in ('tagged_current','tagged_history','tagged_joint','tagged_load'):raise ValueError('Tagged target rule')
            if not 0<=self.required_history<=self.history_sources or self.history_sources<1:raise ValueError('History request count')
            if self.extra_writes not in (-1,*range(max(1,self.history_sources))):raise ValueError('Extra writes')
            if self.extra_writes>=0 and self.required_history>1:raise ValueError('Absent requested histories: use full writes')
            if self.content_mode not in ('physical_k','demand_only') or self.energy_mode not in ('raw','fixed_rms'):raise ValueError('Content/energy mode')
            if max(4,self.k)>self.sensory_dim:raise ValueError('Sensory projection must preserve base dimensions max(4,k)')
            if self.T!=max(12,3*self.history_sources)+self.blank_delay+1:raise ValueError('Tagged T must equal current-write step + blank_delay + 1')
        if self.family=='spatial' and (self.T!=6 or self.image_size%2 or self.image_size<8):raise ValueError('Spatial v2 uses T6 and even size>=8')
        if self.family=='archive' and (not self.archive_root or not self.archive_manifest_sha256):raise ValueError('Canonical archive requires root and pinned manifest hash')
        if self.family=='official_adding' and (self.T<10 or self.objective!='classification'):raise ValueError('Official adding: T>=10 classification')
        if self.family=='copy_memory' and self.objective!='sequence_classification':raise ValueError('Copy task needs masked sequence loss')
        if self.family=='mqar':
            if self.objective not in ('classification','sequence_classification'):raise ValueError('MQAR class or masked multi-query sequence loss')
            if set(self.params)-{'queries','vocab_size'}:raise ValueError('MQAR reference parameters')
            q=int(self.params.get('queries',1));v=int(self.params.get('vocab_size',8))
            if self.k>v or q<1 or self.T<2*self.k+q or (self.objective=='classification' and q!=1):raise ValueError('MQAR vocabulary/count/query schedule')
        if self.family=='protocol':
            from .protocols import validate_protocol
            validate_protocol(self)

    def identity(self):
        d=asdict(self);d.pop('archive_root');return d


@dataclass
class ModelSpec:
    backbone:str='mlp'
    neuron:str='lif_hard'
    width:int=32
    depth:int=2
    beta:float=.8
    threshold:float=1.
    gain:float=2.
    dc_normalize:bool=False
    gradient_mode:str='full'
    surrogate:str='atan'
    sg_scale:float=2.
    detach_reset:bool=True
    reset_strength:float=1.
    adapt_decay:float=.9
    adapt_gain:float=.1
    observation:str='post'
    readout:str='last'
    readout_decay:float=.8
    normalization:str='none'
    push_pull:bool=False
    encoder:str='auto'
    encoder_width:int=64

    def validate(self):
        if self.backbone not in REFERENCE_BACKBONES+BASELINES or self.neuron not in CELLS:raise ValueError('Unknown backbone/cell')
        if self.width<1 or self.depth<1 or self.encoder_width<1:raise ValueError('Positive width/depth')
        if self.observation not in ('pre','post','output'):raise ValueError('Observation')
        if self.readout not in ('last','mean','sum','exponential','trajectory'):raise ValueError('Readout')
        if self.gradient_mode not in ('full','detach','notd'):raise ValueError('Gradient mode')
        if self.encoder not in ('auto','none','mlp'):raise ValueError('Frame encoder')
        if self.normalization not in ('none','group'):raise ValueError('Normalization')
        if not 0<=self.beta<=1 or self.threshold<=0 or not 0<=self.reset_strength<=1:raise ValueError('Dynamics range')
        if not 0<self.readout_decay<=1 or not 0<=self.adapt_decay<1 or self.adapt_gain<0:raise ValueError('Decay/adaptation')
        if self.dc_normalize and (self.neuron=='if_hard' or self.beta==1):raise ValueError('No finite DC normalization for IF/beta1')
        if self.backbone in BASELINES and self.observation!='output':raise ValueError('Continuous baselines expose output/hidden; no spiking pre/post claim')
        if self.backbone not in ('mlp','recurrent_mlp','gru','lstm') and self.depth!=2:raise ValueError('This backbone uses a fixed skeleton; depth=2 is its compatibility sentinel, not a layer-count sweep')
        if self.normalization=='group' and self.backbone not in ('cnn2d','sew_resnet18','spike_attention','unet'):raise ValueError('Sample-local group normalization is implemented for spatial SNN backbones only')
        if self.surrogate not in ('atan','triangle','sigmoid','rectangle') or not math.isfinite(self.sg_scale) or self.sg_scale<=0:raise ValueError('Declared surrogate and positive scale')
        if not math.isfinite(self.gain) or self.gain<0:raise ValueError('Finite nonnegative input gain')


@dataclass
class TrainSpec:
    seed:int=11
    order_seed:int=1011
    epochs:int=60
    batch_size:int=128
    lr:float=.001
    weight_decay:float=.0001
    clip_norm:float=1.
    train_scenes:int=6144
    val_scenes:int=1536
    threads:int=2
    deterministic:bool=True
    save_every_updates:int=100
    checkpoint_fractions:tuple=(0.,.01,.05,.1,.25,.5,.75,1.)
    encoder_checkpoint:str=''
    encoder_checkpoint_sha256:str=''


@dataclass
class AnalysisSpec:
    fit_scenes:int=512
    select_scenes:int=128
    eval_scenes:int=512
    batch_size:int=64
    max_features:int=128
    max_targets:int=128
    ridge_alphas:tuple=(.0001,.01,1.,100.)
    modules:tuple=('task_controls','working_point','mechanism','probes','geometry','temporal','communication','access','selective','interventions','recovery','reconstruction','anova','theory')
    eval_split:str='development'
    interval_resamples:int=1000
    delta:float=.05
    mlp_steps:int=100
    anova_scenes:int=64


@dataclass
class RunSpec:
    task:TaskSpec=field(default_factory=TaskSpec)
    model:ModelSpec=field(default_factory=ModelSpec)
    train:TrainSpec=field(default_factory=TrainSpec)
    analysis:AnalysisSpec=field(default_factory=AnalysisSpec)
    schema:str='snnbench_run_v2'

    def validate(self):
        self.task.validate();self.model.validate()
        t=self.train;a=self.analysis
        if min(t.epochs,t.batch_size,t.train_scenes,t.val_scenes,t.threads,t.save_every_updates)<1:raise ValueError('Training sizes')
        if t.seed<0 or t.order_seed<0 or t.lr<=0 or t.weight_decay<0 or t.clip_norm<=0:raise ValueError('Optimizer/seed')
        if not all(math.isfinite(x) for x in (t.lr,t.weight_decay,t.clip_norm)):raise ValueError('Finite optimizer parameters')
        if not all(0<=f<=1 for f in t.checkpoint_fractions):raise ValueError('Checkpoint fractions in [0,1]')
        if min(a.fit_scenes,a.select_scenes,a.eval_scenes,a.batch_size,a.max_features,a.max_targets)<2:raise ValueError('Analysis sizes >=2')
        if a.eval_split not in ('development','confirmation','audit'):raise ValueError('Explicit evaluation role')
        if not 0<a.delta<1 or a.interval_resamples<1:raise ValueError('Statistical configuration')
        valid={'task_controls','working_point','mechanism','probes','geometry','temporal','communication','access','selective','interventions','recovery','reconstruction','anova','theory'}
        if set(a.modules)-valid:raise ValueError('Unknown analysis module')
        if self.task.objective=='reconstruction' and self.model.backbone not in ('cnn2d','unet','convgru'):raise ValueError('Native spatial decoder required')
        if self.task.objective=='reconstruction' and self.model.readout=='trajectory':raise ValueError('Spatial trajectory decoder not defined')
        if self.task.objective=='sequence_classification' and self.model.readout=='trajectory':raise ValueError('Sequence head requires per-step output, not fixed-T flatten')
        if t.encoder_checkpoint and not t.encoder_checkpoint_sha256:raise ValueError('Pin frozen encoder checkpoint')
        return self

    def identity(self):
        # Analysis choices do not retrain the core; paths are operational, hashes scientific.
        train=asdict(self.train);train.pop('encoder_checkpoint')
        return {'schema':self.schema,'task':self.task.identity(),'model':asdict(self.model),'train':train}

    @property
    def run_id(self):return digest(self.identity())[:24]

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,value):
        value=dict(value)
        allowed={'task','model','train','analysis','schema'}
        if set(value)-allowed:raise ValueError('Unknown run keys: '+str(set(value)-allowed))
        for key,typ in (('task',TaskSpec),('model',ModelSpec),('train',TrainSpec),('analysis',AnalysisSpec)):
            if key in value:value[key]=typ(**value[key])
        return cls(**value).validate()

    @classmethod
    def load(cls,path):return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
