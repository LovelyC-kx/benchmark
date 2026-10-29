"""Common causal interface for reference SNNs and continuous temporal controls.

These are independent reference implementations, not reproductions of S4,
Mamba, Spikformer, or an official SEW checkpoint. No external weights are loaded
implicitly. Every temporal owner and the readout accumulator is snapshotable.
"""
from __future__ import annotations
from dataclasses import fields
import hashlib
import math
import torch
from torch import nn
from .reference.models import BenchmarkModel, ModelConfig
from .reference.neurons import Cell
from .specs import BASELINES,LEGACY_REFERENCE_BACKBONES,LEGACY_BASELINES


def tree_map(fn, obj):
    if isinstance(obj,torch.Tensor):return fn(obj)
    if isinstance(obj,dict):return {k:tree_map(fn,v) for k,v in obj.items()}
    if isinstance(obj,list):return [tree_map(fn,v) for v in obj]
    if isinstance(obj,tuple):return tuple(tree_map(fn,v) for v in obj)
    return obj


class MemoryBackbone(nn.Module):
    def __init__(self,cfg,dim,channels=None):
        super().__init__();self.cfg=cfg;self.state={};self.last={};self.trace={};self.capture=False
        w=cfg.width;self.kind=cfg.backbone;self.cache=[]
        if self.kind in ('gru','lstm'):
            cls=nn.GRUCell if self.kind=='gru' else nn.LSTMCell
            self.layers=nn.ModuleList([cls(dim if j==0 else w,w) for j in range(cfg.depth)])
        elif self.kind=='diagonal_ssm':
            self.input=nn.Linear(dim,w);self.raw_decay=nn.Parameter(torch.full((w,),torch.logit(torch.tensor(cfg.beta).clamp(.001,.999)).item()))
            self.mix=nn.Linear(w,w)
        elif self.kind=='causal_attention':
            self.embed=nn.Linear(dim,w);self.qkv=nn.Linear(w,3*w);self.output=nn.Linear(w,w)
        elif self.kind=='convgru':
            self.gates=nn.Conv2d(channels+w,2*w,3,padding=1);self.candidate=nn.Conv2d(channels+w,w,3,padding=1)

    def reset(self,capture=False):self.state={};self.last={};self.trace={};self.cache=[];self.capture=capture
    def reset_cache(self):self.reset(self.capture)

    def forward(self,x):
        if self.cfg.gradient_mode=='notd':self.state={}
        elif self.cfg.gradient_mode=='detach':self.state=tree_map(lambda v:v.detach(),self.state)
        w=self.cfg.width
        if self.kind in ('gru','lstm'):
            h=x.flatten(1)
            for j,layer in enumerate(self.layers):
                zero=torch.zeros(len(x),w,device=x.device,dtype=x.dtype)
                prev=self.state.get('h'+str(j),zero)
                if self.kind=='lstm':
                    h,c=layer(h,(prev,self.state.get('c'+str(j),zero)));self.state['c'+str(j)]=c
                else:h=layer(h,prev)
                self.state['h'+str(j)]=h
        elif self.kind=='diagonal_ssm':
            u=self.input(x.flatten(1));h=torch.sigmoid(self.raw_decay)*self.state.get('h',torch.zeros_like(u))+u
            self.state['h']=h;h=self.mix(h)
        elif self.kind=='causal_attention':
            e=self.embed(x.flatten(1));q,k,v=self.qkv(e).chunk(3,-1)
            ks=torch.cat([self.state.get('keys',k[:,None,:0]).reshape(len(x),-1,w),k[:,None]],1)
            vs=torch.cat([self.state.get('values',v[:,None,:0]).reshape(len(x),-1,w),v[:,None]],1)
            att=torch.softmax(torch.einsum('bd,btd->bt',q,ks)/math.sqrt(w),-1)
            h=self.output(torch.einsum('bt,btd->bd',att,vs))+e
            self.state.update(keys=ks,values=vs)
        else:
            prev=self.state.get('h',x.new_zeros(len(x),w,*x.shape[-2:]))
            z,r=self.gates(torch.cat([x,prev],1)).sigmoid().chunk(2,1)
            candidate=self.candidate(torch.cat([x,r*prev],1)).tanh();h=(1-z)*prev+z*candidate;self.state['h']=h
        self.last={'output':h,**self.state}
        if self.capture:
            # Attention prefix lengths vary; output remains a fixed coordinate.
            for key,value in self.last.items():
                if key not in ('keys','values'):self.trace.setdefault(key,[]).append(value)
        return h

    def step(self,x):return self(x)
    def cache_scalars(self):return 0


class ResearchModel(nn.Module):
    def __init__(self,spec,contract,input_shape):
        super().__init__();self.cfg=spec.model;self.task=spec.task;self.T=contract['steps'];self.image_output=spec.task.objective=='reconstruction'
        self.sequence_output=contract['target_mode']=='sequence_class_index';self.contract=contract
        self.encoder_gate=contract.get('valid_feature_index');self.encoder=nn.Identity();shape=list(input_shape)
        if self.cfg.encoder=='mlp' or (self.cfg.encoder=='auto' and len(shape)==1):
            if len(shape)!=1:raise ValueError('Frame MLP encoder cannot silently flatten image axes')
            self.encoder=nn.Sequential(nn.Linear(shape[0],self.cfg.encoder_width),nn.Tanh());shape=[self.cfg.encoder_width]
        output=contract['output_shape'];self.output_shape=tuple(output)
        cfg=ModelConfig(**{f.name:getattr(self.cfg,f.name) for f in fields(ModelConfig)})
        if self.cfg.backbone not in LEGACY_REFERENCE_BACKBONES+LEGACY_BASELINES:
            from .extended_models import build_extended
            s=shape.copy()
            if self.cfg.push_pull:s[0]*=2
            self.backbone=build_extended(s,cfg)
            self.head=nn.Conv2d(cfg.width,output[0],1) if self.image_output else nn.Linear(cfg.width*(self.T if cfg.readout=='trajectory' else 1),math.prod(output))
        elif self.cfg.backbone in BASELINES:
            if self.cfg.normalization!='none':raise ValueError('Baseline normalization is not implemented')
            s=shape.copy()
            if self.cfg.push_pull:s[0]*=2
            if self.cfg.backbone=='convgru' and len(s)!=3:raise ValueError('ConvGRU requires image input')
            self.backbone=MemoryBackbone(cfg,math.prod(s),s[0])
            self.head=nn.Conv2d(cfg.width,output[0],1) if self.image_output else nn.Linear(cfg.width*(self.T if cfg.readout=='trajectory' else 1),math.prod(output))
        else:
            ref=BenchmarkModel(shape,output if self.image_output else [math.prod(output)],self.T,cfg,spec.task.objective);self.backbone=ref.backbone;self.head=ref.head
        self.last_features=None;self.time=0;self.readout_state={};self.capture=False

    def temporal_owners(self):
        return {name:m for name,m in self.named_modules() if isinstance(m,(Cell,MemoryBackbone))}

    def reset(self,capture=False):
        self.time=0;self.readout_state={};self.last_features=None;self.capture=capture;self.backbone.reset_cache()
        for m in self.temporal_owners().values():
            if isinstance(m,Cell):m.reset();m.capture=capture
            else:m.reset(capture)

    def step(self,x):
        if not isinstance(self.encoder,nn.Identity):
            gate=None if self.encoder_gate is None else x[:,self.encoder_gate:self.encoder_gate+1]
            x=self.encoder(x)
            if gate is not None:x=x*gate  # learned biases cannot turn exact blanks into writes
        if self.cfg.push_pull:x=torch.cat([x.relu(),(-x).relu()],1)
        # Stream position is an exogenous clock, derived from the replay cursor.
        # Removing recurrent memory must not also erase positional encoding.
        self.backbone.position_t=self.time
        f=self.backbone.step(x)
        if self.cfg.observation!='output':f=self.backbone.last_cell.last[self.cfg.observation]
        if not self.image_output and f.ndim>2:f=f.flatten(2).mean(-1)
        r=self.readout_state;kind=self.cfg.readout
        if kind=='trajectory':
            r['trajectory']=r.get('trajectory',[])+[f];a=f
        elif kind=='last':r['last']=f;a=f
        elif kind in ('sum','mean'):
            r['sum']=r.get('sum',torch.zeros_like(f))+f;r['count']=r.get('count',0)+1
            a=r['sum'] if kind=='sum' else r['sum']/r['count']
        else:
            r['weighted']=self.cfg.readout_decay*r.get('weighted',torch.zeros_like(f))+f
            r['weight']=self.cfg.readout_decay*r.get('weight',0.)+1.;a=r['weighted']/r['weight']
        self.time+=1
        return f,a

    def prediction(self):
        r=self.readout_state;k=self.cfg.readout
        if not self.time:raise ValueError('Empty stream')
        if k=='trajectory':
            if self.time!=self.T:raise ValueError('Trajectory head requires full fixed T')
            a=torch.stack(r['trajectory'],1).flatten(1)
        elif k=='last':a=r['last']
        elif k=='sum':a=r['sum']
        elif k=='mean':a=r['sum']/r['count']
        else:a=r['weighted']/r['weight']
        result=self.head(a)
        return result if self.image_output else result.reshape(len(result),*self.output_shape)

    def forward(self,x,capture=False,return_sequence=False,after_step=None):
        self.reset(capture);features=[]
        for t in range(x.shape[1]):
            f,_=self.step(x[:,t]);features.append(f)
            if after_step is not None:after_step(self,t)
        self.last_features=torch.stack(features,1)
        pred=self.prediction()
        if return_sequence or self.sequence_output:
            if self.cfg.readout=='trajectory':raise ValueError('No trajectory per-step classifier')
            fs=self.last_features
            seq=self.head(fs.flatten(0,1)).reshape(len(x),x.shape[1],*pred.shape[1:]) if self.image_output else self.head(fs).reshape(len(x),x.shape[1],*self.output_shape)
            return pred,seq
        return pred

    def traces(self):
        return {name:{k:torch.stack(v,1) for k,v in m.trace.items()} for name,m in self.temporal_owners().items() if m.trace}

    def snapshot(self,cpu=True):
        """After-step snapshot: state, explicit delays, readout, and stream cursor."""
        fn=lambda t:t.detach().clone().cpu() if cpu else t.detach().clone()
        return tree_map(fn,{'schema':'complete_stream_v1','time':self.time,
            'owners':{name:m.state for name,m in self.temporal_owners().items()},
            'event_controls':{name:getattr(m,'event_controls',{}) for name,m in self.temporal_owners().items()},
            'cache':self.backbone.cache,'readout':self.readout_state})

    def restore(self,snapshot):
        if snapshot['schema']!='complete_stream_v1' or set(snapshot['owners'])!=set(self.temporal_owners()):raise ValueError('Snapshot owner mismatch')
        device=next(self.parameters()).device;s=tree_map(lambda t:t.to(device).clone(),snapshot)
        for name,m in self.temporal_owners().items():m.state=s['owners'][name]
        for name,controls in s.get('event_controls',{}).items():
            if isinstance(self.temporal_owners()[name],Cell):self.temporal_owners()[name].force_events(**controls)
        self.backbone.cache=s['cache'];self.readout_state=s['readout'];self.time=s['time']

    def resources(self):
        snap=self.snapshot();numbers=[]
        tree_map(lambda t:numbers.append(t.numel()) or t,snap)
        return {'parameters':sum(p.numel() for p in self.parameters()),'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
            'complete_state_scalars_batch':sum(numbers),'owners':list(self.temporal_owners()),
            'implementation':'dense PyTorch reference; no measured device energy or asynchronous speed claim',
            'official_model_reproduction':False}


def initialize_paired(model,seed):
    """Shared parameter path+shape gets identical initial weights across cells."""
    preserved={prefix+'.'+name if prefix else name for prefix,module in model.named_modules() for name in getattr(module,'preserve_initialization',set())}
    normalized={id(module.weight) for module in model.modules() if isinstance(module,(nn.LayerNorm,nn.GroupNorm)) and module.weight is not None}
    for name,p in model.named_parameters():
        # Respect declared learned decay initialization.
        if name.endswith('raw_beta') or name.endswith('raw_decay') or name in preserved:continue
        key=hashlib.sha256((str(seed)+'|'+name+'|'+str(tuple(p.shape))).encode()).digest()
        g=torch.Generator(device='cpu').manual_seed(int.from_bytes(key[:8],'little')%(2**63-1))
        with torch.no_grad():
            if id(p) in normalized:p.fill_(1.)
            elif p.ndim>=2:
                fan_in,fan_out=nn.init._calculate_fan_in_and_fan_out(p);bound=math.sqrt(6/(fan_in+fan_out))
                p.copy_(torch.empty_like(p,device='cpu').uniform_(-bound,bound,generator=g).to(p.device))
            else:p.zero_()
    return model


def create_model(spec,contract,device='cpu'):
    return initialize_paired(ResearchModel(spec,contract,contract['input_shape']),spec.train.seed).to(device)
