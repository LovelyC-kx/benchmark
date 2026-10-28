"""Inference-only streaming, complete runtime snapshots and scoped interventions.

Training still uses BenchmarkModel.forward. This interface deliberately stops
autograd and is not an alternative BPTT implementation. No future inputs are
buffered. Mean/sum/exponential/last need no output trajectory unless requested.
"""
from dataclasses import asdict
import hashlib
import json
import torch
from .neurons import Cell


def clone_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().clone()
    if isinstance(value,dict): return {k:clone_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [clone_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(clone_tree(v) for v in value)
    return value


def fingerprint(model):
    h=hashlib.sha256(json.dumps(asdict(model.cfg),sort_keys=True).encode())
    h.update(str(model.objective).encode())
    static=('kind','mode','threshold','gain','dc_normalize','surrogate','sg_scale',
            'detach_reset','adapt_decay','adapt_gain')
    for name,c in model.named_modules():
        if isinstance(c,Cell):
            h.update(name.encode())
            h.update(json.dumps({k:getattr(c,k) for k in static},sort_keys=True).encode())
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class StreamSession:
    """One active session per model; interleaving requires explicit restore."""
    def __init__(self,model,*,capture=False,retain_trajectory=False):
        self.model=model.eval()
        self.retain_trajectory=retain_trajectory
        if model.cfg.readout_decay<0 or model.cfg.readout_decay>1:
            raise ValueError('streaming exponential decay must lie in [0,1]')
        self.start(capture=capture)

    @property
    def cells(self):
        return {n:m for n,m in self.model.named_modules() if isinstance(m,Cell)}

    def start(self,*,capture=False):
        self.model.reset(capture=capture)
        self.model.last_features=None
        self.cursor=0;self.count=0;self.accumulator=None;self.denominator=0.
        self.last_feature=None;self.features=[];self.input_signature=None
        self.capture=capture

    @torch.no_grad()
    def step(self,xt):
        if self.model.training:raise ValueError('StreamSession is inference-only; model must remain in eval mode')
        if xt.ndim not in (2,4):raise ValueError('step expects [B,F] or [B,C,H,W]')
        signature=(tuple(xt.shape),str(xt.dtype),str(xt.device))
        if self.input_signature is None:self.input_signature=signature
        if signature!=self.input_signature:raise ValueError('batch/input shape, dtype or device changed within a stream')
        if self.model.cfg.readout=='trajectory' and self.cursor>=self.model.T:
            raise ValueError('fixed trajectory head exceeded T')
        if self.model.cfg.push_pull:xt=torch.cat([xt.relu(),(-xt).relu()],dim=1)
        f=self.model.backbone.step(xt)
        if self.model.cfg.observation!='output':
            f=self.model.backbone.last_cell.last[self.model.cfg.observation]
        if not self.model.image_output and f.ndim>2:f=f.flatten(2).mean(-1)
        self.last_feature=f
        mode=self.model.cfg.readout
        if mode in ('mean','sum'):
            self.accumulator=f.clone() if self.accumulator is None else self.accumulator+f
        elif mode=='exponential':
            decay=self.model.cfg.readout_decay
            self.accumulator=f.clone() if self.accumulator is None else decay*self.accumulator+f
            self.denominator=decay*self.denominator+1.
        if mode=='trajectory' or self.retain_trajectory:self.features.append(f.clone())
        self.cursor+=1;self.count+=1
        return f

    @torch.no_grad()
    def finish(self):
        if not self.count:raise ValueError('no observations since start/readout clear')
        mode=self.model.cfg.readout
        if mode=='trajectory':
            if len(self.features)!=self.model.T:raise ValueError('trajectory head requires exactly T slots')
            value=torch.stack(self.features,1).flatten(1)
        elif mode=='last':value=self.last_feature
        elif mode=='mean':value=self.accumulator/self.count
        elif mode=='sum':value=self.accumulator
        else:value=self.accumulator/self.denominator
        return self.model.head(value)

    def run(self,x):
        for t in range(x.shape[1]):self.step(x[:,t])
        return self.finish()

    def trajectory(self):
        if not self.features:raise ValueError('trajectory was not retained')
        return torch.stack(self.features,1)

    def snapshot(self):
        """Trusted in-process inference snapshot, including branch policies and RNG.

        Capture traces are diagnostics and not included; restore clears them.
        Model parameters are verified by fingerprint, not copied into snapshots.
        """
        return clone_tree({
            'schema':'l2-stream-v1','model_fingerprint':fingerprint(self.model),
            'retain_trajectory':self.retain_trajectory,'capture':self.capture,
            'cells':{n:{'state':c.state,'last':c.last,'event_controls':c.event_controls,
                        'reset_strength':c.reset_strength} for n,c in self.cells.items()},
            'cache':self.model.backbone.cache,
            'readout':{'cursor':self.cursor,'count':self.count,'accumulator':self.accumulator,
                       'denominator':self.denominator,'last_feature':self.last_feature,
                       'features':self.features,'input_signature':self.input_signature},
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })

    def restore(self,snapshot):
        if snapshot.get('schema')!='l2-stream-v1':raise ValueError('snapshot schema')
        if snapshot['model_fingerprint']!=fingerprint(self.model):
            raise ValueError('snapshot model/weights/config mismatch')
        if set(snapshot['cells'])!=set(self.cells):raise ValueError('cell paths differ')
        if snapshot['retain_trajectory']!=self.retain_trajectory:
            raise ValueError('snapshot trajectory policy differs')
        value=clone_tree(snapshot)
        for name,c in self.cells.items():
            r=value['cells'][name]
            c.state=r['state'];c.last=r['last'];c.trace={};c.capture=value['capture']
            c.event_controls=r['event_controls'];c.reset_strength=r['reset_strength']
        self.model.backbone.cache=value['cache']
        for key,item in value['readout'].items():setattr(self,key,item)
        self.capture=value['capture'];self.model.last_features=None
        torch.set_rng_state(value['torch_rng'].cpu())
        if value['cuda_rng'] is not None:
            if not torch.cuda.is_available():raise ValueError('CUDA snapshot requires CUDA')
            torch.cuda.set_rng_state_all(value['cuda_rng'])

    def clear(self,scope,*,paths=None):
        """Zero history values while preserving clock and legal tensor shapes.

        'core' includes all cells and backbone cache. 'system' additionally clears
        the readout. It retains model parameters and the absolute stream clock.
        Clear operations do not delete externally installed diagnostic policies.
        """
        valid={'v','aux','cell','cache','core','readout','system'}
        if scope not in valid:raise ValueError('unknown scope')
        selected=set(self.cells) if paths is None else set(paths)
        if selected-set(self.cells):raise ValueError('unknown cell path')
        if paths is not None and scope in ('cache','core','readout','system'):
            raise ValueError('paths are only supported for cell/v/aux scopes')
        if scope in ('v','aux','cell','core','system'):
            for name,c in self.cells.items():
                if name not in selected:continue
                for key,value in list(c.state.items()):
                    if scope in ('cell','core','system') or (scope=='v' and key=='v') or (scope=='aux' and key in ('a','syn')):
                        c.state[key]=torch.zeros_like(value)
        if scope in ('cache','core','system'):
            self.model.backbone.cache=[torch.zeros_like(x) for x in self.model.backbone.cache]
        if scope in ('readout','system'):
            self.accumulator=None;self.denominator=0.;self.last_feature=None;self.count=0
            # Fixed-position head: erase prefix values but preserve absolute slots.
            if self.model.cfg.readout=='trajectory':self.features=[torch.zeros_like(f) for f in self.features]
            else:self.features=[]

    def registry(self,*,include_diagnostics=False):
        rows=[]
        def add(name,x,owner,role,updates):
            rows.append({'path':name,'owner':owner,'role':role,'updates':updates,
                         'shape':list(x.shape),'dtype':str(x.dtype),'scalars':x.numel(),
                         'tensor_bytes':x.numel()*x.element_size(),'device':str(x.device)})
        for name,c in self.cells.items():
            for key,x in c.state.items():add(name+'.state.'+key,x,'cell','core','every Cell call; post-update state')
            for key,x in c.event_controls.items():add(name+'.event_controls.'+key,x,'policy','diagnostic_policy','external intervention')
            if include_diagnostics:
                for key,x in c.last.items():add(name+'.last.'+key,x,'diagnostics','diagnostic','current-step observation; not independently recurrent')
                for key,values in c.trace.items():
                    for i,x in enumerate(values):add(f'{name}.trace.{key}.{i}',x,'diagnostics','diagnostic','retained history for capture only')
        for i,x in enumerate(self.model.backbone.cache):add(f'backbone.cache.{i}',x,'backbone','core','each backbone step')
        if self.accumulator is not None:add('readout.accumulator',self.accumulator,'readout','readout','each consumed observation')
        if self.last_feature is not None:add('readout.last_feature',self.last_feature,'readout','readout','latest observation')
        for i,x in enumerate(self.features):add(f'readout.features.{i}',x,'readout','readout','trajectory retention')
        return {'rows':rows,'cursor':self.cursor,'readout_count':self.count,
                'readout_denominator':self.denominator,
                'logical_tensor_bytes_sum':sum(r['tensor_bytes'] for r in rows),
                'note':'logical role-wise tensor sizes can alias; excludes parameters, autograd and allocator overhead; not peak/hardware memory',
                'scalar_bookkeeping':'cursor/count are Python integers, denominator is a Python float; reported separately'}
