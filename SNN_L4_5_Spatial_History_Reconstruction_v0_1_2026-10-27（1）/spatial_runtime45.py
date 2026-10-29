"""Ordered post-state hooks on the inherited actual CNN/U-Net cells."""
from dataclasses import dataclass
import numpy as np
import torch
from bridge45 import Cell,parameter_hash

@dataclass
class Edit:
    path:str
    time:int
    operation:object
    label:str
    owner:str='v'

@torch.no_grad()
def run(model,x,edits=(),times=(2,3,5),batch_size=64,sites=None):
    x=np.asarray(x);cells={k:v for k,v in model.named_modules() if isinstance(v,Cell)}
    if x.ndim!=5 or not len(x):raise ValueError('Nonempty BTCHW')
    if len({e.label for e in edits})!=len(edits):raise ValueError('Unique edit labels')
    for e in edits:
        if e.path not in cells or not isinstance(e.time,int) or not 0<=e.time<x.shape[1] or e.owner not in ('v','a','syn'):raise ValueError('Declared cell/time/owner')
    sites=tuple(cells) if sites is None else tuple(sites)
    if not set(sites)<=set(cells):raise ValueError('Known capture sites')
    features=[];pred=[];states={};records={};event_counts={};state_sizes={}
    for start in range(0,len(x),batch_size):
        xb=torch.as_tensor(x[start:start+batch_size],dtype=torch.float32);clocks={k:0 for k in cells};hooks=[]
        def after(path,m,args,out):
            t=clocks[path];clocks[path]+=1
            for e in edits:
                if e.path!=path or e.time!=t:continue
                if e.owner not in m.state:raise ValueError('Absent recurrent owner')
                h=m.state[e.owner];raw=h.cpu().numpy().copy();v=np.asarray(e.operation(raw,start),float)
                if v.shape!=raw.shape or not np.isfinite(v).all():raise ValueError('Finite same-shape edit')
                value=h if np.array_equal(v,raw) else torch.as_tensor(v,dtype=h.dtype).reshape_as(h)
                m.state[e.owner]=value;alias='post' if e.owner=='v' else e.owner;m.last[alias]=value
                records.setdefault(e.label,{'before':[],'after':[]})['before'].append(raw);records[e.label]['after'].append(value.cpu().numpy().copy())
            if path in sites and t in times:
                for owner in ('v','a','syn'):
                    if owner in m.state:states.setdefault(path+'::'+owner+'::'+str(t),[]).append(m.state[owner].cpu().numpy().copy())
            if path in sites:
                events=m.last.get('natural_event')
                if events is not None:event_counts.setdefault(path,[]).append(float(events.sum()))
                state_sizes[path]={k:int(v[0].numel()) for k,v in m.state.items()}
            return out
        for path,m in cells.items():hooks.append(m.register_forward_hook(lambda m,a,o,path=path:after(path,m,a,o)))
        try:p=model(xb,capture=False)
        finally:
            for h in hooks:h.remove()
        if any(v!=x.shape[1] for v in clocks.values()):raise ValueError('One actual cell call per frame')
        features.append(model.last_features.cpu().numpy().copy());pred.append(p.cpu().numpy())
    return {'features':np.concatenate(features),'prediction':np.concatenate(pred),
        'states':{k:np.concatenate(v) for k,v in states.items()},
        'edits':{k:{j:np.concatenate(v) for j,v in row.items()} for k,row in records.items()},
        'captured_cell_state_sizes_per_scene':state_sizes,'natural_event_counts':{k:float(sum(v)) for k,v in event_counts.items()},
        'phase':'After Cell update/current emit, before current observation. Current message is not recomputed.'}

def state(run,path,time,owner='v'):return run['states'][path+'::'+owner+'::'+str(time)].astype(float)
def additive(path,time,delta,label='patch',owner='v'):
    return Edit(path,time,lambda h,start:h+delta[start:start+len(h)],label,owner)

def masked_replace(path,time,reference,mask,label='damage',owner='v'):
    mask=np.asarray(mask,bool)
    return Edit(path,time,lambda h,start:np.where(mask[None,None],reference[start:start+len(h)],h),label,owner)
