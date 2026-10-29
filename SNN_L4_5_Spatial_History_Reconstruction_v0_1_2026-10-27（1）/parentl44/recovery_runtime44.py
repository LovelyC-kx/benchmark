"""Multiple ordered real-state edits; optional explicit fixed-event diagnostic."""
from dataclasses import dataclass
import numpy as np
import torch
from bridge44 import Cell

@dataclass
class Edit:
    path:str
    time:int
    operation:object
    label:str
    keys:tuple=('v',)

@torch.no_grad()
def scheduled_run(model,x,edits=(),batch_size=128,keep_traces=True,fixed_events=None):
    core=getattr(model,'core',model);cells={n:c for n,c in core.named_modules() if isinstance(c,Cell)}
    x=np.asarray(x);T=x.shape[1]
    if len({e.label for e in edits})!=len(edits):raise ValueError('Distinct edit labels')
    for e in edits:
        if e.path not in cells or not isinstance(e.time,int) or isinstance(e.time,bool) or not 0<=e.time<T:raise ValueError('Declared cell and time')
        if not e.keys or len(set(e.keys))!=len(e.keys) or set(e.keys)-{'v','a','syn'}:raise ValueError('Explicit recurrent owners')
    if fixed_events is not None:
        if set(fixed_events)!=set(cells):raise ValueError('Fixed-event policy must declare all cells')
        if any(not np.isin(v,[0,1]).all() for v in fixed_events.values()):raise ValueError('Binary fixed-event policy')
    model.eval();device=next(model.parameters()).device;predictions=[];traces={};records={}
    for start in range(0,len(x),batch_size):
        xb=torch.as_tensor(x[start:start+batch_size],device=device);clocks={p:0 for p in cells};hooks=[]
        def before(path,cell,args):
            if fixed_events is not None:
                raw=fixed_events[path][start:start+len(xb),clocks[path]]
                value=torch.as_tensor(raw,device=args[0].device,dtype=args[0].dtype).reshape_as(args[0])
                cell.force_events(emit=value,reset=value,aux=value)
        def after(path,cell,args,out):
            t=clocks[path];clocks[path]+=1
            for e in edits:
                if e.path!=path or e.time!=t:continue
                if any(k not in cell.state for k in e.keys):raise ValueError('Requested owner absent')
                parts=[cell.state[k] for k in e.keys];widths=[h[0].numel() for h in parts]
                raw=np.concatenate([h.detach().cpu().numpy().reshape(len(h),-1) for h in parts],axis=1).copy()
                edited=np.asarray(e.operation(raw,start),float)
                if edited.shape!=raw.shape or not np.isfinite(edited).all():raise ValueError('Finite same-shape state edit')
                values=[];offset=0
                for key,h,width in zip(e.keys,parts,widths):
                    region=edited[:,offset:offset+width];old=raw[:,offset:offset+width]
                    value=h if np.array_equal(region,old) else torch.as_tensor(region,device=h.device,dtype=h.dtype).reshape_as(h)
                    cell.state[key]=value;alias='post' if key=='v' else key;cell.last[alias]=value
                    if cell.capture:cell.trace[alias][-1]=value
                    values.append(value.cpu().numpy().reshape(len(h),-1).copy());offset+=width
                row=records.setdefault(e.label,{'before':[],'after':[]});row['before'].append(raw);row['after'].append(np.concatenate(values,axis=1))
            return out
        for path,cell in cells.items():
            hooks.append(cell.register_forward_pre_hook(lambda m,args,path=path:before(path,m,args)))
            hooks.append(cell.register_forward_hook(lambda m,args,out,path=path:after(path,m,args,out)))
        try:prediction=model(xb,capture=keep_traces)
        finally:
            for hook in hooks:hook.remove()
            for cell in cells.values():cell.event_controls={}
        if any(v!=T for v in clocks.values()):raise ValueError('One call per declared cell per step required')
        predictions.append(prediction.cpu().numpy())
        if keep_traces:
            for path,trace in core.traces().items():
                for key in ('post','natural_event','emit_event','reset_event','a','syn'):
                    if key in trace:traces.setdefault(path+'/'+key,[]).append(trace[key].cpu().numpy().copy())
    return {'prediction':np.concatenate(predictions),
        'edits':{name:{k:np.concatenate(v) for k,v in row.items()} for name,row in records.items()},
        'traces':{k:np.concatenate(v) for k,v in traces.items()},
        'event_policy':'all emit/reset/aux branches replay clean events' if fixed_events is not None else 'natural',
        'phase':'Ordered edits after Cell state update, before observation; current emitted output is not recomputed.'}

def trace_site(run,layer,time,key='post'):
    a=run['traces'][f'backbone.cells.{layer}/'+key]
    return a[:,time].reshape(len(a),-1).astype(float)

def damage_edit(context,eraser):
    from bridge44 import apply_eraser
    return Edit(f"backbone.cells.{context['layer']}",context['time'],lambda h,start:apply_eraser(h,eraser),'damage')

def additive_edit(layer,time,delta,label='patch'):
    return Edit(f'backbone.cells.{layer}',time,lambda h,start:h+delta[start:start+len(h)],label)
