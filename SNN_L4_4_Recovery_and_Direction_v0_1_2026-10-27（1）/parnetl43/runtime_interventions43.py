"""Post-storage branch operations before G, and explicit after-step controls."""
import numpy as np
import torch
from bridge43 import Cell,StreamSession,clone_tree

@torch.no_grad()
def run_post_edit(model,x,path,time,operation,batch_size=128,keep_traces=True,state_keys=('v',)):
    core=getattr(model,'core',model);cells={n:c for n,c in core.named_modules() if isinstance(c,Cell)}
    if path not in cells or not isinstance(time,int) or not 0<=time<x.shape[1]:raise ValueError('Declared cell and within-sequence zero-based time')
    if not state_keys or len(set(state_keys))!=len(state_keys) or set(state_keys)-{'v','a','syn'}:raise ValueError('Ordered unique explicit recurrent state keys')
    pred=[];before=[];after=[];traces={};dev=next(model.parameters()).device
    for start in range(0,len(x),batch_size):
        xb=torch.as_tensor(x[start:start+batch_size],device=dev);clock=[0];seen=[False]
        def hook(cell,args,out):
            t=clock[0];clock[0]+=1
            if t==time:
                if any(k not in cell.state for k in state_keys):raise ValueError('Requested recurrent owner is not present in this cell')
                parts=[cell.state[k] for k in state_keys];widths=[h[0].numel() for h in parts]
                raw=np.concatenate([h.detach().cpu().numpy().reshape(len(h),-1) for h in parts],axis=1).copy()
                edited=np.asarray(operation(raw,start),float)
                if edited.shape!=raw.shape or not np.isfinite(edited).all():raise ValueError('Finite same-shape post-storage operation')
                # A value-identical operation must also preserve tensor layout.
                # Rebuilding a non-contiguous attention state can change the
                # subsequent reduction kernel even though every value matches.
                values=[];offset=0
                for k,h,width in zip(state_keys,parts,widths):
                    region=edited[:,offset:offset+width];original=raw[:,offset:offset+width]
                    value=h if np.array_equal(region,original) else torch.as_tensor(region,device=h.device,dtype=h.dtype).reshape_as(h)
                    cell.state[k]=value;alias='post' if k=='v' else k;cell.last[alias]=value
                    if cell.capture:cell.trace[alias][-1]=value
                    values.append(value.cpu().numpy().reshape(len(h),-1).copy());offset+=width
                before.append(raw);after.append(np.concatenate(values,axis=1));seen[0]=True
            # The emitted output branch has already been computed and stays as returned.
            return out
        handle=cells[path].register_forward_hook(hook)
        try:y=model(xb,capture=keep_traces)
        finally:handle.remove()
        if clock[0]!=x.shape[1] or not seen[0]:raise ValueError('Adapter requires one call to this Cell per observed time step')
        pred.append(y.cpu().numpy())
        if keep_traces:
            for name,trace in core.traces().items():
                for key in ('post','pre','theta','effective_current','natural_event','a','syn'):
                    if key in trace:traces.setdefault(name+'/'+key,[]).append(trace[key].cpu().numpy().astype(float))
    return {'prediction':np.concatenate(pred),'site_before':np.concatenate(before),'site_after':np.concatenate(after),
        'traces':{k:np.concatenate(v) for k,v in traces.items()},
        'state_keys':list(state_keys),
        'phase':'After Cell state/aux update, before G consumes post. Only declared recurrent owners are edited; current pre/emission is not recomputed.'}

@torch.no_grad()
def after_step_edit(model,x,path,time,operation):
    core=getattr(model,'core',model)
    encoded=model.encode(torch.as_tensor(x)) if hasattr(model,'encode') else torch.as_tensor(x)
    session=StreamSession(core)
    for t in range(time+1):session.step(encoded[:,t])
    cell=session.cells[path];h=cell.state['v'];edited=operation(h.cpu().numpy().reshape(len(h),-1),0)
    cell.state['v']=torch.as_tensor(edited,device=h.device,dtype=h.dtype).reshape_as(h)
    # G/readout has already consumed this step. Do not silently rewrite it.
    for t in range(time+1,encoded.shape[1]):session.step(encoded[:,t])
    return session.finish().cpu().numpy()

@torch.no_grad()
def full_snapshot_donor(model,recipient,donor,time):
    a=model.encode(torch.as_tensor(recipient));b=model.encode(torch.as_tensor(donor))
    if not torch.equal(a[:,time+1:],b[:,time+1:]):raise ValueError('Full donor replay requires common future inputs')
    session=StreamSession(model.core)
    for t in range(time+1):session.step(b[:,t])
    snap=session.snapshot()
    for t in range(time+1,b.shape[1]):session.step(b[:,t])
    direct=session.finish().cpu().numpy()
    recipient_session=StreamSession(model.core)
    for t in range(time+1):recipient_session.step(a[:,t])
    recipient_session.restore(snap)
    for t in range(time+1,a.shape[1]):recipient_session.step(a[:,t])
    replay=recipient_session.finish().cpu().numpy()
    return {'prediction':replay,'direct_donor':direct,'max_abs':float(np.max(np.abs(replay-direct))),
        'scope':'Whole operational snapshot includes all cell states, backbone cache and readout; no trained weights changed.'}
