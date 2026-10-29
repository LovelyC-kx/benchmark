"""Actual output-branch reblocking while preserving the restored sender state."""
import numpy as np
import torch
from recovery_runtime44 import scheduled_run

@torch.no_grad()
def run_with_edge_policy(model,x,edits,replacement,window,batch_size=128):
    core=getattr(model,'core',model);x=np.asarray(x);replacement=np.asarray(replacement)
    if core.cfg.backbone!='mlp' or len(core.backbone.cells)!=2:raise ValueError('Declared two-layer feedforward MLP edge')
    if replacement.shape!=(len(x),x.shape[1],core.cfg.width) or not np.isin(replacement,[0,1]).all():raise ValueError('Legal paired binary messages')
    T=x.shape[1];times=set(window)
    if any(not isinstance(t,int) or isinstance(t,bool) or not 0<=t<T for t in times):raise ValueError('Declared edge window')
    clock=[0];sent=[];arrived=[]
    def send(cell,args,out):
        count=clock[0];t=count%T;start=(count//T)*batch_size;clock[0]+=1
        value=torch.as_tensor(replacement[start:start+len(out),t],device=out.device,dtype=out.dtype) if t in times else out
        sent.append(value.detach().cpu().numpy().copy());return value
    def arrival(layer,args):arrived.append(args[0].detach().cpu().numpy().copy())
    h1=core.backbone.cells[0].register_forward_hook(send);h2=core.backbone.layers[1].register_forward_pre_hook(arrival)
    try:r=scheduled_run(model,x,edits,batch_size=batch_size)
    finally:h1.remove();h2.remove()
    assert len(sent)==len(arrived)==T*((len(x)+batch_size-1)//batch_size)
    gap=max(float(np.max(np.abs(a-b))) for a,b in zip(sent,arrived));assert gap==0
    r['edge_arrival_max_abs']=gap;return r
