"""Real graph hooks and complete stream replay. Message != recurrent state.

A message hook intercepts one module output before its consumers, including
skip branches. It does not overwrite the emitting cell's recurrent state.
State edits occur after a declared step. Arbitrary edge cuts are only supported
when an actual consumer input is named; no fictitious two-layer graph is used.
"""
from __future__ import annotations
import contextlib
import numpy as np
import torch
from .models import tree_map


def tensor_input(batch,device):return torch.as_tensor(np.array(batch['x'],copy=True),device=device,dtype=torch.float32)


@torch.no_grad()
def collect(model,batch,device='cpu',snapshots=False):
    model.eval();states={}
    def after(m,t):
        if snapshots:states[t]=m.snapshot()
    pred=model(tensor_input(batch,device),capture=True,after_step=after)
    convert=lambda x:x.detach().cpu().numpy()
    return {'prediction':tree_map(convert,pred),'features':convert(model.last_features),
            'traces':tree_map(convert,model.traces()),'snapshots':states,'resources':model.resources()}


@contextlib.contextmanager
def message_edit(model,site,time,replacement,kind='message'):
    """Replacement is a tensor, or callable(tensor)->tensor. Exact graph phase."""
    modules=dict(model.named_modules())
    if site not in modules:raise ValueError('Unknown module output/consumer input: '+site)
    def apply(value):
        if model.time!=time:return value
        result=replacement(value) if callable(replacement) else replacement.to(value)
        if not isinstance(result,torch.Tensor) or result.shape!=value.shape:raise ValueError('Intervention shape mismatch')
        return result
    if kind=='message':handle=modules[site].register_forward_hook(lambda m,args,out:apply(out))
    elif kind=='edge':
        def pre(m,args):
            if len(args)!=1 or not isinstance(args[0],torch.Tensor):raise ValueError('Only single-tensor consumer edges are supported')
            return (apply(args[0]),)
        handle=modules[site].register_forward_pre_hook(pre)
    else:raise ValueError('message or edge')
    try:yield
    finally:handle.remove()


@contextlib.contextmanager
def event_edit(model,site,time,branch,value=0):
    """Override emission/reset/auxiliary event independently for exactly one step."""
    from .models import Cell
    cell=model.temporal_owners().get(site)
    if not isinstance(cell,Cell) or cell.kind in ('continuous','tanh'):raise ValueError('An actual event-based cell is required')
    if branch not in ('emit','reset','aux') or value not in (0,1):raise ValueError('Binary event branch control')
    def pre(module,args):
        if model.time==time:module.force_events(**{branch:torch.full_like(args[0],value)})
    def post(module,args,out):
        if model.time==time:module.force_events()
    first=cell.register_forward_pre_hook(pre);second=cell.register_forward_hook(post)
    try:yield
    finally:first.remove();second.remove();cell.force_events()


@torch.no_grad()
def intervene(model,batch,site,time,donor=None,mode='zero',edit_state=True,edit_message=False,device='cpu'):
    """A paired state/message factorial arm with exactly declared step and site."""
    owners=model.temporal_owners()
    if site not in owners:raise ValueError('State interventions require a temporal owner')
    if not 0<=time<batch['x'].shape[1]:raise ValueError('Intervention time')
    if mode not in ('zero','donor'):raise ValueError('zero or donor')
    if mode=='donor' and donor is None:raise ValueError('Donor capture required')
    def change(value,key=None):
        if mode=='zero':return torch.zeros_like(value)
        if key is None:replacement=donor['traces'][site]['output'][:,time]
        else:replacement=donor['snapshots'][time]['owners'][site][key]
        return torch.as_tensor(replacement,device=value.device,dtype=value.dtype)
    def after(m,t):
        if edit_state and t==time:
            owners[site].state={k:change(v,k) for k,v in owners[site].state.items()}
    ctx=message_edit(model,site,time,change) if edit_message else contextlib.nullcontext()
    model.eval()
    with ctx:pred=model(tensor_input(batch,device),after_step=after)
    return tree_map(lambda x:x.detach().cpu().numpy(),pred)


@torch.no_grad()
def replay_suffix(model,batch,snapshot,device='cpu'):
    model.eval();model.reset();model.restore(snapshot)
    x=tensor_input(batch,device)
    for t in range(model.time,x.shape[1]):model.step(x[:,t])
    return model.prediction().detach().cpu().numpy()
