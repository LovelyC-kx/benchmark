"""Actual two-layer MLP edge, including arrival phase and sender-only overrides.

Other backbones require graph-specific adapters; no inferred layer-name mapping.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from bridge35 import Cell
from finite_messages import admissible_binary

WANTED=('pre','post','previous_v','base_current','effective_current','theta',
        'previous_a','a','previous_syn','syn','natural_event','emit_event','reset_event','aux_event')


def core_model(model):
    core=getattr(model,'core',model)
    if core.cfg.backbone!='mlp' or len(core.backbone.cells)!=2:raise ValueError('This adapter requires the declared two-layer feedforward MLP')
    return core


@torch.no_grad()
def capture_edge(model,x,replacement=None,window=None,batch_size=128):
    core=core_model(model);model.eval();device=next(model.parameters()).device
    x=np.asarray(x);n,t=x.shape[:2];binary=core.backbone.cells[0].kind not in ('continuous','tanh','analog_hard_ref')
    if replacement is not None:
        replacement=np.asarray(replacement)
        expected=(n,t,core.cfg.width)
        if replacement.shape!=expected or not np.isfinite(replacement).all():raise ValueError('Paired whole-trajectory edge replacement')
        if binary:admissible_binary(replacement)
        if window is None:raise ValueError('Explicit override time window required')
        if any(isinstance(k,bool) or int(k)!=k or not 0<=k<t for k in window):raise ValueError('Legal within-trajectory override times')
    times=set(window or ());all_values={};predictions=[]
    for start in range(0,n,batch_size):
        xb=torch.as_tensor(x[start:start+batch_size],device=device);values={};counters={'sender':0,'receiver':0}
        def record(name,value):values.setdefault(name,[]).append(value.detach().cpu().numpy().reshape(len(xb),-1).copy())
        def sender_hook(cell,args,out):
            time=counters['sender'];counters['sender']+=1
            for key in WANTED:
                if key in cell.last:record('sender_'+key,cell.last[key])
            result=out
            if replacement is not None and time in times:
                result=torch.as_tensor(replacement[start:start+len(xb),time],device=out.device,dtype=out.dtype)
            record('sender_output',result)
            return result
        def arrival_hook(layer,args):record('receiver_affine_input',args[0])
        def receiver_hook(cell,args,out):
            counters['receiver']+=1
            for key in WANTED:
                if key in cell.last:record('receiver_'+key,cell.last[key])
            record('receiver_drive',args[0]);record('receiver_output',out)
            record('receiver_beta',cell.beta_value(args[0]).expand_as(args[0]))
        hooks=[core.backbone.cells[0].register_forward_hook(sender_hook),
               core.backbone.layers[1].register_forward_pre_hook(arrival_hook),
               core.backbone.cells[1].register_forward_hook(receiver_hook)]
        try:pred=model(xb,capture=False)
        finally:
            for hook in hooks:hook.remove()
        if counters!={'sender':t,'receiver':t}:raise AssertionError('Each declared node must execute once per step')
        predictions.append(pred.cpu().numpy())
        for key,rows in values.items():all_values.setdefault(key,[]).append(np.stack(rows,axis=1))
    arrays={key:np.concatenate(rows) for key,rows in all_values.items()};arrays['native_prediction']=np.concatenate(predictions)
    if not np.array_equal(arrays['sender_output'],arrays['receiver_affine_input']):raise AssertionError('Observed edge arrival differs from returned sender output')
    return arrays


def edge_operator(model):
    core=core_model(model);layer=core.backbone.layers[1];cell=core.backbone.cells[1]
    w=layer.weight.detach().cpu().numpy().copy();b=layer.bias.detach().cpu().numpy().copy()
    return {'weight':w,'bias':b,'gain':cell.gain,'dc_normalize':cell.dc_normalize,
            'sender_kind':core.backbone.cells[0].kind,'receiver_kind':cell.kind,
            'transmitted_carrier':'returned Cell.output, not post membrane or necessarily natural_event',
            'source_time_to_receiver_time':'same Python step; layer-topological precedence',
            'native_head_observation':core.cfg.observation,'native_head_time_readout':core.cfg.readout,
            'graph_scope':'two-layer feedforward MLP; no other input into receiver affine layer'}


def structural_checks(capture,operator):
    x=capture['sender_output'];expected=(x@operator['weight'].T+operator['bias'])*operator['gain']
    if operator['dc_normalize']:expected=expected*(1-capture['receiver_beta'])
    pre=capture['receiver_beta']*capture['receiver_previous_v']+capture['receiver_effective_current']
    return {'actual_edge_arrival_max_abs':float(np.max(np.abs(x-capture['receiver_affine_input']))),
            'current_equation_max_abs':float(np.max(np.abs(expected-capture['receiver_base_current']))),
            'pre_equation_max_abs':float(np.max(np.abs(pre-capture['receiver_pre']))),
            'all_finite':all(np.isfinite(v).all() for v in capture.values())}


def window_view(x,length=4,kind='raster'):
    x=np.asarray(x);n,t,d=x.shape
    if kind not in ('raster','count') or length<1:raise ValueError('Declared causal window')
    padded=np.pad(x,((0,0),(length-1,0),(0,0)))
    windows=np.stack([padded[:,j:j+length] for j in range(t)],axis=1)
    return windows.reshape(n,t,length*d) if kind=='raster' else windows.sum(2)


def chain_views(c):
    out={name:c[name] for name in ('sender_pre','sender_post','sender_output','receiver_base_current',
        'receiver_pre','receiver_post','receiver_output','receiver_previous_v')}
    out['sender_raster4']=window_view(c['sender_output'],4)
    out['sender_count4']=window_view(c['sender_output'],4,'count')
    out['receiver_raster4']=window_view(c['receiver_output'],4)
    out['receiver_post_current']=np.concatenate([c['receiver_post'],c['receiver_base_current']],axis=-1)
    return out


def target_bank(z):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]%5 or z.shape[1]//5 not in (1,2,4):raise ValueError('Five sources with 1/2/4 visible coordinates')
    q=z.shape[1]//5;rng=np.random.default_rng(360904)
    theta=rng.normal(size=(5,q,16));theta/=np.linalg.norm(theta,axis=1,keepdims=True)
    binary=np.concatenate([np.where(z[:,q*j:q*j+q]@theta[j]>=0,1.,-1.) for j in range(5)],axis=1)
    groups=[{'source':name,'continuous_columns':list(range(q*j,q*j+q)),
             'binary_columns':list(range(5*q+16*j,5*q+16*j+16)),
             'role':'original_supervised' if name in ('H1','C') else 'original_unrequested'}
            for j,name in enumerate(('H1','H2','H3','H4','C'))]
    return np.c_[z,binary],theta,groups
