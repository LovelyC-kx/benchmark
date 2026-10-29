"""Audit actual G/P/R and bounded streaming state without changing parameters."""
import hashlib
import numpy as np
import torch
from bridge35 import StreamSession
from access_operators import aggregate,spatial_access

def parameter_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def manual_head(model,value):
    w=model.head.weight.detach().cpu().numpy();b=model.head.bias.detach().cpu().numpy()
    if w.ndim==2:return np.asarray(value)@w.T+b
    if w.ndim==4 and w.shape[-2:]==(1,1):return np.einsum('nchw,kc->nkhw',value,w[:,:,0,0])+b[None,:,None,None]
    raise ValueError('Explicit linear or 1x1 image head adapter required')

def registry_with_aliases(session):
    registry=session.registry();tensors=[]
    for c in session.cells.values():tensors.extend(c.state.values())
    tensors.extend(session.model.backbone.cache)
    if session.accumulator is not None:tensors.append(session.accumulator)
    if session.last_feature is not None:tensors.append(session.last_feature)
    tensors.extend(session.features)
    seen={}
    for value in tensors:
        storage=value.untyped_storage();key=(str(value.device),storage.data_ptr())
        seen[key]=max(seen.get(key,0),storage.nbytes())
    registry['unique_storage_bytes_within_registered_operational_tensors']=sum(seen.values())
    registry['unique_storage_scope']='Only registered operational tensors; excludes cell.last diagnostics, traces, parameters, autograd, allocator overhead and peak/device power.'
    return registry

@torch.no_grad()
def audit_model(model,x):
    """BenchmarkModel (not wrapper) and one same-size minibatch."""
    model.eval();before=parameter_hash(model);x=torch.as_tensor(x,dtype=next(model.parameters()).dtype,device=next(model.parameters()).device)
    pred=model(x,capture=True).detach().cpu().numpy();features=model.last_features.detach().cpu().numpy().copy()
    path=next(n for n,m in model.named_modules() if m is model.backbone.last_cell)
    trace=model.traces()[path];phase=model.cfg.observation
    raw=trace[phase].detach().cpu().numpy()
    expected_features=spatial_access(raw,'reconstruction' if model.image_output else 'classification')
    assert expected_features.shape==features.shape
    feature_gap=float(np.max(np.abs(expected_features-features)))
    manual=manual_head(model,aggregate(features,model.cfg.readout,model.cfg.readout_decay))
    manual_gap=float(np.max(np.abs(manual-pred)))
    stream=StreamSession(model,capture=False,retain_trajectory=False)
    streamed=stream.run(x).detach().cpu().numpy();stream_gap=float(np.max(np.abs(streamed-pred)))
    registry=registry_with_aliases(stream)
    assert parameter_hash(model)==before
    return {'phase':phase,'readout':model.cfg.readout,'cell_module':path,
            'raw_carrier_shape':list(raw.shape),'deployed_feature_shape':list(features.shape),
            'spatial_coordinates_pooled_for_classification':raw.ndim>3 and not model.image_output,
            'carrier_to_features_max_abs':feature_gap,'manual_GPR_prediction_max_abs':manual_gap,
            'stream_vs_batch_prediction_max_abs':stream_gap,
            'batch_forward_retained_feature_stack_bytes':int(features.nbytes),
            'stream_registry':registry,'parameters_unchanged':True,
            'scope':'Numerical G/P/R/streaming consistency, not a trained performance comparison.'}
