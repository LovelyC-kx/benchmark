"""Observation and access ladders with explicit independently trained readers."""
import numpy as np
import torch
from torch import nn
from .utils import seeded,atomic_torch,digest,file_hash
from pathlib import Path
from .engine import native_losses


def compress(xs,limit,name,return_matrix=False):
    d=xs[0].shape[1]
    if d<=limit:
        result=(xs,{'original_dimension':d,'dimension':d,'compression':'none'})
        return (*result,None) if return_matrix else result
    # Label-independent fixed random projection; it is a bounded-budget empirical control.
    matrix=seeded(271,'view_projection',name,d,limit).choice([-1.,1.],size=(d,limit)).astype('float32')/np.sqrt(limit)
    result=([x@matrix for x in xs],{'original_dimension':d,'dimension':limit,'compression':'fixed Rademacher projection; not an optimal full-observation reader'})
    return (*result,matrix) if return_matrix else result


def train_reader(xs,batches,a,name,artifact_dir=None):
    xs,obs,projection=compress([np.asarray(x,'float32') for x in xs],max(a.max_features,8),name,True)
    mean=xs[0].mean(0);scale=np.maximum(xs[0].std(0),1e-6);x,v,e=[torch.tensor((z-mean)/scale,dtype=torch.float32) for z in xs]
    yshape=batches[0]['meta']['output_shape'];steps=max(10,a.mlp_steps);torch.manual_seed(924)
    shape=tuple(yshape);outdim=int(np.prod(shape));head=nn.Linear(x.shape[1],outdim);opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    if batches[0]['meta']['target_mode']=='sequence_class_index':return {'status':'not_applicable','reason':'Sequence-aligned reader needs a per-position output protocol'}
    best=None;bestloss=float('inf');stepbest=0
    for step in range(steps):
        opt.zero_grad();p=head(x).reshape(len(x),*shape);loss,_=native_losses(p,batches[0]);loss.mean().backward();opt.step()
        with torch.no_grad():vl=float(native_losses(head(v).reshape(len(v),*shape),batches[1])[0].mean())
        if vl<bestloss:bestloss=vl;best={k:z.detach().clone() for k,z in head.state_dict().items()};stepbest=step+1
    head.load_state_dict(best)
    with torch.no_grad():loss,correct=native_losses(head(e).reshape(len(e),*shape),batches[2])
    result={'status':'computed','native_loss':float(loss.mean()),'accuracy':None if correct is None else float(correct.mean()),
        'selection_loss':bestloss,'selected_updates':stepbest,'max_updates':steps,'parameters':sum(p.numel() for p in head.parameters()),
        'observation':obs,'scope':'empirical linear reader optimized on probe_train and selected on probe_val; not the native trained head or a Bayes bound'}
    if artifact_dir is not None:
        path=Path(artifact_dir)/(digest(name)[:20]+'.pt');atomic_torch(path,{'schema':'fitted_native_reader_v1','view':name,'model':head.state_dict(),
            'input_mean':torch.from_numpy(mean),'input_scale':torch.from_numpy(scale),'projection':None if projection is None else torch.from_numpy(projection),
            'output_shape':shape,'selected_updates':stepbest,'selection_loss':bestloss,'observation':obs})
        result['fitted_reader']={'file':str(path),'sha256':file_hash(path)}
    return result


def task_controls(data,a,artifact_dir=None):
    splits=('probe_train','probe_val',a.eval_split);batches=[data[k] for k in splits];xs=[np.asarray(b['x']) for b in batches];rows={}
    views={'current':[x[:,-1].reshape(len(x),-1) for x in xs],
           'recent4':[x[:,-min(4,x.shape[1]):].reshape(len(x),-1) for x in xs],
           'unordered_sum':[x.sum(1).reshape(len(x),-1) for x in xs],
           'ordered_trajectory':[x.reshape(len(x),-1) for x in xs]}
    for name,values in views.items():rows[name]=train_reader(values,batches,a,'input_'+name,artifact_dir)
    return {'views':rows,'purpose':'distinguish empirical history demand, order demand and static shortcuts under a fixed reader budget',
        'warning':'A failed bounded reader cannot certify that its observation is insufficient; unordered sum is one summary of a set, not every permutation-invariant function'}


def access_controls(groups,data,a,artifact_dir=None):
    splits=('probe_train','probe_val',a.eval_split);batches=[data[k] for k in splits];keys=[k for k in groups[splits[0]]['traces'] if k.endswith('::output')]
    if not keys:return {'status':'not_applicable','reason':'No declared temporal output'}
    last=keys[-1];rows={};xs=[groups[k]['traces'][last] for k in splits]
    for name,values in {'last':[x[:,-1] for x in xs],'mean':[x.mean(1) for x in xs],'sum':[x.sum(1) for x in xs],'trajectory':[x.reshape(len(x),-1) for x in xs]}.items():
        rows[last+'|'+name]=train_reader(values,batches,a,'access_'+name,artifact_dir)
    for channel in ('pre','post','a','syn'):
        candidates=[k for k in groups[splits[0]]['traces'] if k.endswith('::'+channel)]
        if candidates:
            key=candidates[-1];rows[key+'|last']=train_reader([groups[s]['traces'][key][:,-1] for s in splits],batches,a,'access_'+channel,artifact_dir)
    statekeys=[k for k in groups[splits[0]]['traces'] if k.endswith(('::post','::a','::syn','::h0','::c0','::h'))]
    if statekeys:rows['all_captured_owners|last']=train_reader([np.concatenate([groups[s]['traces'][k][:,-1] for k in statekeys],1) for s in splits],batches,a,'all_captured_owners',artifact_dir)
    return {'status':'computed','readers':rows,'scope':'access/readout interventions with new fitted readers; empirical scores need not form monotone risk gaps',
        'complete_state_claim':False,'reason':'explicit caches, attention prefixes and feature-compressed coordinates are not silently counted as full state'}
