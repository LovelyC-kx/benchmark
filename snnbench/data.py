"""Counter-based controlled tasks and read-only canonical data stores.

Only x is model input. Targets, noise, source identity and pairing are analysis data.
Legacy v1 files are recovered unchanged; this wrapper is a separately versioned
counter-per-scene sampler, so increasing/chunking N never changes an earlier scene.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import socket
import time
import numpy as np
from .specs import TaskSpec, IMAGE_TASKS
from .utils import seeded, digest, atomic_json, file_hash

SPLITS=('train','val','calibration','probe_train','probe_val','verification_train','verification_val','development','confirmation','audit')


def requested_sources(task):
    h=task.history_sources
    if task.name=='tagged_current':return [h]
    if task.name=='tagged_history':return [0]
    if task.name=='tagged_joint':return [0,h]
    return list(range(task.required_history))+[h]


def projection(task):
    d=task.sensory_dim
    matrix=np.stack([seeded(task.projection_seed,'projection_column',j).normal(size=task.sensory_dim) for j in range(d)],axis=1)
    return np.linalg.qr(matrix)[0]


def render_tagged(task,z,noise,times,present):
    n,s,k=z.shape;active=z.copy()
    visible_k=k if task.content_mode=='demand_only' else task.k
    active[:,:,visible_k:]=0
    scale=1. if task.energy_mode=='raw' else np.sqrt((k+task.sensory_dim*task.sigma**2)/(visible_k+task.sensory_dim*task.sigma**2))
    obs=scale*(active@projection(task).T+noise)
    x=np.zeros((n,task.T,task.sensory_dim+s+1),dtype='float32')
    for j in range(s):
        rows=np.flatnonzero(present[:,j]);tt=times[rows,j]
        x[rows,tt,:task.sensory_dim]=obs[rows,j]
        x[rows,tt,task.sensory_dim+j]=1;x[rows,tt,-1]=1
    return x


def _tagged_targets(task,z):
    n,s,k=z.shape
    if task.objective=='classification':
        theta=np.stack([seeded(401,'direction',j).normal(size=task.k) for j in range(s)])
        theta/=np.linalg.norm(theta,axis=1,keepdims=True)
        y=(np.einsum('nsk,sk->ns',z[:,:,:task.k],theta)>=0).astype('float32')
        mask=np.zeros((n,s),'float32');mask[:,requested_sources(task)]=1
    elif task.objective=='regression':
        y=z.reshape(n,-1).astype('float32');m=np.zeros((s,k),'float32');m[requested_sources(task),:task.k]=1
        mask=np.broadcast_to(m.reshape(1,-1),y.shape).copy()
    else:raise ValueError('Tagged objective is classification or vector regression')
    return y,mask


def tagged_batch(task,ids,split):
    n=len(ids);s=task.history_sources+1;k=task.sensory_dim;ct=max(12,3*task.history_sources)
    if ct+task.blank_delay>=task.T:raise ValueError('Current source and blank delay exceed T')
    z=np.empty((n,s,k));noise=np.empty((n,s,task.sensory_dim));times=np.empty((n,s),int);present=np.ones((n,s),bool)
    for i,sid in enumerate(ids):
        g=seeded(task.data_seed,split,int(sid),'world');z[i]=g.normal(size=(s,k));noise[i]=g.normal(0,task.sigma,(s,task.sensory_dim))
        slots=2+3*np.arange(task.history_sources);times[i,:-1]=slots
        if task.history_sources>1:times[i,1:-1]=g.permutation(slots[1:])
        times[i,-1]=ct
        if task.extra_writes>=0:
            present[i,1:-1]=False
            chosen=g.permutation(np.arange(1,s-1))[:task.extra_writes];present[i,chosen]=True
    y,mask=_tagged_targets(task,z)
    names=[f'H{j+1}' for j in range(s-1)]+['C']
    meta={'target_mode':'multilabel' if task.objective=='classification' else 'continuous',
          'classes':s if task.objective=='classification' else None,'source_names':names,'source_dims':[k]*s,
          'requested_sources':[names[j] for j in requested_sources(task)],'independent_factors':True,
          'requested_coordinate_indices':{name:(list(range(task.k)) if j in requested_sources(task) else []) for j,name in enumerate(names)},
          'physical_coordinate_indices':{name:list(range(k if task.content_mode=='demand_only' else task.k)) for name in names},
          'model_fields':['x'],'valid_feature_index':task.sensory_dim+s,'time_semantics':'discrete_observation',
          'availability':'source_times with source_present; source truth never input',
          'generator':'tagged_counter_v2','output_shape':[s] if task.objective=='classification' else [s*k]}
    return {'x':render_tagged(task,z,noise,times,present),'y':y,'loss_mask':mask,
            'targets':z.reshape(n,-1).astype('float32'),'z':z.astype('float32'),'noise':noise.astype('float32'),
            'source_times':times,'source_present':present,'meta':meta}


def _spatial_render(z,size):
    h=size//2;yy,xx=np.mgrid[:h,:h];basis=np.stack([np.ones_like(xx),np.cos(np.pi*(xx+.5)/h),np.cos(np.pi*(yy+.5)/h),(-1.)**(xx+yy)])
    spectral=np.array([[1,.7,.8,.9],[.9,1,.6,.7],[.7,.8,1,.6],[.8,.6,.9,1.]])
    tiles=.5+.07*np.einsum('nqk,ck,khw->nqchw',z,spectral,basis)
    img=np.empty((len(z),4,size,size),'float32')
    for j,(a,b) in enumerate(((0,0),(0,h),(h,0),(h,h))):img[:,:,a:a+h,b:b+h]=tiles[:,j]
    return img


def spatial_batch(task,ids,split):
    n=len(ids);h=task.image_size;half=h//2;z=[];old=[];noise=[]
    for sid in ids:
        g=seeded(task.data_seed,split,int(sid),'spatial');z.append(g.uniform(-1,1,(4,4)));old.append(g.uniform(-1,1,(2,4)));noise.append(g.normal(0,task.sigma,(6,4,h,h)))
    z=np.array(z);old=np.array(old);noise=np.array(noise);target=_spatial_render(z,h)
    regions={k:np.zeros((h,h),bool) for k in ('C','H','U_vis','changed_C')}
    regions['C'][:,:half]=1;regions['H'][:half,half:]=1;regions['U_vis'][half:,half:]=1;regions['changed_C'][half:,:half]=1
    visible=np.zeros((n,6,1,h,h),bool)
    visible[:,0,0]=regions['H']|regions['changed_C'];visible[:,2,0]=visible[:,0,0];visible[:,5,0]=regions['C']
    zo=z.copy();zo[:,1]=old[:,0];zo[:,2]=old[:,1];zh=z.copy();zh[:,2]=old[:,1]
    frames=np.repeat(target[:,None],6,axis=1);frames[:,0]=_spatial_render(zo,h);frames[:,2]=_spatial_render(zh,h)
    x=np.concatenate([np.where(visible,np.clip(frames+noise,0,1),0),visible],axis=2).astype('float32')
    y=(z[:,0,1]>=0).astype('int64') if task.objective=='classification' else target
    present=np.broadcast_to([True,True,True,False,True,True],(n,6)).copy()
    meta={'target_mode':'class_index' if task.objective=='classification' else 'continuous','classes':2,
          'output_shape':[2] if task.objective=='classification' else [4,h,h],
          'source_names':['C','H','changed_C','U_vis','old_H','old_changed_C'],'source_dims':[4]*6,
          'requested_sources':['C'] if task.objective=='classification' else ['C','H','changed_C','U_vis'],
          'independent_factors':True,'model_fields':['x'],'generator':'spatial_sources_counter_v2',
          'target_time':5,'timestamps_analysis_only':[0,1,3,6,10,15],
          'claim_limit':'synthetic four-band scenes; no physical satellite simulation'}
    meta['requested_coordinate_indices']={name:([1] if task.objective=='classification' and name=='C' else list(range(4)) if task.objective!='classification' and name in ('C','H','changed_C','U_vis') else []) for name in meta['source_names']}
    result={'x':x,'y':y,'loss_mask':np.ones_like(y,dtype='float32'),'targets':np.concatenate([z.reshape(n,-1),old.reshape(n,-1)],axis=1).astype('float32'),
            'z':z.astype('float32'),'stale':old.astype('float32'),'noise':noise.astype('float32'),'visible':visible,
            'source_times':np.broadcast_to([5,2,5,-1,0,0],(n,6)).copy(),'source_present':present,'meta':meta}
    result.update({'region_'+k:np.broadcast_to(v[None,None],(n,1,h,h)).copy() for k,v in regions.items()})
    return result


def legacy_batch(task,ids,split):
    from .legacy.tasks import TaskConfig,generate as original
    from .legacy.l1_contracts import task_contract,reconstruction_regions
    rows=[];contract=None
    for sid in ids:
        local_seed=int(seeded(task.data_seed,split,int(sid),'legacy').integers(0,2**31-1))
        cfg=TaskConfig(name=task.name,objective=task.objective,T=task.T,latent_dim=task.k,input_dim=task.sensory_dim,
                       sigma=task.sigma,new_steps=task.new_steps,delay=task.delay,interval=task.interval,
                       image_size=task.image_size,repeated_noise=task.repeated_noise,projection_seed=task.projection_seed,
                       data_seed=local_seed,shift=task.shift)
        a=original(cfg,1,'train');row={k:a[k] for k in ('x','y')};row['targets']=a['target'].reshape(1,-1).astype('float32')
        row['loss_mask']=np.ones_like(a['y'],dtype='float32')
        if task.name in IMAGE_TASKS:
            rr=reconstruction_regions(a,cfg)
            row.update({'region_C':rr['current_visible'],'region_H':rr['history_only_visible'],'region_U_vis':rr['never_visible_for_target']})
        rows.append(row);contract=task_contract(cfg);classes=a['classes']
    result={k:np.concatenate([row[k] for row in rows]) for k in rows[0]}
    result['source_times']=np.full((len(ids),1),task.T-1,dtype=int);result['source_present']=np.ones((len(ids),1),bool)
    result['meta']={'target_mode':'class_index' if task.objective=='classification' else 'continuous','classes':classes,
                    'output_shape':[classes] if task.objective=='classification' else list(result['y'].shape[1:]),
                    'source_names':['target'],'source_dims':[result['targets'].shape[1]],'requested_sources':['target'],
                    'independent_factors':False,'model_fields':['x'],'generator':'legacy_counter_scene_v2',
                    'legacy_generator_unchanged':True,'legacy_corpus_bitwise_reproduction':False,
                    'l1_contract':contract,'source_availability':'derived target; not a uniquely timed independent source'}
    return result


def symbolic_batch(task,ids,split):
    xs=[];ys=[];masks=[]
    for sid in ids:
        g=seeded(task.data_seed,split,int(sid),'symbolic')
        if task.family=='official_adding':
            # Conditional construction matches the balanced nine-marker task,
            # not the author's finite oversampling/with-replacement corpus.
            block,slot=divmod(int(sid),10);label=int(seeded(task.data_seed,split,block,'balance').permutation(10)[slot])
            x=np.zeros((task.T,2),'float32');x[:,0]=g.integers(0,2,task.T)
            positions=g.choice(task.T,9,replace=False);bits=np.r_[np.ones(label),np.zeros(9-label)];g.shuffle(bits)
            x[positions,0]=bits;x[positions,1]=1;y=np.array(label,dtype='int64');mask=np.array(1.,'float32')
        elif task.family=='copy_memory':
            count=task.k
            if task.T<2*count+1:raise ValueError('Copy: T>=2*k+1')
            symbols=g.integers(1,9,count);seq=np.zeros(task.T,int);seq[:count]=symbols;seq[-count-1]=9
            x=np.eye(10,dtype='float32')[seq];y=np.zeros(task.T,'int64');y[-count:]=symbols
            mask=np.zeros(task.T,'float32');mask[-count:]=1
        else:
            count=task.k;vocab=int(task.params.get('vocab_size',8));queries=int(task.params.get('queries',1))
            keys=g.choice(vocab,count,replace=False);values=g.integers(0,vocab,count);query=g.integers(count,size=queries)
            x=np.zeros((task.T,2*vocab+2),'float32')
            for j,(key,val) in enumerate(zip(keys,values)):
                x[2*j,key]=1;x[2*j,2*vocab]=1;x[2*j+1,vocab+val]=1
            for j,key in enumerate(keys[query]):x[-queries+j,key]=1;x[-queries+j,2*vocab+1]=1
            if task.objective=='sequence_classification':
                y=np.zeros(task.T,'int64');y[-queries:]=values[query];mask=np.zeros(task.T,'float32');mask[-queries:]=1
            else:y=np.array(values[query[0]],dtype='int64');mask=np.array(1.,'float32')
        xs.append(x);ys.append(y);masks.append(mask)
    x=np.stack(xs);y=np.stack(ys);classes=10 if task.family in ('official_adding','copy_memory') else int(task.params.get('vocab_size',8))
    if task.objective=='sequence_classification':
        positions=np.flatnonzero(np.stack(masks)[0]>0);targets=np.eye(classes,dtype='float32')[y[:,positions]].reshape(len(ids),-1)
        names=[f'answer_at_{int(p)}' for p in positions];dims=[classes]*len(positions)
        times=np.arange(len(positions)) if task.family=='copy_memory' else positions
    else:targets=np.eye(classes,dtype='float32')[y];names=['target'];dims=[classes];times=np.array([task.T-1])
    return {'x':x,'y':y,'loss_mask':np.stack(masks),'targets':targets,
            'source_times':np.broadcast_to(times,(len(ids),len(times))).copy(),'source_present':np.ones((len(ids),len(times)),bool),
            'meta':{'target_mode':'sequence_class_index' if task.objective=='sequence_classification' else 'class_index','classes':classes,
                    'output_shape':[classes],'source_names':names,'source_dims':dims,
                    'requested_sources':names,'independent_factors':False,'model_fields':['x'],'target_semantics':'categorical one-hot answers; no ordinal value-label geometry',
                    'generator':task.family+'_contract_reference_v1','author_corpus_reproduction':False}}


def generate(task,n,split='train',sample_ids=None):
    task.validate()
    if split not in SPLITS or n<1:raise ValueError('Known split and positive sample count')
    ids=np.arange(n,dtype='int64') if sample_ids is None else np.asarray(sample_ids,dtype='int64')
    if len(ids)!=n or (ids<0).any() or len(np.unique(ids))!=n:raise ValueError('Unique nonnegative scene IDs')
    if task.family=='archive':raise ValueError('Use DataStore for canonical external data')
    if task.family=='protocol':
        from .protocols import protocol_batch
        fn=protocol_batch
    else:fn={'legacy':legacy_batch,'tagged':tagged_batch,'spatial':spatial_batch}.get(task.family,symbolic_batch)
    result=fn(task,ids,split);result['scene_ids']=ids
    result['group_ids']=np.array([f'{task.data_seed}:{split}:{int(i)}' for i in ids],dtype='<U96')
    result['meta'].update(task_contract=task.identity(),task_fingerprint=digest(task.identity()),split=split)
    return result


def source_pair(task,batch,source,pair_seed=0):
    """IID resampling of one independent source, with target recomputation."""
    names=batch['meta']['source_names']
    if task.family!='tagged' or source not in names:raise ValueError('Source resampling currently has exact tagged-world semantics')
    j=names.index(source);out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in batch.items()}
    for i,sid in enumerate(batch['scene_ids']):out['z'][i,j]=seeded(task.data_seed,batch['meta']['split'],int(sid),'pair',source,pair_seed).normal(size=out['z'].shape[-1])
    out['targets']=out['z'].reshape(len(out['x']),-1);out['y'],out['loss_mask']=_tagged_targets(task,out['z'])
    out['x']=render_tagged(task,out['z'],out['noise'],out['source_times'],out['source_present'])
    out['meta']=dict(batch['meta'],changed_source=source,targets_recomputed=True)
    return out


class DataStore:
    """Atomic fixed-size NPY shards, shared across runs by task identity."""
    def __init__(self,task,root):
        self.task=task;code=Path(__file__).parent
        self.generator_hash=digest({name:file_hash(code/name) for name in ('data.py','protocols.py','utils.py','legacy/tasks.py','legacy/l1_contracts.py')})
        self.root=Path(root)/digest({'task':task.identity(),'generator_sha256':self.generator_hash})[:24]

    def ensure(self,split,n):
        if self.task.family=='archive':return self._external(split,n)
        dest=self.root/split/str(n)
        if not (dest/'complete.json').exists():
            dest.parent.mkdir(parents=True,exist_ok=True);lock=dest.with_name(dest.name+'.lock');acquired=False
            for attempt in range(1801):
                if (dest/'complete.json').exists():break
                try:
                    lock.mkdir();atomic_json(lock/'owner.json',{'host':socket.gethostname(),'pid':os.getpid()});acquired=True;break
                except FileExistsError:
                    ownerpath=lock/'owner.json'
                    if ownerpath.exists():
                        owner=json.loads(ownerpath.read_text())
                        if owner['host']==socket.gethostname():
                            try:os.kill(owner['pid'],0)
                            except ProcessLookupError:shutil.rmtree(lock,ignore_errors=True);continue
                    if attempt==1800:raise RuntimeError('Preparation lease did not complete; inspect '+str(lock))
                    time.sleep(1)
            if acquired:
                temp=dest.with_name(dest.name+f'.tmp-{os.getpid()}')
                try:
                    shutil.rmtree(temp,ignore_errors=True);temp.mkdir();arrays={};meta=None
                    for start in range(0,n,64):
                        b=generate(self.task,min(64,n-start),split,np.arange(start,min(start+64,n)))
                        meta=b.pop('meta')
                        for key,value in b.items():
                            if key not in arrays:arrays[key]=np.lib.format.open_memmap(temp/(key+'.npy'),mode='w+',dtype=value.dtype,shape=(n,*value.shape[1:]))
                            arrays[key][start:start+len(value)]=value
                    for a in arrays.values():a.flush()
                    arrays.clear();meta.update(samples=n,input_shape=list(b['x'].shape[2:]),steps=b['x'].shape[1],generator_implementation_sha256=self.generator_hash)
                    atomic_json(temp/'complete.json',meta);os.replace(temp,dest)
                finally:
                    shutil.rmtree(temp,ignore_errors=True);shutil.rmtree(lock)
        meta=json.loads((dest/'complete.json').read_text(encoding='utf-8'))
        if meta['task_fingerprint']!=digest(self.task.identity()) or meta['samples']!=n:raise ValueError('Prepared data contract mismatch')
        arrays={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in dest.glob('*.npy')};arrays['meta']=meta
        return arrays

    def _external(self,split,n):
        root=Path(self.task.archive_root);manifest=root/'dataset.json'
        if file_hash(manifest)!=self.task.archive_manifest_sha256:raise ValueError('External dataset manifest changed')
        receipt=root/'verification.json'
        if not receipt.exists() or json.loads(receipt.read_text())['manifest_sha256']!=self.task.archive_manifest_sha256:raise ValueError('Run verify-data on this canonical dataset before training')
        meta=json.loads(manifest.read_text(encoding='utf-8'))
        if split not in meta['splits']:raise ValueError('External split not prepared: '+split)
        path=root/meta['splits'][split]['directory'];rows=meta['splits'][split]['samples']
        if n>rows:raise ValueError(f'Requested {n}, available {rows} in {split}; never silently recycle data')
        for name,info in meta['splits'][split]['files'].items():
            if (path/name).stat().st_size!=info['bytes']:raise ValueError('External prepared file size changed: '+name)
        arrays={p.stem:np.load(p,mmap_mode='r',allow_pickle=False)[:n] for p in path.glob('*.npy')}
        arrays['meta']=dict(meta['contract'],samples=n,split=split,input_shape=list(arrays['x'].shape[2:]),steps=arrays['x'].shape[1],external_manifest_sha256=self.task.archive_manifest_sha256)
        return arrays


def subset(batch,indices):
    ids=np.asarray(indices);return {k:(np.asarray(v[ids]) if isinstance(v,np.ndarray) else v) for k,v in batch.items()}
