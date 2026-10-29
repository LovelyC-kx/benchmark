"""Lossless, complete observations and stream states on a declared scene bank.

Coordinate-limited probe caches are separate products. This archive saves every
leaf-module output, every temporal owner's observation dictionary, and every
future-influencing stream state after every input step. Inputs/targets/IDs are
included. Shards are batch-sized, atomic, checksummed and independently reusable.
"""
from __future__ import annotations
import hashlib
import json
import os
import struct
import time
from pathlib import Path
import zipfile
import numpy as np
import torch
from .data import DataStore,subset
from .engine import load_run,implementation_hash,numpy_losses,forecast_metrics,environment
from .runtime import tensor_input
from .utils import atomic_json,digest,file_hash,exclusive_file


class TensorArchive:
    """NPY inside ZIP; binary arrays bit-packed, original dtype restored exactly.

    Deduplication uses original bytes, dtype and shape, including signed zero.
    There is no float16 conversion, coordinate sampling, pickle or PCA.
    """
    def __init__(self,path):
        self.path=Path(path);self.zip=zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,allowZip64=True)
        self.arrays={};self.trees={};self.logical_bytes=0;self.prefixes={};self.active_tree=''
    def __enter__(self):return self
    def __exit__(self,kind,error,traceback):
        if error is not None:self.tree('incomplete_artifact',{'error':type(error).__name__+': '+str(error)})
        if self.zip.fp is not None:self.close()
    def tree(self,key,value):
        if key in self.trees:raise ValueError('Duplicate semantic observation: '+key)
        self.active_tree=key;self.trees[key]=self._encode(value,())
    def _prefix_key(self,location):
        if not location or not str(location[-1]).endswith(('keys','values')):return None
        if '/owner/' in self.active_tree:
            site=self.active_tree.split('/owner/',1)[1].rsplit('/last',1)[0];return site+'/'+('/'.join(location))
        if self.active_tree.endswith('/snapshot') and len(location)>=3 and location[0]=='owners':return '/'.join(location[1:])
        return None
    def _encode(self,value,location):
        tensor=isinstance(value,torch.Tensor);torch_dtype=None
        if tensor:
            torch_dtype=str(value.dtype);value=value.detach().resolve_conj().resolve_neg().cpu().contiguous()
            value=value.view(torch.uint16).numpy() if value.dtype==torch.bfloat16 else value.numpy()
        if isinstance(value,np.ndarray):
            a=np.asarray(value)
            if a.dtype.hasobject:raise ValueError('Object arrays cannot enter a scientific archive')
            shape=list(a.shape);a=np.ascontiguousarray(a);h=hashlib.sha256()
            h.update(a.dtype.str.encode());h.update(json.dumps(shape).encode())
            if a.nbytes:h.update(memoryview(a).cast('B'))
            name=h.hexdigest()
            self.logical_bytes+=a.nbytes
            prefix_key=self._prefix_key(location);previous=self.prefixes.get(prefix_key) if prefix_key else None;delta=None;axis=None
            if previous and a.ndim in (3,4):
                old_name,old=previous;different=[i for i,(x,y) in enumerate(zip(a.shape,old.shape)) if x!=y]
                if a.ndim==old.ndim and a.dtype==old.dtype and len(different)==1:
                    axis=different[0]
                    if a.shape[axis]>old.shape[axis]:
                        indices=[slice(None)]*a.ndim;indices[axis]=slice(0,old.shape[axis])
                        if a[tuple(indices)].tobytes()==old.tobytes():
                            indices[axis]=slice(old.shape[axis],None);delta=np.ascontiguousarray(a[tuple(indices)])
            if name not in self.arrays:
                payload=a if delta is None else delta
                binary=payload.dtype.kind=='b' or (payload.dtype.kind in 'fiu' and payload.size>=8 and np.all((payload==0)|(payload==1)) and not (payload.dtype.kind=='f' and np.any(np.signbit(payload)&(payload==0))))
                packed=np.packbits(payload.reshape(-1).astype(bool),bitorder='little') if binary else payload
                entry='arrays/'+name+'.npy'
                with self.zip.open(entry,'w',force_zip64=True) as stream:np.lib.format.write_array(stream,packed,allow_pickle=False)
                self.arrays[name]={'entry':entry,'shape':shape,'dtype':a.dtype.str,'encoding':'packbits-little' if binary else 'npy','original_bytes':a.nbytes,
                                   'payload_shape':list(payload.shape),'payload_bytes':payload.nbytes}
                if delta is not None:self.arrays[name].update(prefix_base=previous[0],prefix_axis=axis)
            if prefix_key:self.prefixes[prefix_key]=(name,a)
            return {'type':'tensor' if tensor else 'array','array':name,'torch_dtype':torch_dtype}
        if isinstance(value,dict):return {'type':'dict','items':[[str(k),self._encode(v,(*location,str(k)))] for k,v in value.items()]}
        if isinstance(value,(list,tuple)):return {'type':'tuple' if isinstance(value,tuple) else 'list','items':[self._encode(v,(*location,str(i))) for i,v in enumerate(value)]}
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,float) and not np.isfinite(value):return {'type':'float_bits','bits':struct.pack('!d',value).hex()}
        if value is None or isinstance(value,(str,int,float,bool)):return {'type':'scalar','value':value}
        raise TypeError('Unsupported archive value: '+str(type(value)))
    def close(self):
        index={'schema':'complete_tensor_archive_v1','arrays':self.arrays,'trees':self.trees,'logical_bytes_before_dedup':self.logical_bytes,
               'unique_array_bytes_before_zip':sum(v['original_bytes'] for v in self.arrays.values())}
        index['payload_bytes_after_prefix_reuse']=sum(v['payload_bytes'] for v in self.arrays.values())
        self.zip.writestr('index.json',json.dumps(index,ensure_ascii=False,allow_nan=False));self.zip.close();return index


class ArchiveReader:
    def __init__(self,path):
        self.zip=zipfile.ZipFile(path);self.index=json.loads(self.zip.read('index.json'))
        if self.index['schema']!='complete_tensor_archive_v1':raise ValueError('Archive schema')
    def tree(self,key,tensors=False):return self._decode(self.index['trees'][key],tensors)
    def _array(self,name):
        pieces=[];axis=None;original_shape=self.index['arrays'][name]['shape']
        while True:
            info=self.index['arrays'][name]
            with self.zip.open(info['entry']) as f:a=np.lib.format.read_array(f,allow_pickle=False)
            shape=info.get('payload_shape',info['shape'])
            if info['encoding']=='packbits-little':a=np.unpackbits(a,bitorder='little',count=int(np.prod(shape))).astype(np.dtype(info['dtype']))
            a=a.reshape(shape);pieces.append(a)
            if 'prefix_base' not in info:break
            if axis is not None and axis!=info['prefix_axis']:raise ValueError('Prefix axis changed inside an archive chain')
            axis=info['prefix_axis'];name=info['prefix_base']
        result=np.concatenate(list(reversed(pieces)),axis=axis) if len(pieces)>1 else pieces[0]
        return result.reshape(original_shape)
    def _decode(self,node,tensors):
        kind=node['type']
        if kind in ('array','tensor'):
            a=self._array(node['array'])
            if kind=='tensor' and tensors:
                a=torch.from_numpy(a.copy())
                if node['torch_dtype']=='torch.bfloat16':a=a.view(torch.bfloat16)
            return a
        if kind=='scalar':return node['value']
        if kind=='float_bits':return struct.unpack('!d',bytes.fromhex(node['bits']))[0]
        if kind=='dict':return {k:self._decode(v,tensors) for k,v in node['items']}
        values=[self._decode(v,tensors) for v in node['items']]
        return tuple(values) if kind=='tuple' else values
    def close(self):self.zip.close()


def _snapshot(model):
    # Values are synchronously copied to CPU by the writer. Avoid a full extra
    # GPU clone, which would otherwise double large KV/spatial state peaks.
    owners=model.temporal_owners()
    return {'schema':'complete_stream_v1','time':model.time,'owners':{k:m.state for k,m in owners.items()},
            'event_controls':{k:getattr(m,'event_controls',{}) for k,m in owners.items()},
            'cache':model.backbone.cache,'readout':model.readout_state}


@torch.no_grad()
def capture_batch(model,batch,path,device='cpu',leaf_outputs=True):
    started=time.monotonic()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial-'+str(os.getpid()))
    writer=TensorArchive(temp);handles=[];calls={}
    def owner_hook(site):
        def save(module,args,output):writer.tree(f'step/{model.time:08d}/owner/{site}/last',module.last)
        return save
    def leaf_hook(site):
        def save(module,args,output):
            key=(model.time,site);count=calls.get(key,0);calls[key]=count+1
            writer.tree(f'step/{model.time:08d}/leaf/{site}/call/{count:04d}',{'input':args,'output':output})
        return save
    try:
        writer.tree('batch',{k:v for k,v in batch.items() if k!='meta'});writer.tree('metadata',batch['meta'])
        model.eval();model.reset();writer.tree('initial/snapshot',_snapshot(model))
        for site,module in model.temporal_owners().items():handles.append(module.register_forward_hook(owner_hook(site)))
        if leaf_outputs:
            for site,module in model.named_modules():
                if site and not any(module.children()):handles.append(module.register_forward_hook(leaf_hook(site)))
        pred=model(tensor_input(batch,device),after_step=lambda m,t:writer.tree(f'step/{t:08d}/snapshot',_snapshot(m)))
        loss,correct=numpy_losses(tuple(p.cpu().numpy() for p in pred) if isinstance(pred,tuple) else pred.cpu().numpy(),batch)
        writer.tree('prediction',pred);writer.tree('native_loss',loss);writer.tree('correct',correct);writer.tree('forecast_metrics',forecast_metrics(pred,batch))
        index=writer.close();os.replace(temp,path)
        return {'file':path.name,'sha256':file_hash(path),'bytes':path.stat().st_size,'samples':len(batch['x']),
                'steps':batch['x'].shape[1],'semantic_records':len(index['trees']),'logical_bytes_before_dedup':index['logical_bytes_before_dedup'],
                'unique_array_bytes_before_zip':index['unique_array_bytes_before_zip'],'payload_bytes_after_prefix_reuse':index['payload_bytes_after_prefix_reuse'],
                'archive_elapsed_seconds':time.monotonic()-started,'complete_coordinates':True}
    except BaseException as exc:
        failed=path.with_name(path.name+'.failed-'+str(time.time_ns())+'.zip')
        try:
            writer.tree('capture_exception',{'type':type(exc).__name__,'message':str(exc),'scope':'partial evidence, not a complete archive'})
            writer.close();os.replace(temp,failed)
        except Exception:
            writer.zip.close()
            if temp.exists():os.replace(temp,failed)
        raise
    finally:
        for handle in handles:handle.remove()


def archive_run(run,data_root='prepared',checkpoint='best.pt',device='cpu',roles=None,scenes=None,batch_size=8,leaf_outputs=True):
    if batch_size<1 or (scenes is not None and scenes<1):raise ValueError('Positive capture sizes')
    roles=tuple(roles or ('calibration','probe_train','probe_val','verification_train','verification_val','development'))
    allowed={'train','val','calibration','probe_train','probe_val','verification_train','verification_val','development','confirmation','audit'}
    if not roles or set(roles)-allowed or len(set(roles))!=len(roles):raise ValueError('Distinct explicit data roles')
    with exclusive_file(Path(run)/'archive.lock'):
        spec,model,ck,path=load_run(run,checkpoint,device);a=spec.analysis
        sizes={role:int(scenes if scenes is not None else a.fit_scenes if role in ('probe_train','verification_train') else a.select_scenes if role in ('calibration','probe_val','verification_val') else a.eval_scenes) for role in roles}
        contract={'schema':'snnbench_complete_archive_v1','run_id':spec.run_id,'checkpoint_sha256':file_hash(path),'parameter_sha256':ck['parameter_sha256'],
                  'implementation_sha256':implementation_hash('analysis'),'training_implementation_sha256':ck['implementation_sha256'],'sizes':sizes,'batch_size':batch_size,'leaf_outputs':leaf_outputs,
                  'environment':environment(device),
                  'precision':'original tensor bits, lossless ZIP and exact binary packing','coordinate_policy':'all coordinates; no feature/target budget',
                  'scope':'declared scene banks, every input step; not every training batch activation','access':'model receives x only; saved targets/latents are analysis metadata'}
        out=Path(run)/'archive'/digest(contract)[:24];out.mkdir(parents=True,exist_ok=True);receipt=out/'receipt.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            for shard in saved['shards']:
                if file_hash(out/shard['path'])!=shard['sha256']:raise ValueError('Sealed archive checksum mismatch')
            return out
        atomic_json(out/'contract.json',contract);store=DataStore(spec.task,data_root);shards=[]
        for role,n in sizes.items():
            data=store.ensure(role,n)
            for start in range(0,n,batch_size):
                name=f'{role}/batch_{start:010d}.zip';target=out/name;metadata=target.with_suffix('.receipt.json')
                if metadata.exists() and target.exists():
                    info=json.loads(metadata.read_text())
                    if info['sha256']!=file_hash(target):raise ValueError('Partial archive checksum mismatch')
                else:
                    info=capture_batch(model,subset(data,np.arange(start,min(start+batch_size,n))),target,device,leaf_outputs);atomic_json(metadata,info)
                shards.append(dict(info,path=name,role=role,start=start))
        atomic_json(receipt,{'status':'archived','contract':contract,'shards':shards,'stored_bytes':sum(s['bytes'] for s in shards),
                            'logical_bytes_before_dedup':sum(s['logical_bytes_before_dedup'] for s in shards),'complete_coordinates':True})
        return out
