"""Native-task training, atomic checkpoints, and exact same-environment resume."""
from __future__ import annotations
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import DataStore,subset
from .models import create_model
from .runtime import tensor_input
from .specs import RunSpec
from .utils import atomic_json,atomic_torch,digest,file_hash,parameter_hash,seeded


def implementation_hash(scope='training'):
    root=Path(__file__).parent
    training=('engine.py','telemetry.py','utils.py','specs.py','models.py','extended_models.py','data.py','protocols.py','reference/__init__.py','reference/models.py','reference/neurons.py','legacy/tasks.py','legacy/l1_contracts.py')
    paths=[root/p for p in training] if scope=='training' else sorted(root.rglob('*.py'))
    return digest({'scope':scope,'files':{str(p.relative_to(root)):file_hash(p) for p in paths if '__pycache__' not in str(p)}})


def configure(seed,threads=2,deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark=False


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s):
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype='uint32'),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'])
    if s['cuda']:
        if len(s['cuda'])!=torch.cuda.device_count():raise ValueError('Resume CUDA topology differs')
        torch.cuda.set_rng_state_all(s['cuda'])


def native_losses(pred,batch,device='cpu'):
    """Per-scene native loss; auxiliary latent probes never replace task labels."""
    meta=batch['meta'];mode=meta['target_mode'];seq=None
    if isinstance(pred,tuple):pred,seq=pred
    y=torch.as_tensor(np.array(batch['y'],copy=True),device=device)
    mask=torch.as_tensor(np.array(batch['loss_mask'],copy=True),device=device,dtype=torch.float32)
    correct=None
    if mode=='class_index':
        y=y.long().reshape(-1);loss=F.cross_entropy(pred,y,reduction='none');correct=(pred.argmax(-1)==y).float()
    elif mode=='sequence_class_index':
        if seq is None:raise ValueError('Sequence task needs a per-step prediction')
        y=y.long();raw=F.cross_entropy(seq.transpose(1,2),y,reduction='none')
        den=mask.flatten(1).sum(1)
        if (den<=0).any():raise ValueError('Every scene needs a supervised position')
        loss=(raw*mask).flatten(1).sum(1)/den;correct=((seq.argmax(-1)==y)*mask).flatten(1).sum(1)/den
    else:
        y=y.float()
        if pred.shape!=y.shape:raise ValueError(f'Native target shape {tuple(y.shape)} != prediction {tuple(pred.shape)}')
        raw=F.binary_cross_entropy_with_logits(pred,y,reduction='none') if mode=='multilabel' else (pred-y).square()
        mask=torch.broadcast_to(mask,raw.shape);den=mask.flatten(1).sum(1)
        if (den<=0).any():raise ValueError('Every scene needs a supervised coordinate')
        loss=(raw*mask).flatten(1).sum(1)/den
        if mode=='multilabel':correct=(((pred>=0)==(y>=.5))*mask).flatten(1).sum(1)/den
    if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite native loss')
    return loss,correct


def numpy_losses(pred,batch):
    convert=lambda x:torch.as_tensor(x,dtype=torch.float32)
    p=tuple(convert(x) for x in pred) if isinstance(pred,tuple) else convert(pred)
    with torch.no_grad():loss,correct=native_losses(p,batch)
    return loss.numpy(),None if correct is None else correct.numpy()


def forecast_metrics(pred,batch):
    """Descriptive metrics, including original units and the last-value control.

    Training/selection still use the native standardized MSE. Raw-unit MSE pooled
    across heterogeneous channels is descriptive, not a scale-free ranking.
    """
    c=batch['meta']
    if c.get('protocol')!='chronological_ten_role_no_crossing_v1':return {}
    p=pred[0] if isinstance(pred,tuple) else pred
    if isinstance(p,torch.Tensor):p=p.detach().cpu().numpy()
    y=np.asarray(batch['y']);error=np.asarray(p,dtype='float64')-y;scale=np.asarray(c['normalization']['std'])
    persistence=np.asarray(batch['x'])[:,-1,None,:]-y
    return {name:value.reshape(len(y),-1).mean(1) for name,value in {
        'MSE_standardized':error**2,'MAE_standardized':np.abs(error),'MSE_original_units':(error*scale)**2,'MAE_original_units':np.abs(error*scale),
        'persistence_MSE_standardized':persistence**2,'persistence_MAE_standardized':np.abs(persistence)}.items()}


@torch.no_grad()
def evaluate(model,data,batch_size,device):
    model.eval();losses=[];corrects=[];forecast={}
    for start in range(0,len(data['x']),batch_size):
        batch=subset(data,np.arange(start,min(start+batch_size,len(data['x']))));p=model(tensor_input(batch,device))
        loss,c=native_losses(p,batch,device);losses.append(loss.cpu().numpy())
        for name,values in forecast_metrics(p,batch).items():forecast.setdefault(name,[]).append(values)
        if c is not None:corrects.append(c.cpu().numpy())
    loss=np.concatenate(losses)
    _,groups=np.unique(data['group_ids'],return_inverse=True);g=int(groups.max())+1
    sums=np.bincount(groups,weights=loss.astype(float)-float(loss.mean()))
    se=float(np.sqrt(g/(g-1)*np.sum(sums*sums))/len(loss)) if g>1 else None
    result={'native_loss':float(loss.mean()),'native_loss_se':se,'native_loss_se_unit':'source group; cluster robust descriptive SE','independent_groups':g,
            'accuracy':float(np.concatenate(corrects).mean()) if corrects else None,'scenes':len(loss)}
    if forecast:result['forecasting']={name:float(np.concatenate(values).mean()) for name,values in forecast.items()}
    return result


def environment(device):
    return {'torch':str(torch.__version__),'numpy':np.__version__,'python':__import__('platform').python_version(),
            'device':str(device),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            'cuda_count':torch.cuda.device_count() if str(device).startswith('cuda') else 0,
            'total_gpu_memory_bytes':torch.cuda.get_device_properties(device).total_memory if str(device).startswith('cuda') else None,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}


def train(spec,out_root='runs',data_root='prepared',device='cpu',resume=True,max_updates=None):
    spec.validate();cfg=spec.train;configure(cfg.seed,cfg.threads,cfg.deterministic)
    if max_updates is not None and max_updates<1:raise ValueError('max_updates must be a positive invocation limit')
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    run=Path(out_root)/spec.run_id;run.mkdir(parents=True,exist_ok=True)
    lock=run/'training.lock'
    if lock.exists() and resume:
        owner=json.loads(lock.read_text());host=__import__('socket').gethostname()
        if owner.get('host')==host:
            try:os.kill(int(owner['pid']),0)
            except ProcessLookupError:lock.unlink()
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Training already leased. Inspect owner before removing '+str(lock))
    os.write(fd,json.dumps({'pid':os.getpid(),'host':__import__('socket').gethostname()}).encode());os.close(fd)
    try:return _train_locked(spec,run,data_root,device,resume,max_updates)
    finally:lock.unlink(missing_ok=True)


def _train_locked(spec,run,data_root,device,resume,max_updates):
    cfg=spec.train;store=DataStore(spec.task,data_root);train_data=store.ensure('train',cfg.train_scenes);val=store.ensure('val',cfg.val_scenes)
    model=create_model(spec,train_data['meta'],device)
    expected_mode={'classification':('class_index','multilabel'),'sequence_classification':('sequence_class_index',),'regression':('continuous',),'reconstruction':('continuous',)}
    if train_data['meta']['target_mode'] not in expected_mode[spec.task.objective]:raise ValueError('Task objective and canonical native labels disagree')
    if cfg.encoder_checkpoint:
        if file_hash(cfg.encoder_checkpoint)!=cfg.encoder_checkpoint_sha256:raise ValueError('Frozen encoder file hash differs')
        enc=torch.load(cfg.encoder_checkpoint,map_location=device,weights_only=True);model.encoder.load_state_dict(enc['encoder'],strict=True)
        for p in model.encoder.parameters():p.requires_grad=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr,weight_decay=cfg.weight_decay)
    impl=implementation_hash();env=environment(device);contract=train_data['meta'];initial=parameter_hash(model)
    progress={'epoch':0,'cursor':0,'updates':0,'epoch_loss_sum':0.,'epoch_scenes':0,'best_val':None,'history':[]}
    last=run/'last.pt';total=cfg.epochs*math.ceil(cfg.train_scenes/cfg.batch_size)
    milestones={int(math.ceil(float(f)*total)) for f in cfg.checkpoint_fractions}
    journal=None
    def save(path):
        if journal is not None:journal.sync()
        atomic_torch(path,{'schema':'snnbench_checkpoint_v2','run_id':spec.run_id,'spec':spec.to_dict(),'contract':contract,
            'implementation_sha256':impl,'environment':env,'model':model.state_dict(),'optimizer':opt.state_dict(),
            'rng':rng_state(),'progress':progress,'parameter_sha256':parameter_hash(model),'initial_parameter_sha256':initial})
    if last.exists():
        if not resume:raise FileExistsError('Run already exists; choose a new scientific specification or resume')
        ck=torch.load(last,map_location=device,weights_only=True)
        if ck['run_id']!=spec.run_id or ck['implementation_sha256']!=impl:raise ValueError('Strict resume requires identical specification and implementation')
        if ck['environment']!=env:raise ValueError('Strict exact resume requires the same runtime and device topology')
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng']);progress=ck['progress'];initial=ck['initial_parameter_sha256']
        if parameter_hash(model)!=ck['parameter_sha256']:raise ValueError('Checkpoint parameter hash mismatch')
    else:
        atomic_json(run/'spec.json',spec.to_dict());save(run/'step_00000000.pt');save(last)
    from .telemetry import UpdateJournal,gradient_statistics,torch_safe
    journal=UpdateJournal(run,progress['updates'],spec.run_id)
    invocation_updates=0;started=time.monotonic()
    while progress['epoch']<cfg.epochs:
        epoch=progress['epoch'];order=seeded(cfg.order_seed,'epoch_order',epoch).permutation(cfg.train_scenes)
        while progress['cursor']<len(order):
            start=progress['cursor'];ids=order[start:start+cfg.batch_size];batch=subset(train_data,ids)
            before=time.monotonic();model.train();opt.zero_grad(set_to_none=True);stage='forward';before_rng=rng_state()
            try:
                prediction=model(tensor_input(batch,device));stage='native_loss';loss,correct=native_losses(prediction,batch,device)
                stage='backward';loss.mean().backward();stage='gradient_clip';norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_norm,error_if_nonfinite=True)
                gradient=gradient_statistics(model);stage='optimizer';opt.step()
            except Exception as exc:
                failure=run/f'failure_update_{progress["updates"]+1:08d}.pt'
                atomic_torch(failure,{'schema':'failed_update_evidence_v1','stage':stage,'run_id':spec.run_id,'progress':progress,
                    'batch':torch_safe(batch),'pre_forward_rng':before_rng,'post_failure_rng':rng_state(),'model':model.state_dict(),'optimizer':opt.state_dict(),
                    'implementation_sha256':impl,'error':type(exc).__name__+': '+str(exc)})
                atomic_json(run/'failure.json',{'stage':stage,'update':progress['updates']+1,'evidence':failure.name,'error':type(exc).__name__+': '+str(exc)})
                atomic_json(run/'status.json',{'status':'failed','run_id':spec.run_id,'progress':progress,'failure_stage':stage,
                    'failure_evidence':failure.name,'error':type(exc).__name__+': '+str(exc),'environment':env})
                raise
            progress['cursor']+=len(ids);progress['updates']+=1;invocation_updates+=1
            progress['last_gradient_norm']=float(norm.cpu())
            progress['epoch_loss_sum']+=float(loss.detach().sum().cpu());progress['epoch_scenes']+=len(ids)
            u=progress['updates']
            journal.append({'update':u,'epoch':epoch,'cursor_after_update':progress['cursor'],'train_indices':ids.tolist(),
                'scene_ids':batch['scene_ids'],'source_ids':batch.get('source_ids'),'group_ids':batch['group_ids'],
                'native_losses':loss.detach().cpu().numpy(),'correct':None if correct is None else correct.detach().cpu().numpy(),
                'gradient_norm_before_clip':float(norm.cpu()),'gradient_statistics_after_clip':gradient,
                'learning_rates':[g['lr'] for g in opt.param_groups],'elapsed_seconds':time.monotonic()-before,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
                'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if str(device).startswith('cuda') else None})
            if u in milestones:save(run/f'step_{u:08d}.pt')
            if u%cfg.save_every_updates==0:save(last)
            if max_updates is not None and invocation_updates>=max_updates:
                save(last);atomic_json(run/'status.json',{'status':'paused','progress':progress,'run_id':spec.run_id,'parameter_sha256':parameter_hash(model)})
                return run
        metrics=evaluate(model,val,cfg.batch_size,device)
        row={'epoch':epoch+1,'updates':progress['updates'],'train_native_loss':progress['epoch_loss_sum']/progress['epoch_scenes'],
             'val':metrics,'last_gradient_norm':progress.get('last_gradient_norm')}
        progress['history'].append(row);progress.update(epoch=epoch+1,cursor=0,epoch_loss_sum=0.,epoch_scenes=0)
        if progress['best_val'] is None or metrics['native_loss']<progress['best_val']:
            progress['best_val']=metrics['native_loss'];save(run/'best.pt')
        save(last);atomic_json(run/'history.json',progress['history'])
        print(json.dumps({'run_id':spec.run_id,**row}),flush=True)
    atomic_json(run/'status.json',{'status':'trained','run_id':spec.run_id,'updates':progress['updates'],'epochs':progress['epoch'],
        'parameter_sha256':parameter_hash(model),'initial_parameter_sha256':initial,'parameters_changed':initial!=parameter_hash(model),
        'elapsed_seconds_this_invocation':time.monotonic()-started,'environment':env,'analysis_completed':False})
    return run


def load_run(run,checkpoint='best.pt',device='cpu'):
    run=Path(run);spec=RunSpec.load(run/'spec.json');path=run/checkpoint
    ck=torch.load(path,map_location=device,weights_only=True)
    if ck['schema']!='snnbench_checkpoint_v2' or ck['run_id']!=spec.run_id:raise ValueError('Checkpoint identity')
    if ck['implementation_sha256']!=implementation_hash():raise ValueError('Model/training implementation changed; analyze with the pinned source revision')
    configure(spec.train.seed,spec.train.threads,spec.train.deterministic);model=create_model(spec,ck['contract'],device)
    model.load_state_dict(ck['model'],strict=True)
    if spec.train.encoder_checkpoint:
        for p in model.encoder.parameters():p.requires_grad=False
    if parameter_hash(model)!=ck['parameter_sha256']:raise ValueError('Checkpoint parameter hash mismatch')
    return spec,model,ck,path
