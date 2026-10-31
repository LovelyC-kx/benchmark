from pathlib import Path
import json,hashlib,platform,time,random,os,tempfile
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .tasks import TaskConfig,generate
from .models import ModelConfig,BenchmarkModel
from .data import load_splits

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)

def clean_json(v):
    if isinstance(v,dict):return {k:clean_json(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean_json(x) for x in v]
    if isinstance(v,np.ndarray):return clean_json(v.tolist())
    if isinstance(v,np.generic):return clean_json(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def write_json(path,v):
    """Replace complete records atomically, including incrementally updated summaries."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(clean_json(v),f,indent=2,ensure_ascii=False)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def source_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def tensor_data(a,device):
    x=torch.as_tensor(a['x'],dtype=torch.float32,device=device)
    y=torch.as_tensor(a['y'],device=device)
    return x,y.long() if y.ndim==1 and not y.is_floating_point() else y.float()

def loss_fn(p,y,objective):return nn.functional.cross_entropy(p,y) if objective=='classification' else nn.functional.mse_loss(p,y)

def metrics(pred,y,objective,data_range=1.):
    from sklearn.metrics import r2_score,balanced_accuracy_score
    if objective=='classification':
        p=pred.softmax(-1);guess=p.argmax(-1);k=min(5,p.shape[-1])
        return {'accuracy':float((guess==y).float().mean()),
                'top5':float((p.topk(k,-1).indices==y[:,None]).any(-1).float().mean()),
                'balanced_accuracy':float(balanced_accuracy_score(y.cpu(),guess.cpu())),
                'cross_entropy':float(nn.functional.cross_entropy(pred,y))}
    mse=float((pred-y).square().mean());a=y.cpu().numpy().reshape(len(y),-1);b=pred.cpu().numpy().reshape(len(y),-1)
    active=np.var(a,0)>1e-12
    out={'mse':mse,'r2_mean':float(r2_score(a[:,active],b[:,active])) if active.any() else None}
    if objective=='reconstruction' and y.ndim==4:
        from scipy.ndimage import uniform_filter
        per=(pred-y).square().flatten(1).mean(1).cpu().numpy()
        if data_range<=0:raise ValueError('data_range must be positive')
        out['psnr_db_mean']=float(np.mean(10*np.log10(data_range**2/np.maximum(per,1e-12))))
        out['psnr_data_range']=data_range;out['ssim_definition']='uniform 7x7 per-channel; declared data range; NOT official Y-channel paper metric'
        yp=pred.cpu().numpy();yt=y.cpu().numpy();ux=uniform_filter(yp,size=(1,1,7,7));uy=uniform_filter(yt,size=(1,1,7,7))
        vx=uniform_filter(yp*yp,size=(1,1,7,7))-ux*ux;vy=uniform_filter(yt*yt,size=(1,1,7,7))-uy*uy
        vxy=uniform_filter(yp*yt,size=(1,1,7,7))-ux*uy
        c1=(.01*data_range)**2;c2=(.03*data_range)**2
        s=((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
        out['ssim_mean']=float(s.mean());out['predictions_clipped_for_metrics']=False
    return out

@torch.no_grad()
def evaluate(model,a,objective,device,batch=64):
    model.eval();pred=[];ys=[]
    for i in range(0,len(a['x']),batch):
        x,y=tensor_data({'x':a['x'][i:i+batch],'y':a['y'][i:i+batch]},device)
        pred.append(model(x).cpu());ys.append(y.cpu())
    p=torch.cat(pred);y=torch.cat(ys)
    return metrics(p,y,objective,getattr(model,'metric_data_range',1.)),p

def train_run(config,out):
    if config.get('lr_grid'):
        return hyperparameter_search(config,out)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    seed=config.get('seed',0);seed_all(seed)
    torch.set_num_threads(config.get('threads',2))
    device=torch.device(config.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    tc=TaskConfig(**config.get('task',{}));mc=ModelConfig(**config.get('model',{}))
    objective=tc.objective
    if config.get('data_root'):datasets=load_splits(config['data_root'],objective)
    else:
        sizes=config.get('sizes',{'train':512,'val':128,'test':512})
        datasets={s:generate(tc,sizes[s],s) for s in ('train','val','test')}
    train,val,test=(datasets[s] for s in ('train','val','test'))
    output_shape=((int(np.max(train['y']))+1,) if objective=='classification' else train['y'].shape[1:])
    if objective=='classification' and set(np.unique(test['y']))-set(np.unique(train['y'])):raise ValueError('test class missing from training')
    model=BenchmarkModel(train['x'].shape[2:],output_shape,train['x'].shape[1],mc,objective).to(device)
    model.metric_data_range=float(config.get('data_range',1.))
    if config.get('freeze_backbone',False):
        for p in model.backbone.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.get('lr',.001),weight_decay=config.get('weight_decay',.0001))
    batch=config.get('batch_size',64);epochs=config.get('epochs',12);patience=config.get('patience',10)
    loss_mode=config.get('loss_mode','aggregate')
    if loss_mode not in ('aggregate','per_step'):raise ValueError('loss_mode')
    logs=[];best=float('inf');best_epoch=-1;stale=0;start=time.perf_counter()
    generator=torch.Generator().manual_seed(seed+1223)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(train['x']),generator=generator).numpy();total=0
        for j in range(0,len(order),batch):
            ids=order[j:j+batch];x,y=tensor_data({'x':train['x'][ids],'y':train['y'][ids]},device)
            optimizer.zero_grad(set_to_none=True)
            if loss_mode=='per_step':
                pred,seq=model(x,return_sequence=True)
                loss=sum(loss_fn(seq[:,t],y,objective) for t in range(seq.shape[1]))/seq.shape[1]
            else:pred=model(x);loss=loss_fn(pred,y,objective)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config.get('clip_grad',1.))
            optimizer.step();total+=float(loss.detach())*len(ids)
        vm,_=evaluate(model,val,objective,device,batch)
        vloss=vm['cross_entropy'] if objective=='classification' else vm['mse']
        logs.append({'epoch':epoch,'train_loss':total/len(train['x']),'val_loss':vloss})
        if vloss<best:
            best=vloss;best_epoch=epoch;stale=0
            torch.save({'state_dict':model.state_dict(),'model':asdict(mc),'task':asdict(tc),
                        'input_shape':train['x'].shape[2:],'output_shape':output_shape,
                        'T':train['x'].shape[1],'objective':objective,'data_range':model.metric_data_range,
                        'data_root':config.get('data_root'),'dataset_metadata':train.get('meta',{}),
                        'source_hash':source_hash()},out/'best.pt')
        else:stale+=1
        if stale>=patience:break
    ckpt=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ckpt['state_dict'],strict=True)
    defer_test=config.get('defer_test',False)
    tm,p=({},None) if defer_test else evaluate(model,test,objective,device,batch)
    resources=model.resources()
    result={'config':config,'seed':seed,'best_epoch_validation_only':best_epoch,
            'epochs_completed':len(logs),'updates_completed':len(logs)*int(np.ceil(len(train['x'])/batch)),
            'test':tm,'validation_best_loss':best,'test_deferred':defer_test,
            'train_log':logs,'wall_seconds':time.perf_counter()-start,'resources':resources,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':str(device)},
            'source_hash':source_hash(),'stage':config.get('stage','controlled_feasibility'),
            'formal_benchmark_claim':False,'test_never_used_for_selection':True,'dataset_metadata':train.get('meta',{})}
    if not defer_test and objective=='reconstruction' and test['y'].ndim==4 and 'mask' in test:
        sq=(p.numpy()-test['y'])**2;mask=test['mask'][:,-1]
        for key,m in [('visible',mask),('currently_missing',1-mask)]:
            result['test'][key+'_mse']=float((sq*m).sum()/max(1,m.sum()))
    if device.type=='cuda':result['resources']['peak_cuda_memory_bytes']=torch.cuda.max_memory_allocated(device)
    write_json(out/'result.json',result)
    if p is not None:np.savez_compressed(out/'predictions.npz',prediction=p.numpy(),target=test['y'])
    # Held-out conditions are evaluated AFTER validation selection; no per-condition refit.
    if not defer_test and not config.get('data_root') and config.get('heldout'):
        held=[]
        for change in config['heldout']:
            cfg=TaskConfig(**{**asdict(tc),**change});a=generate(cfg,config.get('heldout_size',256),'heldout')
            if mc.readout=='trajectory' and cfg.T!=tc.T:
                held.append({'condition':change,'status':'incompatible_fixed_T_head'});continue
            met,_=evaluate(model,a,objective,device,batch);held.append({'condition':change,'test':met,'refitted':False})
        write_json(out/'heldout.json',held)
    return result

def hyperparameter_search(config,out):
    """All candidates select/checkpoint on val only; test evaluated once after selection."""
    import copy,shutil
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,lr in enumerate(config['lr_grid']):
        c=copy.deepcopy(config);c.pop('lr_grid');c['lr']=lr;c['defer_test']=True;c['heldout']=[]
        r=train_run(c,out/'candidates'/f'lr_{i}')
        rows.append({'lr':lr,'validation_loss':r['validation_best_loss'],'result':r,'index':i})
    best=min(rows,key=lambda r:r['validation_loss']);idx=best['index']
    shutil.copy2(out/'candidates'/f'lr_{idx}'/'best.pt',out/'best.pt')
    device=config.get('device','cpu');model,ck=load_checkpoint(out/'best.pt',device)
    tc=TaskConfig(**config['task'])
    test=(load_splits(config['data_root'],tc.objective)['test'] if config.get('data_root')
          else generate(tc,config.get('sizes',{}).get('test',20000),'test'))
    met,p=evaluate(model,test,tc.objective,torch.device(device),config.get('batch_size',64))
    r=best['result'];r.update(config=config,test=met,test_deferred=False,selected_lr=best['lr'],resources=model.resources(),
           search=[{'lr':x['lr'],'validation_loss':x['validation_loss'],'test_evaluated':False} for x in rows],
           search_wall_seconds=sum(x['result']['wall_seconds'] for x in rows))
    if tc.objective=='reconstruction' and test['y'].ndim==4 and 'mask' in test:
        sq=(p.numpy()-test['y'])**2;mask=test['mask'][:,-1]
        for key,v in [('visible',mask),('currently_missing',1-mask)]:r['test'][key+'_mse']=float((sq*v).sum()/max(1,v.sum()))
    np.savez_compressed(out/'predictions.npz',prediction=p.numpy(),target=test['y'])
    held=[]
    if not config.get('data_root'):
        for change in config.get('heldout',[]):
            ht=TaskConfig(**{**asdict(tc),**change})
            if model.cfg.readout=='trajectory' and ht.T!=tc.T:
                held.append({'condition':change,'status':'incompatible_fixed_T_head'});continue
            data=generate(ht,config.get('heldout_size',256),'heldout')
            hmet,_=evaluate(model,data,tc.objective,torch.device(device),config.get('batch_size',64))
            held.append({'condition':change,'test':hmet,'refitted':False})
    if held:write_json(out/'heldout.json',held)
    write_json(out/'result.json',r);return r

def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True)
    m=BenchmarkModel(ck['input_shape'],ck['output_shape'],ck['T'],ModelConfig(**ck['model']),ck['objective']).to(device)
    m.metric_data_range=ck.get('data_range',1.)
    m.load_state_dict(ck['state_dict'],strict=True);m.eval();return m,ck

def diagnostic_datasets(checkpoint,n=128,data_root=None):
    """Use the actual prepared archive for real-data checkpoints, never a toy substitute."""
    root=data_root or checkpoint.get('data_root')
    if root:
        data=load_splits(root,checkpoint['objective'])
        for split,a in data.items():
            if tuple(a['x'].shape[2:])!=tuple(checkpoint['input_shape']) or a['x'].shape[1]!=checkpoint['T']:
                raise ValueError(f'{split}: diagnostic data and checkpoint input shape/time differ')
            N=len(a['x'])
            data[split]={k:(v[:n] if isinstance(v,np.ndarray) and v.ndim and len(v)==N else v)
                         for k,v in a.items()}
        return data
    tc=TaskConfig(**checkpoint['task'])
    data={s:generate(tc,n,s) for s in ('train','val','test')}
    if tuple(data['train']['x'].shape[2:])!=tuple(checkpoint['input_shape']):
        raise ValueError('checkpoint has no matching synthetic input; supply --data-root explicitly')
    return data

def export_features(model,datasets,out,device='cpu',batch=32):
    """Layer traces with explicit spatial averaging, without refitting the encoder."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for split,a in datasets.items():
            packs={}
            for i in range(0,len(a['x']),batch):
                x=torch.tensor(a['x'][i:i+batch],dtype=torch.float32,device=device);model(x,capture=True)
                for layer,tr in model.traces().items():
                    for key,v in tr.items():
                        if v.ndim>3:v=v.flatten(3).mean(-1)
                        packs.setdefault(f'{layer}__{key}',[]).append(v.cpu().numpy())
            values={k:np.concatenate(v) for k,v in packs.items()}
            values.update(y=a['y'],target=a.get('target',a['y']))
            if 'distractor' in a:values['distractor']=a['distractor']
            np.savez_compressed(out/f'{split}.npz',**values)
    write_json(out/'metadata.json',{'spatial_trace_reduction':'mean for archived diagnostics; full spatial state exists in model.traces()',
               'independent_sequence_splits':True,'privileged_internal_observations':True,
               'target_semantics':'provided analysis target; otherwise task label, not an invented latent variable'})
