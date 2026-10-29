"""Checkpoint-linked L1--L4 analysis on disjoint fit/select/evaluation scenes.

Every table records its estimand. A finite probe, compressed observation, or
Gaussian reference never certifies absence of information in the complete SNN.
"""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import DataStore,generate,source_pair,subset,render_tagged
from .engine import load_run,numpy_losses,implementation_hash,forecast_metrics
from .runtime import collect,tensor_input,intervene,message_edit,event_edit
from .models import Cell,MemoryBackbone
from .utils import atomic_json,digest,file_hash,seeded,exclusive_file
from .kernels.linear_reference import fit_ridge,predict_ridge,regression_metrics
from .kernels.decomposition import fit_joint,apply_joint,covariance_accounting,representation_score
from .kernels.subspace_geometry import basis,overlap
from .kernels.temporal_alignment import fit_transport,transport
from .kernels.functional_anova import decompose,power_summary
from .kernels.recovery_math44 import four_losses,output_decomposition
from .kernels.spatial_metrics45 import frequency_losses,spectral_angle,psnr_summary
from .kernels.continuous_risk import continuous_moments,exact_continuous_risk
from .kernels.task_spectrum import crossfit_task_spectrum
from .kernels.geometry_generalization import sample_geometry,geometry_terms,predict_error


def interval(values,draws,delta,seed=4201,groups=None):
    """Group bootstrap; overlapping windows are not independent source groups."""
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Independent finite scene contrasts')
    ids=np.arange(len(x)) if groups is None else np.asarray(groups)
    if ids.shape!=(len(x),):raise ValueError('One source group per scene')
    _,assignment=np.unique(ids,return_inverse=True);ng=int(assignment.max())+1
    totals=np.bincount(assignment,weights=x);counts=np.bincount(assignment)
    if ng<2:return {'mean':float(x.mean()),'interval':None,'n_scenes':len(x),'independent_groups':ng,'method':'insufficient independent groups for an interval','delta':delta}
    rng=seeded(seed,'bootstrap');means=[]
    for _ in range(draws):
        index=rng.integers(0,ng,ng);means.append(totals[index].sum()/counts[index].sum())
    return {'mean':float(x.mean()),'interval':np.quantile(means,[delta/2,1-delta/2]),'n_scenes':len(x),
            'method':'source-group percentile bootstrap; marginal exploratory interval','independent_groups':ng,'delta':delta}


def source_schema(meta):
    offsets=np.cumsum([0]+meta['source_dims']);requested=set(meta['requested_sources']);first=meta['requested_sources'][0]
    return [{'source':name,'start':int(offsets[j]),'stop':int(offsets[j+1]),'role':'focal' if name==first else 'cotarget' if name in requested else 'unrequested',
             'requested_coordinates':meta.get('requested_coordinate_indices',{}).get(name,list(range(meta['source_dims'][j])) if name in requested else []),
             'physical_coordinates':meta.get('physical_coordinate_indices',{}).get(name,None)}
            for j,name in enumerate(meta['source_names'])]


def coordinate_mean(values):
    values=np.asarray(values);finite=values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def diagnostic_targets(data,budget):
    """Fixed, role-stratified coordinate budget; native y/loss remain complete."""
    first=next(iter(data.values()));meta=first['meta'];dims=meta['source_dims'];total=sum(dims)
    if total<=budget:return data,{'original_dimension':total,'diagnostic_dimension':total,'selection':'all target coordinates'}
    if len(dims)>budget:raise ValueError('Diagnostic target budget must cover at least one coordinate per source')
    allocation=np.ones(len(dims),int)
    while allocation.sum()<budget:
        eligible=np.array(dims)>allocation;ratios=np.where(eligible,np.array(dims)/allocation,-1);allocation[int(np.argmax(ratios))]+=1
    columns=[];offset=0;by_source={};requested={};physical={}
    for name,d,n in zip(meta['source_names'],dims,allocation):
        req=meta.get('requested_coordinate_indices',{}).get(name,list(range(d)) if name in meta['requested_sources'] else [])
        rng=seeded(319,'target_coordinates',name,d);take=rng.choice(req,min(int(n),len(req)),replace=False).tolist()
        remaining=[i for i in range(d) if i not in set(take)];take+=rng.choice(remaining,int(n)-len(take),replace=False).tolist()
        idx=np.sort(take);columns.extend((offset+idx).tolist());by_source[name]=idx.tolist();offset+=d
        requested[name]=[j for j,i in enumerate(idx) if i in set(req)]
        phys=meta.get('physical_coordinate_indices',{}).get(name)
        if phys is not None:physical[name]=[j for j,i in enumerate(idx) if i in set(phys)]
    result={}
    for split,batch in data.items():result[split]=dict(batch,targets=np.asarray(batch['targets'][:,columns]),meta=dict(batch['meta'],source_dims=allocation.tolist(),requested_coordinate_indices=requested,physical_coordinate_indices=physical))
    return result,{'original_dimension':total,'diagnostic_dimension':budget,'selected_coordinates_by_source':by_source,
        'selection':'fixed label-independent stratified coordinate subset; not all native pixels or a complete latent space'}


def collect_store(model,data,batch_size,max_features,device,storage_dir=None):
    pred=[];seq=[];traces={};selectors={};resources=None
    if storage_dir is not None:storage_dir=Path(storage_dir);storage_dir.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(data['x']),batch_size):
        b=subset(data,np.arange(start,min(start+batch_size,len(data['x']))));local={};handles=[]
        def hook(site):
            def save(module,args,output):
                for channel,value in module.last.items():
                    if channel not in ('pre','post','output','a','syn','h','c0','h0','margin','emit_event','reset_event','base_current','effective_current'):continue
                    x=value.detach().flatten(1);key=site+'::'+channel;d=x.shape[-1]
                    idx=np.arange(d) if d<=max_features else np.sort(seeded(618,'feature_coordinates',site,d).choice(d,max_features,replace=False))
                    selectors[key]={'original_features':d,'selected_indices':idx.tolist(),'complete_observation':len(idx)==d,
                                    'method':'all coordinates' if len(idx)==d else 'fixed label-independent coordinate subset'}
                    local.setdefault(key,[]).append(x[:,idx].cpu().numpy())
            return save
        for site,module in model.temporal_owners().items():handles.append(module.register_forward_hook(hook(site)))
        try:
            model.eval()
            with torch.no_grad():prediction=model(tensor_input(b,device))
            p=tuple(v.cpu().numpy() for v in prediction) if isinstance(prediction,tuple) else prediction.cpu().numpy()
        finally:
            for handle in handles:handle.remove()
        if isinstance(p,tuple):pred.append(p[0]);seq.append(p[1])
        else:pred.append(p)
        for key,values in local.items():
            value=np.stack(values,1)
            if storage_dir is None:traces.setdefault(key,[]).append(value)
            else:
                if key not in traces:traces[key]=np.lib.format.open_memmap(storage_dir/(digest(key)[:20]+'.npy'),mode='w+',dtype=value.dtype,shape=(len(data['x']),*value.shape[1:]))
                traces[key][start:start+len(value)]=value
        resources=model.resources()
    prediction=np.concatenate(pred)
    if seq:prediction=(prediction,np.concatenate(seq))
    if storage_dir is not None:
        for v in traces.values():v.flush()
        atomic_json(storage_dir/'index.json',{key:{'file':digest(key)[:20]+'.npy',**selectors[key]} for key in traces})
        traces={key:np.load(storage_dir/(digest(key)[:20]+'.npy'),mmap_mode='r',allow_pickle=False) for key in traces}
    else:traces={k:np.concatenate(v) for k,v in traces.items()}
    return {'prediction':prediction,'traces':traces,
            'observation_contracts':selectors,'resources':resources}


def select_probe(x,y,v,vy,alphas):
    # Normalization is fitted on probe_train, never on selection/evaluation rows.
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    xx=(x-mean)/scale;vv=(v-mean)/scale;candidates=[]
    for alpha in alphas:
        fit=fit_ridge(xx,y,alpha);score=float(np.mean((predict_ridge(fit,vv)-vy)**2));candidates.append((score,float(alpha),fit))
    _,alpha,fit=min(candidates,key=lambda item:(item[0],-item[1]))
    return {'fit':fit,'mean':mean,'scale':scale,'alpha':alpha,'selection_MSE':min(x[0] for x in candidates)}


def probe_predict(probe,x):return predict_ridge(probe['fit'],(x-probe['mean'])/probe['scale'])


def fitted_probes(groups,data,schema,a):
    fit,sel,ev=(groups[k] for k in ('probe_train','probe_val',a.eval_split));yy=[np.asarray(data[k]['targets'],float) for k in ('probe_train','probe_val',a.eval_split)]
    rows=[];readers={};matrices={}
    for key,x in fit['traces'].items():
        if not key.endswith(('::pre','::post','::output','::a','::syn','::h0','::c0','::h')):continue
        v,e=sel['traces'][key],ev['traces'][key];T=x.shape[1];scores=np.full((T,T,len(schema)),np.nan)
        for t in range(T):
            p=select_probe(x[:,t],yy[0],v[:,t],yy[1],a.ridge_alphas);prediction=probe_predict(p,e[:,t]);m=regression_metrics(prediction,yy[2])
            roles={s['source']:coordinate_mean(m['per_coordinate_r2'][s['start']:s['stop']]) for s in schema}
            demanded={s['source']:coordinate_mean(m['per_coordinate_r2'][np.array(s['requested_coordinates'],int)+s['start']]) if s['requested_coordinates'] else None for s in schema}
            rows.append({'site_channel':key,'time':t,'alpha':p['alpha'],'mean_MSE':m['mean_mse'],'source_R2_all_diagnostic_coordinates':roles,'requested_coordinate_R2':demanded,
                         'available_source_fraction':{s['source']:float(np.mean(data[a.eval_split]['source_present'][:,j] & (data[a.eval_split]['source_times'][:,j]<=t))) for j,s in enumerate(schema)},
                         'negative_R2_retained':True,'scope':'linear recovery of analysis targets; native labels are separately evaluated'})
            readers[(key,t)]=p
            for u in range(T):
                r=regression_metrics(probe_predict(p,e[:,u]),yy[2])['per_coordinate_r2']
                for j,s in enumerate(schema):scores[t,u,j]=coordinate_mean(r[s['start']:s['stop']])
        matrices[key]=scores
    return rows,readers,matrices


def nonlinear_probe(fit,sel,ev,y,vy,ey,steps,seed):
    torch.manual_seed(seed);mean=fit.mean(0);scale=np.maximum(fit.std(0),1e-6)
    x=torch.tensor((fit-mean)/scale,dtype=torch.float32);v=torch.tensor((sel-mean)/scale,dtype=torch.float32);e=torch.tensor((ev-mean)/scale,dtype=torch.float32)
    target=torch.tensor(y,dtype=torch.float32);model=nn.Sequential(nn.Linear(x.shape[1],min(128,2*x.shape[1]+8)),nn.Tanh(),nn.Linear(min(128,2*x.shape[1]+8),target.shape[1]))
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001);best=None;bestloss=float('inf');beststep=0
    for step in range(steps):
        opt.zero_grad();loss=(model(x)-target).square().mean();loss.backward();opt.step()
        with torch.no_grad():vl=float((model(v)-torch.tensor(vy,dtype=torch.float32)).square().mean())
        if vl<bestloss:bestloss=vl;best={k:z.detach().clone() for k,z in model.state_dict().items()};beststep=step+1
    model.load_state_dict(best)
    with torch.no_grad():pred=model(e).numpy()
    return {'metrics':regression_metrics(pred,ey),'selected_step':beststep,'selection_MSE':bestloss,
            'scope':'finite trained nonlinear decoder; failure is not an information-absence certificate'}


def geometry(groups,data,schema,a):
    keys=[k for k in groups['probe_train']['traces'] if k.endswith(('::pre','::post','::output','::a','::syn','::h0','::c0','::h'))];out=[]
    z=np.asarray(data['probe_train']['targets'],float);ez=np.asarray(data[a.eval_split]['targets'],float)
    roles=[s['role'] if i in set(s['requested_coordinates']) else 'unrequested' for s in schema for i in range(s['stop']-s['start'])]
    for key in keys:
        x=groups['probe_train']['traces'][key][:,-1];e=groups[a.eval_split]['traces'][key][:,-1]
        fit=fit_joint(x,z,roles);parts=apply_joint(fit,e,ez);account=covariance_accounting(e,parts)
        cov=account['total_covariance'];values=np.linalg.eigvalsh(cov);axes={}
        for role in ('focal','cotarget','unrequested'):axes[role]=basis(fit['a'][:,np.array(roles)==role])
        overlaps={r+'__'+s:overlap(axes[r],axes[s]) for r,s in (('focal','cotarget'),('focal','unrequested'),('cotarget','unrequested'))}
        out.append({'site_channel':key,'coefficient_identifiable':fit['coefficients_identifiable'],'design_rank':fit['design_rank'],
            'coordinate_role_counts':{r:roles.count(r) for r in ('focal','cotarget','unrequested')},
            'linear_encoding':representation_score(e,e-parts['unexplained_by_joint_linear_model'],fit['rmean']),
            'component_trace':account['component_trace'],'cross_trace':account['cross_trace'],
            'covariance_identity_max_error':account['max_covariance_identity_error'],'subspace_overlaps':overlaps,
            'participation_ratio':float(values.sum()**2/(values@values)) if values@values>0 else 0.,
            'scope':'joint statistical encoding with roles from the task contract; no causal attribution'})
    return out


def temporal(groups,data,readers,a):
    rows=[];f=groups['probe_train'];ev=groups[a.eval_split];y=data[a.eval_split]['targets']
    for key,x in f['traces'].items():
        if (key,0) not in readers:continue
        e=ev['traces'][key];source_time=x.shape[1]-1;p=readers[(key,source_time)]
        for dest in range(x.shape[1]):
            for family in ('raw','center','orthogonal','similarity','affine'):
                transport_fit=fit_transport(x[:,dest],x[:,source_time],family)
                pred=probe_predict(p,transport(transport_fit,e[:,dest]));scores=regression_metrics(pred,y)
                rows.append({'site_channel':key,'reader_time':source_time,'observed_time':dest,'transport':family,
                    'MSE':scores['mean_mse'],'rank':transport_fit['rank_cross_covariance'],
                    'label_access_for_alignment':False,'fit_scenes':len(x),'geometry_direction':'observed time -> reader time'})
    return rows


def _predict(model,batch,a,device,after_factory=None,context_factory=None):
    from contextlib import nullcontext
    predictions=[];seq=[]
    with torch.no_grad():
        model.eval()
        for start in range(0,len(batch['x']),a.batch_size):
            ids=np.arange(start,min(start+a.batch_size,len(batch['x'])));b=subset(batch,ids)
            ctx=nullcontext() if context_factory is None else context_factory(ids)
            with ctx:p=model(tensor_input(b,device),after_step=None if after_factory is None else after_factory(ids))
            if isinstance(p,tuple):predictions.append(p[0].cpu().numpy());seq.append(p[1].cpu().numpy())
            else:predictions.append(p.cpu().numpy())
    pred=np.concatenate(predictions)
    return (pred,np.concatenate(seq)) if seq else pred


def interventions(model,batch,clean,a,device):
    baseline=numpy_losses(clean,batch)[0];rows=[];raw={};T=batch['x'].shape[1]
    times=sorted(set([max(0,T//3-1),max(0,2*T//3-1),T-2]))
    for site in model.temporal_owners():
        for t in times:
            for state,message,label in ((True,False,'state'),(False,True,'message'),(True,True,'state_and_message')):
                predictions=[];sequences=[]
                for start in range(0,len(batch['x']),a.batch_size):
                    b=subset(batch,np.arange(start,min(start+a.batch_size,len(batch['x']))))
                    p=intervene(model,b,site,t,edit_state=state,edit_message=message,device=device)
                    if isinstance(p,tuple):predictions.append(p[0]);sequences.append(p[1])
                    else:predictions.append(p)
                pred=np.concatenate(predictions);pred=(pred,np.concatenate(sequences)) if sequences else pred
                loss=numpy_losses(pred,batch)[0];key=site+'|'+str(t)+'|'+label;raw[key]=loss
                rows.append({'site':site,'time':t,'edit':label,'replacement':'zero','loss_change':interval(loss-baseline,a.interval_resamples,a.delta,groups=batch['group_ids']),
                    'phase':'after-step recurrent state; module output before consumers','scope':'local diagnostic intervention; zero may be off-distribution'})
    return rows,raw


def mechanism(model,batch,clean,a,device):
    baseline=numpy_losses(clean,batch)[0];rows=[];raw={};T=batch['x'].shape[1]
    common=[int(t) for j in range(batch['source_times'].shape[1]) if np.all(batch['source_times'][:,j]==batch['source_times'][0,j]) for t in [batch['source_times'][0,j]] if 0<=t<T-1]
    times=sorted(set(common+[max(0,T-2)]))
    for site,cell in model.temporal_owners().items():
        if not isinstance(cell,Cell) or cell.kind in ('continuous','tanh'):continue
        for t in times:
            for branch in ('emit','reset','aux'):
                pred=_predict(model,batch,a,device,context_factory=lambda ids,s=site,u=t,b=branch:event_edit(model,s,u,b,0))
                loss=numpy_losses(pred,batch)[0];key=site+'|'+str(t)+'|'+branch;raw[key]=loss
                rows.append({'site':site,'time':t,'branch':branch,'forced_event':0,'loss_change':interval(loss-baseline,a.interval_resamples,a.delta,groups=batch['group_ids']),
                    'scope':'diagnostic split of natural emission, reset and auxiliary update branches; one branch overridden, others remain natural'})
    return {'status':'computed' if rows else 'not_applicable','rows':rows,'times':times,'reason':None if rows else 'Continuous baseline has no binary event branches'},raw


def realized_current_accounting(model,calibration):
    """Exact realized-current expansion, with observed reset gates, zero start.

    This is an algebraic identity on a realized path. Reset gates depend on the
    input, so coefficients must not be renamed independent causal influences.
    """
    rows=[];traces=calibration['traces']
    for site,cell in model.temporal_owners().items():
        if not isinstance(cell,Cell) or cell.kind=='tanh':continue
        key=site+'::post'
        if key not in traces:continue
        I=traces[site+'::effective_current'].astype(float);post=traces[key].astype(float)
        b=torch.sigmoid(cell.raw_beta).detach().cpu().numpy() if hasattr(cell,'raw_beta') else cell.fixed_beta.detach().cpu().numpy()
        d=calibration['observation_contracts'][key]['original_features'];indices=calibration['observation_contracts'][key]['selected_indices']
        b=np.full(len(indices),float(b)) if np.ndim(b)==0 else np.repeat(b,d//len(b))[indices]
        T=I.shape[1];weights=[];contributions=[]
        if cell.kind=='lif_soft':
            event=traces[site+'::reset_event'];reset=cell.reset_strength*cell.threshold*event
            reconstructed=sum((I[:,k]-reset[:,k])*b**(T-1-k) for k in range(T))
            contribution=[float(np.mean(np.abs(I[:,k]*b**(T-1-k)))) for k in range(T)]
            branch='soft reset: current and threshold-subtraction sums'
        else:
            event=np.zeros_like(I) if cell.kind in ('threshold','continuous') else traces[site+'::reset_event']
            for k in range(T):
                survival=np.prod(1-cell.reset_strength*event[:,k:],axis=1);w=b**(T-1-k)*survival;weights.append(w);contributions.append(I[:,k]*w)
            reconstructed=sum(contributions);contribution=[float(np.mean(np.abs(x))) for x in contributions];branch='hard/no reset: leak times realized reset survival'
        rows.append({'site':site,'cell':cell.kind,'identity':branch,'max_abs_final_state_residual':float(np.max(np.abs(reconstructed-post[:,-1]))),
            'mean_absolute_current_contribution_by_write_step':contribution,'coordinate_contract':calibration['observation_contracts'][key],
            'scope':'effective cell currents including upstream computation, bias, and any auxiliary-state filtering; not original-source causal shares'})
    return rows


def communication(model,batch,clean,a,device):
    baseline=numpy_losses(clean,batch)[0];rows=[];raw={};t=batch['x'].shape[1]-2
    # Actual consumer inputs, including residual branches and attention projections.
    for name,module in model.named_modules():
        if not isinstance(module,(nn.Linear,nn.Conv1d,nn.Conv2d)) or name.startswith(('head','encoder')):continue
        pred=_predict(model,batch,a,device,context_factory=lambda ids,n=name:message_edit(model,n,t,lambda v:torch.zeros_like(v),kind='edge'))
        loss=numpy_losses(pred,batch)[0];raw[name]=loss
        rows.append({'consumer':name,'time':t,'operation':'zero actual tensor input on this consumer edge',
            'loss_change':interval(loss-baseline,a.interval_resamples,a.delta,groups=batch['group_ids']),
            'scope':'one declared module input; all other consumers of a shared producer remain intact'})
    return rows,raw


def recovery(model,task,batch,clean,a,device):
    if task.family!='tagged':return {'status':'not_applicable','reason':'Exact independent source-world renderer required; real/spatial input damage needs its own intervention contract'},{}
    # Fixed task-selected source, site, and write time. Evaluation never selects these.
    source=batch['meta']['requested_sources'][0];j=batch['meta']['source_names'].index(source)
    damaged=source_pair(task,batch,source,41);damaged['y']=batch['y'];damaged['loss_mask']=batch['loss_mask']
    sites=list(model.temporal_owners());site=sites[0];times=batch['source_times'][:,j]
    if len(set(times.tolist()))!=1:return {'status':'not_applicable','reason':'This fixed-time recovery contract requires a common source-write step'},{}
    t=int(times[0]);p0=[];pd=[];pp=[];pr=[];seqs=[[],[],[],[]]
    for start in range(0,len(batch['x']),a.batch_size):
        ids=np.arange(start,min(start+a.batch_size,len(batch['x'])));b=subset(batch,ids);d=subset(damaged,ids)
        cc=collect(model,b,device,snapshots=True);dc=collect(model,d,device,snapshots=True)
        delta={k:cc['snapshots'][t]['owners'][site][k]-v for k,v in dc['snapshots'][t]['owners'][site].items()}
        def patch(m,u):
            if u==t:
                owner=m.temporal_owners()[site];owner.state={k:v+delta[k].to(v) for k,v in owner.state.items()}
        with torch.no_grad():
            p=model(tensor_input(b,device),after_step=patch);r=model(tensor_input(d,device),after_step=patch)
        cells=[cc['prediction'],dc['prediction'],p,r]
        for idx,(cell,dest) in enumerate(zip(cells,(p0,pd,pp,pr))):
            if isinstance(cell,tuple):
                dest.append(cell[0] if isinstance(cell[0],np.ndarray) else cell[0].cpu().numpy());seqs[idx].append(cell[1] if isinstance(cell[1],np.ndarray) else cell[1].cpu().numpy())
            else:dest.append(cell if isinstance(cell,np.ndarray) else cell.cpu().numpy())
    cells=[np.concatenate(c) for c in (p0,pd,pp,pr)]
    preds=[(c,np.concatenate(s)) if s else c for c,s in zip(cells,seqs)]
    losses=[numpy_losses(p,batch)[0] for p in preds];stats=four_losses(*losses,damage_floor=.01)
    stats.update(status='computed',site=site,time=t,source=source,
        damage='IID replacement of one world source; original native target held fixed',
        patch='same additive complete local-state correction in both contexts',
        privilege='paired clean-state oracle; mechanism diagnostic, not a deployable learned repair',
        interaction_interval=interval(losses[1]-losses[3]-losses[0]+losses[2],a.interval_resamples,a.delta,groups=batch['group_ids']))
    if batch['meta']['target_mode']=='continuous' and cells[0].ndim==2:stats['output_accounting']=output_decomposition(*cells,batch['y'])
    return stats,{k:v for k,v in zip(('R00','R10','R01','R11'),losses)}


def reconstruction(pred,batch):
    if isinstance(pred,tuple):pred=pred[0]
    if pred.ndim!=4 or batch['y'].ndim!=4:return {'status':'not_applicable','reason':'Native NCHW reconstruction target required'},{}
    p=np.asarray(pred,float);y=np.asarray(batch['y'],float);err=(p-y)**2;clipped=np.clip(p,0,1);raw={'full':err.mean((1,2,3))};masks={}
    for name,mask in batch.items():
        if not name.startswith('region_'):continue
        m=np.broadcast_to(mask,p.shape);count=m.sum((1,2,3));eligible=count>0
        losses=np.full(len(p),np.nan);losses[eligible]=(err*m).sum((1,2,3))[eligible]/count[eligible];raw[name]=losses
        masks[name]={'mean_raw_MSE':float(np.nanmean(losses)) if eligible.any() else None,'eligible_scenes':int(eligible.sum()),'mean_pixels_per_scene':float(count.mean()/p.shape[1])}
    # Full-image DCT is exact; regional low/high cross terms are retained when masks are shared.
    common={k:np.asarray(v[0,0],bool) for k,v in batch.items() if k.startswith('region_') and np.all(v==v[0])}
    freq=frequency_losses(p,y,common)
    result={'status':'computed','raw_MSE':float(raw['full'].mean()),'clipped_MSE':float(np.mean((clipped-y)**2)),
        'out_of_range_fraction':float(np.mean((p<0)|(p>1))),'regions':masks,'DCT':{k:float(v.mean()) for k,v in freq.items()},
        'PSNR_peak1_clipped':psnr_summary(np.mean((clipped-y)**2,(1,2,3))),
        'scope':'PSNR peak=1 requires data normalized to [0,1]; raw risk remains separately available'}
    if p.shape[1]>1:result['spectral_angle_clipped']=spectral_angle(clipped,y)
    raw.update(freq)
    return result,raw


def anova(model,task,a,device):
    if task.family!='tagged':return {'status':'not_applicable','reason':'Complete product-distribution pick-freeze renderer is implemented for tagged Gaussian worlds only'}
    A=generate(task,a.anova_scenes,'probe_train');B=generate(replace(task,data_seed=task.data_seed+73001),a.anova_scenes,'probe_train')
    requested=set(A['meta']['requested_sources']);focal=A['meta']['requested_sources'][0]
    groups={'focal':[],'cotarget':[],'unrequested':[],'observation_noise':[]}
    for j,name in enumerate(A['meta']['source_names']):
        demand=set(A['meta']['requested_coordinate_indices'][name])
        for coordinate in range(A['z'].shape[-1]):
            role=('focal' if name==focal else 'cotarget') if coordinate in demand else 'unrequested'
            groups[role].append((j,coordinate))
    names=[k for k,v in groups.items() if v or k=='observation_noise'];values=[]
    for mask in range(2**len(names)):
        z=B['z'].copy();noise=B['noise'].copy()
        for j,name in enumerate(names):
            if mask&(1<<j):
                if name=='observation_noise':noise=A['noise']
                else:
                    for source,coordinate in groups[name]:z[:,source,coordinate]=A['z'][:,source,coordinate]
        batch=dict(A,x=render_tagged(task,z,noise,A['source_times'],A['source_present']))
        pred=_predict(model,batch,a,device);pred=pred[0] if isinstance(pred,tuple) else pred
        output_indices=np.flatnonzero(A['loss_mask'][0].reshape(-1)>0);values.append(pred.reshape(len(z),-1)[:,output_indices])
    fit=decompose(np.stack(values),names,estimator='symmetric_difference');summary=power_summary(fit)
    return {'status':'computed','summary':summary,'native_supervised_output_indices':output_indices.tolist(),'coordinate_groups':groups,'covariance_closure_max_error':fit['covariance_closure_max_error'],
        'negative_estimates_retained':True,'scope':'grouped output functional ANOVA conditional on the sampled schedule; sources/noise are independent, source presence and timing are held fixed'}


def theory(groups,data,a):
    keys=[k for k in groups['probe_train']['traces'] if k.endswith('::output')];out=[]
    for key in keys:
        x=groups['probe_train']['traces'][key][:,-1].astype(float);z=data['probe_train']['targets'].astype(float)
        mean=x.mean(0);zm=z.mean(0);x=x-mean;z=z-zm;v=groups['probe_val']['traces'][key][:,-1]-mean;vz=data['probe_val']['targets']-zm
        ex=groups[a.eval_split]['traces'][key][:,-1]-mean;ey=data[a.eval_split]['targets']-zm
        moments=continuous_moments(x,z);M=moments['M'];C=moments['C'];other=v.T@vz/len(v)
        spectrum=crossfit_task_spectrum(M,C,other);curve=[]
        geom=sample_geometry(x,z);named=geometry_terms(geom['Psi'],geom['Phi'],geom['Omega'])
        for n in sorted(set([1,8,32,128,min(len(x),512)])):
            if n>len(x):continue
            forecast=exact_continuous_risk(M,C,moments['T'],moments['target_squared_norm'],moments['target_dimension'],n)
            errs=[]
            for rep in range(16):
                idx=seeded(671,'hebbian',n,rep).integers(0,len(x),n);w=x[idx].T@z[idx]/n;errs.append(float(np.mean((ex@w-ey)**2)))
            curve.append({'n':n,'plug_in_reference':forecast,'heldout_hebbian_MSE_mean':float(np.mean(errs))})
        out.append({'site_channel':key,'spectrum':spectrum,'learning_curve':curve,
                    'named_covariance_geometry':named,'random_teacher_gaussian_reference':[predict_error(named,int(n)) for n in (8,32,128)],
                    'scope':'mixed-moment plug-in diagnostics for fixed Hebbian readers, not a theorem for the trained native decoder'})
    return out


def analyze(run,data_root='prepared',checkpoint='best.pt',device='cpu',analysis_override=None):
    with exclusive_file(Path(run)/'analysis.lock'):
        return _analyze(run,data_root,checkpoint,device,analysis_override)


def _analyze(run,data_root='prepared',checkpoint='best.pt',device='cpu',analysis_override=None):
    spec,model,ck,path=load_run(run,checkpoint,device);a=spec.analysis if analysis_override is None else analysis_override
    if a.eval_split not in ('development','confirmation','audit'):raise ValueError('Unknown evaluation role')
    # Audit is only opened by this explicit split request, never by the trainer/compiler.
    contract={'checkpoint_sha256':file_hash(path),'parameter_sha256':ck['parameter_sha256'],'analysis':a.__dict__,
              'implementation_sha256':implementation_hash('analysis'),'training_implementation_sha256':ck['implementation_sha256'],'run_id':spec.run_id}
    analysis_id=digest(contract)[:20];out=Path(run)/'analysis'/analysis_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists() and (out/'report.json').exists():
        receipt=json.loads((out/'receipt.json').read_text())
        if receipt['report_sha256']==file_hash(out/'report.json'):return out
    store=DataStore(spec.task,data_root);sizes={'calibration':a.select_scenes,'probe_train':a.fit_scenes,'probe_val':a.select_scenes,a.eval_split:a.eval_scenes}
    if 'selective' in a.modules:sizes.update(verification_train=a.fit_scenes,verification_val=a.select_scenes)
    data={k:store.ensure(k,n) for k,n in sizes.items()}
    data,target_contract=diagnostic_targets(data,a.max_targets)
    # No shared source group may cross fitting/selection/evaluation boundaries.
    allgroups={k:set(b['group_ids'].tolist()) for k,b in data.items()}
    for k,ids in allgroups.items():
        for other,jds in allgroups.items():
            if k<other and ids&jds:raise ValueError('Source group leakage: '+k+' / '+other)
    captured={k:collect_store(model,b,a.batch_size,a.max_features,device,out/'observations'/k) for k,b in data.items()};ev=data[a.eval_split];ec=captured[a.eval_split]
    loss,correct=numpy_losses(ec['prediction'],ev);report={'contract':contract,'analysis_id':analysis_id,'native':{'loss':interval(loss,a.interval_resamples,a.delta,groups=ev['group_ids']),'accuracy':None if correct is None else float(correct.mean())},
        'data_roles':{k:{'scenes':len(v['x']),'groups':len(allgroups[k])} for k,v in data.items()},'source_schema':source_schema(ev['meta']),
        'observation_contracts':ec['observation_contracts'],'diagnostic_target_contract':target_contract,'resources':ec['resources'],'modules':{}}
    raw={'native_loss':loss,'scene_ids':ev['scene_ids'],'group_ids':ev['group_ids']};schema=report['source_schema'];modules=report['modules'];readers=None
    forecast=forecast_metrics(ec['prediction'],ev)
    if forecast:
        report['native']['forecasting']={name:interval(values,a.interval_resamples,a.delta,groups=ev['group_ids']) for name,values in forecast.items()}
        raw.update({'forecast_'+name:values for name,values in forecast.items()})
    if 'task_controls' in a.modules:
        from .readouts import task_controls
        modules['task_controls']=task_controls(data,a)
        from .oracles import task_reference
        modules['task_controls']['population_observation_reference']=task_reference(spec.task)
    if 'working_point' in a.modules:
        modules['working_point']={key:{'mean':float(v.mean()),'std':float(v.std()),'zero_fraction':float(np.mean(v==0)),'q01_q50_q99':np.quantile(v,[.01,.5,.99])} for key,v in captured['calibration']['traces'].items()}
    if 'mechanism' in a.modules:
        modules['mechanism'],rows=mechanism(model,ev,ec['prediction'],a,device);raw.update({'mechanism|'+k:v for k,v in rows.items()})
        modules['mechanism']['realized_current_accounting']=realized_current_accounting(model,captured['calibration'])
    if 'probes' in a.modules or 'temporal' in a.modules:
        rows,readers,tgm=fitted_probes(captured,data,schema,a);modules['probes']={'rows':rows,'TGM_orientation':'train time, evaluation time, source',
            'label_shuffle_control':{},'nonlinear_control':{}}
        for key,matrix in tgm.items():raw['TGM|'+key]=matrix
        outputkeys=[k for k in captured['probe_train']['traces'] if k.endswith('::output')]
        if outputkeys:
            key=outputkeys[-1];x=captured['probe_train']['traces'][key][:,-1];v=captured['probe_val']['traces'][key][:,-1];e=ec['traces'][key][:,-1]
            y=data['probe_train']['targets'];vy=data['probe_val']['targets'];ey=ev['targets']
            shuffled=y[seeded(881,'shuffle_labels').permutation(len(y))];p=select_probe(x,shuffled,v,vy,a.ridge_alphas)
            modules['probes']['label_shuffle_control']=regression_metrics(probe_predict(p,e),ey)
            if a.mlp_steps>0:modules['probes']['nonlinear_control']=nonlinear_probe(x,v,e,y,vy,ey,a.mlp_steps,spec.train.seed+1000)
    if 'geometry' in a.modules:modules['geometry']=geometry(captured,data,schema,a)
    if 'temporal' in a.modules:modules['temporal']=temporal(captured,data,readers,a)
    if 'access' in a.modules:
        from .readouts import access_controls
        modules['access']=access_controls(captured,data,a)
    if 'selective' in a.modules:
        from .selective import selective_analysis
        modules['selective'],rows=selective_analysis(model,captured,data,schema,a,device);raw.update({'selective|'+k:v for k,v in rows.items()})
    if 'communication' in a.modules:
        modules['communication'],rows=communication(model,ev,ec['prediction'],a,device);raw.update({'edge|'+k:v for k,v in rows.items()})
    if 'interventions' in a.modules:
        modules['interventions'],rows=interventions(model,ev,ec['prediction'],a,device);raw.update({'edit|'+k:v for k,v in rows.items()})
    if 'recovery' in a.modules:
        modules['recovery'],rows=recovery(model,spec.task,ev,ec['prediction'],a,device);raw.update({'recovery|'+k:v for k,v in rows.items()})
    if 'reconstruction' in a.modules:
        modules['reconstruction'],rows=reconstruction(ec['prediction'],ev);raw.update({'reconstruction|'+k:v for k,v in rows.items()})
    if 'anova' in a.modules:modules['anova']=anova(model,spec.task,a,device)
    if 'theory' in a.modules:modules['theory']=theory(captured,data,a)
    # Persist the observations needed to reproduce the fitted analysis, not just summaries.
    for split,c in captured.items():
        np.savez_compressed(out/'observations'/split/'labels_and_ids.npz',targets=data[split]['targets'],scene_ids=data[split]['scene_ids'],
                            group_ids=data[split]['group_ids'],source_times=data[split]['source_times'],source_present=data[split]['source_present'])
    np.savez_compressed(out/'scene_statistics.npz',**raw);atomic_json(out/'report.json',report)
    pred=ec['prediction'];native={'prediction':pred[0] if isinstance(pred,tuple) else pred,'target':ev['y'],'loss_mask':ev['loss_mask'],'scene_ids':ev['scene_ids'],'group_ids':ev['group_ids']}
    if isinstance(pred,tuple):native['sequence_prediction']=pred[1]
    np.savez_compressed(out/'native_outputs.npz',**native)
    atomic_json(out/'receipt.json',{'status':'analyzed','contract':contract,'report_sha256':file_hash(out/'report.json'),
        'statistical_scope':'Exploratory marginal bootstrap intervals. Confirmatory families and stopping rules must be frozen before audit.'})
    return out
