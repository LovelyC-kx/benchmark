"""Fitted local-state erasure, matched controls, and independent verification."""
import numpy as np
import torch
from .kernels.intervention_math43 import fit_eraser,apply_eraser,random_eraser,norm_matched


def selective_analysis(model,groups,data,schema,a,device):
    from .analysis import select_probe,probe_predict,interval,_predict,coordinate_mean
    from .engine import numpy_losses
    from .kernels.linear_reference import regression_metrics
    owners=model.temporal_owners();ev=data[a.eval_split];base=numpy_losses(groups[a.eval_split]['prediction'],ev)[0];rows=[];raw={}
    source=next(s for s in schema if s['role']=='focal');target=np.array(source['requested_coordinates'],int)+source['start'];time=ev['x'].shape[1]-2
    for site,owner in owners.items():
        key=site+'::post' if site+'::post' in groups['probe_train']['traces'] else site+'::h' if site+'::h' in groups['probe_train']['traces'] else site+'::h0'
        if key not in groups['probe_train']['traces']:continue
        statekey='v' if key.endswith('::post') else 'h' if key.endswith('::h') else 'h0'
        h=groups['probe_train']['traces'][key][:,time];z=data['probe_train']['targets'][:,target];fit=fit_eraser(h,z)
        selection=groups['probe_val']['traces'][key][:,time];ops={'leace':fit,'matched_rank_random':random_eraser(fit,522),'orthogonal':fit_eraser(h,z,'orth')}
        selected_indices=groups['probe_train']['observation_contracts'][key]['selected_indices']
        vertrain=groups['verification_train']['traces'][key][:,time];verval=groups['verification_val']['traces'][key][:,time];evalh=groups[a.eval_split]['traces'][key][:,time]
        for name,op in ops.items():
            # Independent decoder is trained AFTER applying the frozen operator.
            vt=apply_eraser(vertrain,op);vv=apply_eraser(verval,op);ee=apply_eraser(evalh,op)
            reader=select_probe(vt,data['verification_train']['targets'],vv,data['verification_val']['targets'],a.ridge_alphas)
            metrics=regression_metrics(probe_predict(reader,ee),ev['targets'])
            def after_factory(ids,operator=op):
                def after(m,t):
                    if t!=time:return
                    value=m.temporal_owners()[site].state[statekey];flat=value.flatten(1);new=flat.clone()
                    local=flat[:,selected_indices].detach().cpu().numpy();changed=apply_eraser(local,operator)
                    new[:,selected_indices]=torch.tensor(changed,device=value.device,dtype=value.dtype)
                    m.temporal_owners()[site].state[statekey]=new.reshape_as(value)
                return after
            pred=_predict(model,ev,a,device,after_factory=after_factory);loss=numpy_losses(pred,ev)[0];id_=site+'|'+name;raw[id_]=loss
            rows.append({'site':site,'time':time,'state_key':statekey,'source':source['source'],'operator':name,'removed_rank':op['removed_rank'],
                'fit_cross_covariance_residual':float(np.linalg.norm(op['matrix'].T@((h-h.mean(0)).T@(z-z.mean(0))/(len(h)-1)))),
                'independent_verification_R2':{s['source']:coordinate_mean(metrics['per_coordinate_r2'][s['start']:s['stop']]) for s in schema},
                'requested_coordinate_verification_R2':{s['source']:coordinate_mean(metrics['per_coordinate_r2'][np.array(s['requested_coordinates'],int)+s['start']]) if s['requested_coordinates'] else None for s in schema},
                'native_loss_change':interval(loss-base,a.interval_resamples,a.delta,groups=ev['group_ids']),'observation_contract':groups['probe_train']['observation_contracts'][key],
                'state_scope':'selected coordinates of one continuous recurrent state; auxiliary states, cache, readout and emitted message are preserved',
                'guarantee':op['guarantee'],'frozen_operator':op})
        # A control with the same scene-wise local displacement norm.
        displacement=apply_eraser(evalh,fit)-evalh;noise=np.stack([np.random.default_rng(int(sid)+902).normal(size=evalh.shape[1]) for sid in ev['scene_ids']])
        noise=noise/np.maximum(np.linalg.norm(noise,axis=1,keepdims=True),1e-20)*np.linalg.norm(displacement,axis=1,keepdims=True)
        def after_norm(ids):
            def after(m,t):
                if t==time:
                    value=m.temporal_owners()[site].state[statekey];new=value.flatten(1).clone();new[:,selected_indices]+=torch.tensor(noise[ids],device=value.device,dtype=value.dtype);m.temporal_owners()[site].state[statekey]=new.reshape_as(value)
            return after
        pred=_predict(model,ev,a,device,after_factory=after_norm);loss=numpy_losses(pred,ev)[0];raw[site+'|matched_norm_random']=loss
        rows.append({'site':site,'time':time,'operator':'matched_norm_random','native_loss_change':interval(loss-base,a.interval_resamples,a.delta,groups=ev['group_ids']),'scope':'same local scene-wise L2 displacement norm as LEACE; no equal-off-manifold-distance claim'})
    return {'status':'computed' if rows else 'not_applicable','rows':rows,'split_policy':'operator: probe_train; fresh verifier: verification_train/verification_val; final metrics: explicit evaluation split',
        'scope':'affine squared-loss access to declared targets; erasure is not arbitrary nonlinear independence and not a binary edge override'},raw
