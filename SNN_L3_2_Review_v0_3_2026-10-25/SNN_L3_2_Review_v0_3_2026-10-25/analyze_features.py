"""Independent ridge content probes and exact linear risk accounting.

Operates on actual run_train.py exports; this script creates no mock SNN results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from linear_reference import fit_ridge,predict_ridge,regression_metrics,fit_linear_encoding,risk_accounting
from load_data import SOURCES

def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,list):return [serial(x) for x in v]
    return v

def analyze(root,evaluation='development_test',learning_curve=False,probe_n=None):
    if evaluation not in ('development_test','audit_test'):raise ValueError('Independent evaluation split required')
    arrays={s:np.load(root/(s+'.npz')) for s in ('geometry_fit','probe_train','probe_val',evaluation)}
    metadata=json.loads((root/'run.json').read_text());requested_k=metadata['config']['data']['k']
    k=4 if metadata['config']['data'].get('content_mode')=='demand_only' else requested_k
    ns=(probe_n,) if probe_n is not None else ((128,512,2048,6144) if learning_curve else (6144,))
    results=[]
    for obs in ('last_post','joint_post','last_pre','last_output'):
        for n in ns:
            raw=arrays['probe_train'][obs][:n].astype(float)
            if len(raw)!=n:raise ValueError('Requested probe sample count unavailable')
            mean=raw.mean(0);scale=raw.std(0);scale[scale<1e-8]=1.
            x=(raw-mean)/scale
            val=(arrays['probe_val'][obs]-mean)/scale
            test=(arrays[evaluation][obs]-mean)/scale
            encoding=fit_linear_encoding(arrays['geometry_fit'][obs],arrays['geometry_fit']['targets'])
            for source,j in zip(SOURCES,range(5)):
                sl=slice(j*k,(j+1)*k);target=arrays['probe_train']['targets'][:n,sl]
                fits=[]
                for alpha in (1e-4,.01,1.,100.,10000.):
                    fit=fit_ridge(x,target,alpha)
                    loss=float(np.mean((predict_ridge(fit,val)-arrays['probe_val']['targets'][:,sl])**2))
                    fits.append((loss,alpha,fit))
                _,_,fit=min(fits,key=lambda z:(z[0],z[1]))
                metrics=regression_metrics(predict_ridge(fit,test),arrays[evaluation]['targets'][:,sl])
                raw_w=(fit['coef']/scale[:,None]).T
                raw_b=fit['bias']-(mean/scale) @ fit['coef']
                risk=risk_accounting(arrays[evaluation][obs],arrays[evaluation]['targets'],raw_w,raw_b,
                                     encoding,list(range(j*k,(j+1)*k)))
                present=arrays[evaluation]['source_present'][:,j]
                results.append({'observation':obs,'probe_n':n,'source':source,'chosen_alpha':fit['alpha'],
                                'source_present_fraction':float(present.mean()),
                                'metrics':metrics,'linear_risk_accounting':risk,
                                'function_family':'linear with intercept; probe-train-only scaling',
                                'interpretation':'diagnostic recoverability; not proof of original-head use'})
    summaries=[]
    l=metadata['config']['data']['required_history']
    for obs in ('last_post','joint_post','last_pre','last_output'):
        for n in ns:
            rs=[r for r in results if r['observation']==obs and r['probe_n']==n]
            summaries.append({'observation':obs,'probe_n':n,
                'all_source_sum_heldout_r2':sum(r['metrics']['sum_r2'] for r in rs),
                'anchor_mean_heldout_r2':sum(np.sum(r['metrics']['per_coordinate_r2'][:requested_k]) for r in rs if r['source'] in ('H1','C'))/(2*requested_k),
                'requested_sum_heldout_r2':sum(np.sum(r['metrics']['per_coordinate_r2'][:requested_k]) for r in rs if r['source'] in SOURCES[:l]+('C',)),
                'observation_dimension':int(arrays[evaluation][obs].shape[1]),
                'note':'Empirical probes, not the exact optimal population capacity; no clipping at zero.'})
    original=arrays[evaluation]['prediction']
    requested=list(range(l))+[4]
    if metadata['config']['objective']=='regression':
        predicted=original.reshape(-1,5,4)[:,:,:k].reshape(len(original),-1)
        original_metrics={SOURCES[j]:regression_metrics(predicted[:,j*k:j*k+requested_k],arrays[evaluation]['targets'][:,j*k:j*k+requested_k]) for j in requested}
    else:
        truth=arrays[evaluation]['classification_targets']
        original_metrics={SOURCES[j]:{'accuracy':float(np.mean((original[:,j]>=0)==truth[:,j]))} for j in requested}
    return serial({'scope':evaluation,'run':metadata,'results':results,'summaries':summaries,
                   'original_head_requested_targets':original_metrics,
                   'status':'actual exported features only; no capacity theorem fitted',
                   'negative_R2_retained':True,'encoding_fit_split':'geometry_fit',
                   'readout_selection_split':'probe_val','joint_observation_cost':'2w state scalars versus w for last_post'})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run_directory',type=Path)
    ap.add_argument('--evaluation',choices=['development_test','audit_test'],default='development_test')
    ap.add_argument('--learning-curve',action='store_true');ap.add_argument('--probe-n',type=int);args=ap.parse_args()
    if args.probe_n is not None and args.probe_n<2:raise ValueError('At least two probe samples')
    result=analyze(args.run_directory,args.evaluation,args.learning_curve,args.probe_n)
    p=args.run_directory/('content_analysis_'+args.evaluation+'.json')
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(p)
