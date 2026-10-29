"""Content hit, collateral changes, original risk and dynamic artifact checks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from bridge43 import (ROOT,RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,labels23,baseline_sites,
    capture,load_tree,save_tree,write_json,predict,tree_hash,parameter_hash)
from temporal_probes import paired_bound
from intervention_math43 import apply_eraser,norm_matched,transport,decomposition,finite_event_envelope
from runtime_interventions43 import run_post_edit,after_step_edit,full_snapshot_donor

def native_losses(pred,batch):
    result={}
    for source in ('H1','C'):
        j=SOURCES.index(source);a=np.asarray(pred,float)[:,4*j:4*j+4];y=batch['latents'][:,j]
        result[source]={'MSE':np.mean((a-y)**2,axis=1),
            'derived_classification':np.where(a[:,0]==0,.5,(np.sign(a[:,0])!=np.sign(y[:,0])).astype(float))}
    return result

def probe_report(pred,original,donor,source,time,baseline):
    out={};j=SOURCES.index(source);true=labels23(original)
    for s in SOURCES:
        k=SOURCES.index(s);columns=slice(4*k,4*k+4);before=baseline[:,columns];after=pred[:,columns]
        out[s]={'available_at_site':bool(np.all(original['event_times'][:,k]<=time)),
            'original_label_MSE':float(np.mean((after-true[:,columns])**2)),
            'baseline_MSE':float(np.mean((before-true[:,columns])**2)),
            'prediction_drift_MSE':float(np.mean((after-before)**2))}
    target=donor['latents'][:,j]
    out['focal_donor_label_MSE']=float(np.mean((pred[:,4*j:4*j+4]-target)**2))
    bi={'H1':20,'H4':21,'C':22}[source]
    out['focal_binary_error']=float(np.mean(np.where(pred[:,bi]==0,.5,np.sign(pred[:,bi])!=true[:,bi])))
    out['baseline_focal_variance']=float(np.mean(np.var(true[:,4*j:4*j+4],axis=0)))
    out['baseline_probe_power_warning']=out[source]['baseline_MSE']>.95*out['baseline_focal_variance']
    return out

def trace_by_cell(trace,path):return {k:trace[path+'/'+k] for k in ('post','pre','theta','effective_current','natural_event')}

def run(output,fit_directory=None):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    fit=Path(fit_directory) if fit_directory else ROOT/'evidence/fitted'
    f=json.loads((fit/'freeze_contract.json').read_text());bank=load_tree(fit/'frozen_bank');assert tree_hash(bank)==f['bank_sha256']
    model,cfg,meta=load_model(RUN_DIRECTORY);before=parameter_hash(model)
    assert hashlib.sha256((RUN_DIRECTORY/'best.pt').read_bytes()).hexdigest()==f['checkpoint_sha256']
    a=fixed_batch(cfg.data,2048,'development_test',9060000,0);b=fixed_batch(cfg.data,2048,'development_test',9070000,0)
    av,ap=baseline_sites(model,a);bv,_=baseline_sites(model,b);donors={};neutral={};dv={};nv={};dp={};npred={}
    for source in ('H1','H4','C'):
        donors[source]=hybrid(cfg.data,a,b,source);neutral[source]=hybrid(cfg.data,a,b,source,zero=True)
        dv[source],dp[source]=baseline_sites(model,donors[source]);nv[source],npred[source]=baseline_sites(model,neutral[source])
    results={};scene={};identities={};replay_payload={};native_columns=list(range(4))+list(range(16,20))
    for c in f['contexts']:
        key,site,source,time,layer=(c[k] for k in ('key','site','source','time','layer'));path=f'backbone.cells.{layer}'
        h=av[site];eh=bank['erasers'][key];scale=f['scales'][site];j=SOURCES.index(source)
        base_probes={kind:predict(p,h/scale) for kind,p in bank['baseline_probes'][site].items()}
        local_d=run_post_edit(model,a['x'],path,time,lambda raw,start:dv[source][site][start:start+len(raw)])
        local_z=run_post_edit(model,a['x'],path,time,lambda raw,start:nv[source][site][start:start+len(raw)])
        dz=b['latents'][:,j]-a['latents'][:,j]
        operations={'no_op':h,'zero_v':np.zeros_like(h),'leace':apply_eraser(h,eh['leace']),
            'orth':apply_eraser(h,eh['orth']),'binary_mp':apply_eraser(h,eh['binary_mp']),
            'donor_v':dv[source][site],'oracle_zero_v':nv[source][site],
            'transport_leace_matched':transport(h,dv[source][site],eh['leace']),
            'transport_leace_unmatched':transport(h,bv[site],eh['leace']),
            'transport_orth_unmatched':transport(h,bv[site],eh['orth']),
            'paired_delta_oracle':h+dz@bank['delta_maps'][key]}
        for seed in range(3):
            operations['random_'+str(seed)]=apply_eraser(h,eh['random_'+str(seed)])
            operations['normmatched_'+str(seed)]=norm_matched(h,eh['leace'],seed+4360000)
        # These are diagnostics on an independent validation set, not operation selection.
        rows={}
        for name,edited in operations.items():
            transport_operation=name.startswith('transport') or name in ('donor_v','paired_delta_oracle')
            reference=local_d if transport_operation else local_z
            natural=dp[source] if transport_operation else npred[source]
            reference_labels=donors[source] if transport_operation else neutral[source]
            if name=='donor_v':r=local_d
            elif name=='oracle_zero_v':r=local_z
            else:r=run_post_edit(model,a['x'],path,time,lambda raw,start:edited[start:start+len(raw)])
            pred=r['prediction'];lo=native_losses(pred,a);lh=native_losses(pred,donors[source]);row={}
            row['native_risks']={s:{o:float(v.mean()) for o,v in z.items()} for s,z in lo.items()}
            row['native_hybrid_label_risks']={s:{o:float(v.mean()) for o,v in z.items()} for s,z in lh.items()}
            row['state_displacement_MSE']=float(np.mean((r['site_after']-h)**2))
            row['operation_application_max_abs']=float(np.max(np.abs(r['site_after']-edited)))
            row['original_site_replay_max_abs']=float(np.max(np.abs(r['site_before']-h)))
            row['frozen_validation_probes']={kind:probe_report(predict(p,r['site_after']/scale),a,donors[source],source,time,base_probes[kind]) for kind,p in bank['baseline_probes'][site].items()}
            if name in bank['refit_probes'][key]:
                row['refitted_after_erasure_probes']={kind:probe_report(predict(p,r['site_after']/scale),a,donors[source],source,time,base_probes[kind]) for kind,p in bank['refit_probes'][key][name].items()}
            row['operator_and_scope']=decomposition(pred[:,native_columns],natural[:,native_columns],reference['prediction'][:,native_columns],reference_labels['latents'][:,[0,4]].reshape(len(a['x']),8))
            actual_beta=float(model.core.backbone.cells[0].fixed_beta)
            row['dynamics']={cp:finite_event_envelope(trace_by_cell(r['traces'],cp),trace_by_cell(reference['traces'],cp),time,beta=actual_beta,roundoff_epsilon=np.finfo(np.float32).eps) for cp in ('backbone.cells.0','backbone.cells.1')}
            if name.startswith('normmatched'):
                row['norm_match_max_abs']=float(np.max(np.abs(np.linalg.norm(edited-h,axis=1)-np.linalg.norm(operations['leace']-h,axis=1))))
            rows[name]=row
            for s,z in lo.items():
                for obj,v in z.items():scene[key+'::'+name+'::'+s+'::'+obj]=v
            replay_payload[key+'::'+name]={'site_after_prefix':r['site_after'][:32],'native_prediction_prefix':pred[:32]}
            if name=='no_op':identities[key]={'native_noop_max_abs':float(np.max(np.abs(pred-ap)))}
        results[key]=rows;print('evaluated state context',key,flush=True)
    # Decision-stage control: post storage after G has consumed the final step.
    path='backbone.cells.1';time=15;n=64;x=a['x'][:n]
    after=after_step_edit(model,x,path,time,lambda h,start:np.zeros_like(h))
    beforeG=run_post_edit(model,x,path,time,lambda h,start:np.zeros_like(h),keep_traces=False)['prediction']
    phase={'after_final_step_zero_v_prediction_change_max_abs':float(np.max(np.abs(after-ap[:n]))),
        'before_G_zero_v_prediction_change_max_abs':float(np.max(np.abs(beforeG-ap[:n]))),
        'scope':'Same post-storage value changed on opposite sides of the actual observation/readout operation.'}
    whole={}
    for source in ('H1','H4','C'):
        for time in (12,15):
            z=full_snapshot_donor(model,a['x'][:64],donors[source]['x'][:64],time)
            whole[source+'_t'+str(time)]={'max_abs':z['max_abs'],'scope':z['scope']}
    family=len(f['contexts'])*2*2;contrasts={}
    for c in f['contexts']:
        key=c['key']
        for source in ('H1','C'):
            target=scene[key+'::leace::'+source+'::derived_classification']
            for control in ('random','normmatched'):
                values=np.mean([scene[key+'::'+control+'_'+str(seed)+'::'+source+'::derived_classification'] for seed in range(3)],axis=0)
                contrasts[key+'::'+source+'::'+control]=paired_bound(target,values,family)
    np.savez_compressed(out/'scene_losses.npz',**scene);save_tree(out/'replay_prefix',replay_payload)
    assert parameter_hash(model)==before and tree_hash(bank)==f['bank_sha256']
    report={'module':'L4.3 native state interventions','contexts':len(results),'operations_per_context':len(operations),'native_intervention_conditions':sum(len(v) for v in results.values()),
        'evaluation_pairs':2048,'results':results,'no_op_identities':identities,'phase_control':phase,'whole_snapshot_donor_controls':whole,
        'classification_comparison_family':family,'classification_excess_damage_vs_control':contrasts,
        'saved_loss_arrays':len(scene),'core_parameters_unchanged':True,'checkpoint_sha256':f['checkpoint_sha256'],
        'new_SNN_training':0,'formal_benchmark_completed':0,'audit_test_accessed':False,
        'scope':'One old one-update model. Natural counterfactual, localized donor and learned operation have distinct meanings. MSE point estimates and bounded derived classification comparisons are separate. No general architecture ranking.'}
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--fit-directory');a=p.parse_args();r=run(a.output,a.fit_directory)
    print('Native state conditions',r['native_intervention_conditions'],'loss arrays',r['saved_loss_arrays'])
