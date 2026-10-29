"""Paired four-cell recovery, semantic validators and matched finite patches."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge44 import (ROOT,RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,load_tree,tree_hash,write_json,save_tree,
    predict,labels23,parameter_hash)
from recovery_runtime44 import scheduled_run,trace_site,damage_edit,additive_edit
from recovery_math44 import four_losses,output_decomposition,norm_match,random_dose,bounded_interval

def losses(pred,batch):
    p=np.asarray(pred,float);out={}
    for source in ('H1','C'):
        j=SOURCES.index(source);value=p[:,4*j:4*j+4];target=batch['latents'][:,j];mse=np.mean((value-target)**2,axis=1)
        out[source]={'MSE':mse,'capped_MSE4':np.minimum(mse/4.,1.),
            'derived_sign':np.where(value[:,0]==0,.5,(np.sign(value[:,0])!=np.sign(target[:,0])).astype(float))}
    return out

def validation(h,clean,donor,batch,source,probes,scale,time):
    result={};y=labels23(batch);j=SOURCES.index(source)
    for kind,probe in probes.items():
        p=predict(probe,h/scale);p0=predict(probe,clean/scale);row={}
        for k,name in enumerate(SOURCES):
            col=slice(4*k,4*k+4);variance=float(np.mean(np.var(y[:,col],axis=0)))
            base=float(np.mean((p0[:,col]-y[:,col])**2))
            row[name]={'MSE':float(np.mean((p[:,col]-y[:,col])**2)),'clean_probe_MSE':base,
                'prediction_drift_MSE':float(np.mean((p[:,col]-p0[:,col])**2)),
                'available_at_patch':bool(np.all(batch['event_times'][:,k]<=time)),
                'baseline_probe_power_warning':bool(base>.95*variance)}
        row['focal_donor_MSE']=float(np.mean((p[:,4*j:4*j+4]-donor['latents'][:,j])**2));result[kind]=row
    return result

def build_patches(c,clean,damaged,b_run,f_run,alt_run,eraser):
    layer,time=c['layer'],c['patch_time'];h=trace_site(clean,layer,time);d=trace_site(damaged,layer,time)
    q=np.eye(h.shape[1])-eraser['matrix'];target=(h-d)@q
    opposite_time=time+1 if time<15 else time-1
    raw={'wrong_F_delta':(trace_site(f_run,layer,time)-d)@q,
        'right_F_other_context_delta':(trace_site(alt_run,layer,time)-d)@q,
        'unmatched_donor_delta':(trace_site(b_run,layer,time)-d)@q,
        'wrong_time_delta':(trace_site(clean,layer,opposite_time)-trace_site(damaged,layer,opposite_time))@q}
    patches={'full_self_delta':h-d,'target_self_delta':target,'target_reverse_delta':-target};budgets={}
    for name,value in raw.items():patches[name],budgets[name]=norm_match(value,target)
    for seed in range(3):patches['norm_random_'+str(seed)]=random_dose(target,440700+seed)
    for name in ('target_reverse_delta','norm_random_0','norm_random_1','norm_random_2'):
        budgets[name]={'unmatchable_nonzero_reference_scenes':0,'maximum_norm_gap_on_matchable_scenes':float(np.max(np.abs(np.linalg.norm(patches[name],axis=1)-np.linalg.norm(target,axis=1))))}
    budgets['wrong_time_source_step']=opposite_time;budgets['wrong_time_is_future_oracle']=opposite_time>time
    return patches,budgets

def run(output,pool='development',fit_directory=None):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    fit=Path(fit_directory) if fit_directory else ROOT/'evidence/fitted';f=json.loads((fit/'freeze_contract.json').read_text());bank=load_tree(fit/'frozen_bank')
    assert tree_hash(bank)==f['bank_sha256'];settings=f['evaluation_pools'][pool]
    model,cfg,_=load_model(RUN_DIRECTORY);before=parameter_hash(model);n=settings['n']
    a=fixed_batch(cfg.data,n,'development_test',settings['recipient'],0);b=fixed_batch(cfg.data,n,'development_test',settings['donor'],0)
    clean=scheduled_run(model,a['x']);b_run=scheduled_run(model,b['x']);clean_losses=losses(clean['prediction'],a)
    donors={s:hybrid(cfg.data,a,b,s) for s in ('H1','H4','C')}
    fruns={s:scheduled_run(model,d['x']) for s,d in donors.items()}
    altruns={s:scheduled_run(model,hybrid(cfg.data,b,a,s)['x']) for s in donors}
    damaged={};damage_losses={};damage_reports={};arrays={};prefix={};bounded={};rows={};columns=list(range(4))+list(range(16,20))
    family=f['bounded_comparison_family_per_pool']
    for c in f['damage_contexts']:
        r=scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']])]);damaged[c['key']]=r;dl=losses(r['prediction'],a);damage_losses[c['key']]=dl
        damage_reports[c['key']]={s:{m:float(v.mean()) for m,v in z.items()} for s,z in dl.items()}
        for source,z in dl.items():
            for metric,v in z.items():
                arrays[c['key']+'::damage::'+source+'::'+metric]=v
                if metric!='MSE':bounded[c['key']+'::damage::'+source+'::'+metric]=bounded_interval(v-clean_losses[source][metric],family)
    for source,z in clean_losses.items():
        for metric,v in z.items():arrays['clean::'+source+'::'+metric]=v
    for c in f['recovery_contexts']:
        key,source,site=c['key44'],c['source'],c['patch_site'];d=damaged[c['key']];dl=damage_losses[c['key']];h=trace_site(clean,c['layer'],c['patch_time'])
        damaged_validation=validation(trace_site(d,c['layer'],c['patch_time']),h,donors[source],a,source,bank['validation_probes'][site],f['scales'][site],c['patch_time'])
        patches,budgets=build_patches(c,clean,d,b_run,fruns[source],altruns[source],bank['patch_erasers'][c['patch_key']]);cases={}
        for name in f['methods']:
            delta=patches[name];patch=additive_edit(c['layer'],c['patch_time'],delta)
            p=scheduled_run(model,a['x'],[patch]);r=scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']]),patch])
            pl,rl=losses(p['prediction'],a),losses(r['prediction'],a);risks={}
            for q in ('H1','C'):
                risks[q]={}
                for metric in ('MSE','capped_MSE4','derived_sign'):
                    values=(clean_losses[q][metric],dl[q][metric],pl[q][metric],rl[q][metric]);risks[q][metric]=four_losses(*values,damage_floor=.01 if metric=='MSE' else None)
                    for group,v in [('patch_only',values[2]),('repair',values[3])]:arrays[key+'::'+name+'::'+group+'::'+q+'::'+metric]=v
                    if metric!='MSE':
                        for effect,v,half in [('rescue',values[1]-values[3],1.),('patch_only_gain',values[0]-values[2],1.),('risk_interaction',values[1]-values[3]-values[0]+values[2],2.)]:
                            bounded[key+'::'+name+'::'+q+'::'+metric+'::'+effect]=bounded_interval(v,family,half_width=half)
            y=a['latents'][:,[0,4]].reshape(n,8)
            output=output_decomposition(clean['prediction'][:,columns],d['prediction'][:,columns],p['prediction'][:,columns],r['prediction'][:,columns],y)
            ph=p['edits']['patch']['after'];rh=r['edits']['patch']['after'];rawh=trace_site(d,c['layer'],c['patch_time'])
            rounding=float(np.max(np.abs((r['edits']['patch']['after']-r['edits']['patch']['before'])-(p['edits']['patch']['after']-p['edits']['patch']['before']))))
            row={'four_group_risks':risks,'output_analysis':output,
                'patch_state_validation':validation(ph,h,donors[source],a,source,bank['validation_probes'][site],f['scales'][site],c['patch_time']),
                'repaired_state_validation':validation(rh,h,donors[source],a,source,bank['validation_probes'][site],f['scales'][site],c['patch_time']),
                'focal_state_clean_fidelity_MSE_before':float(np.mean((rawh-h)**2)),
                'focal_state_clean_fidelity_MSE_after':float(np.mean((rh-h)**2)),
                'patch_payload_mean_squared_norm_per_coordinate':float(np.mean(delta**2)),
                'same_external_patch_rounding_gap_max_abs':rounding,
                'native_donor_label_risks':{s:{m:float(v.mean()) for m,v in z.items()} for s,z in losses(r['prediction'],donors[source]).items()}}
            event={}
            for layer in (0,1):
                path=f'backbone.cells.{layer}/emit_event';sl=slice(c['patch_time']+1,None)
                aevent=clean['traces'][path][:,sl];revent=r['traces'][path][:,sl]
                omega=r['traces'][path][:,sl].astype(float)-d['traces'][path][:,sl]-p['traces'][path][:,sl]+clean['traces'][path][:,sl]
                event[str(layer)]={'future_emitted_event_clean_mismatch_fraction':float(np.mean(aevent!=revent)) if aevent.size else 0.,
                    'future_event_four_group_nonadditivity_fraction':float(np.mean(omega!=0)) if omega.size else 0.,'future_steps':max(0,15-c['patch_time'])}
            row['event_diagnostics']=event
            if c['patch_time']==c['time'] and name=='full_self_delta':row['immediate_algebraic_restore_prediction_max_abs']=float(np.max(np.abs(r['prediction']-clean['prediction'])))
            if c['layer']==1 and c['patch_time']==15 and name=='full_self_delta':row['terminal_readout_overwrite_prediction_max_abs']=float(np.max(np.abs(r['prediction']-clean['prediction'])))
            cases[name]=row
            prefix[key+'::'+name]={'delta_prefix':delta[:16],'patch_only_prediction_prefix':p['prediction'][:16],'repair_prediction_prefix':r['prediction'][:16],
                'repair_site_prefix':rh[:16]}
        rows[key]={'context':c,'matched_budgets':budgets,'damaged_state_validation':damaged_validation,'methods':cases};print('evaluated four-group recovery',pool,key,flush=True)
    assert len(bounded)==family and parameter_hash(model)==before and tree_hash(bank)==f['bank_sha256']
    np.savez_compressed(out/'scene_losses.npz',**arrays);save_tree(out/'replay_prefix',prefix)
    report={'module':'L4.4 four-group state recovery','pool':pool,'independent_scene_pairs':n,'damage_contexts':14,'recovery_contexts':len(rows),
        'methods_per_context':len(f['methods']),'four_group_cells':sum(len(v['methods']) for v in rows.values()),
        'distinct_scheduled_run_conditions':1+14+2*len(rows)*len(f['methods']),
        'clean_risks':{s:{m:float(v.mean()) for m,v in z.items()} for s,z in clean_losses.items()},'damage_risks':damage_reports,
        'results':rows,'bounded_comparisons':bounded,'bounded_comparison_family':family,'saved_scene_loss_arrays':len(arrays),
        'core_parameters_unchanged':True,'new_SNN_training':0,'formal_benchmark_completed':0,'audit_test_accessed':False,
        'scope':'One old one-update core. Four groups share input, label, weights and a fixed external additive patch. All methods frozen before both scene pools; confirmation repeats scenes, not training. Full/same-step and final-readout restoration are explicitly separate implementation ceilings.'}
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--pool',choices=('development','confirmation'),default='development');p.add_argument('--fit-directory');a=p.parse_args()
    r=run(a.output,a.pool,a.fit_directory);print('Four-group cells',r['four_group_cells'],'loss arrays',r['saved_scene_loss_arrays'])
