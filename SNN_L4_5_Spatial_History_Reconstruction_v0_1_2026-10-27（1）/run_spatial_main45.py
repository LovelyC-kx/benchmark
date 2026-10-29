"""Sequence-level pixel evidence and paired four-group spatial state interventions."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from bridge45 import ROOT,CONFIGS,VIEWS,model_for,config_key,site_paths,parameter_hash,load_tree,tree_hash,save_tree,dump,read_json
from spatial_data45 import batch,replace_source,latest_visible,region_masks,labels,input_hash
from spatial_readers45 import aggregate,decode,ridge_predict
from spatial_runtime45 import run,state,masked_replace,additive
from spatial_metrics45 import summary,regional_losses,frequency_losses,bounded_interval

def norm_match(u,target):
    u=np.asarray(u,float);target=np.asarray(target,float);shape=u.shape
    a=u.reshape(len(u),-1);r=np.linalg.norm(target.reshape(len(target),-1),axis=1);s=np.linalg.norm(a,axis=1)
    valid=s>1e-12;value=np.zeros_like(a);value[valid]=a[valid]*(r[valid]/s[valid])[:,None]
    return value.reshape(shape),int(np.sum((~valid)&(r>1e-12)))

def patch_vectors(c,d,wrong,path,t,mask):
    base=state(c,path,t)-state(d,path,t);region=base*mask[None,None]
    w=state(wrong,path,t)-state(d,path,t);wt=state(c,path,2)-state(d,path,2)
    w,miss1=norm_match(w*mask[None,None],region);wt,miss2=norm_match(wt*mask[None,None],region)
    rand=np.random.default_rng(450808).normal(size=base.shape)*mask[None,None];rand,_=norm_match(rand,region)
    return {'owner_self':base,'region_self':region,'region_reverse':-region,'wrong_H':w,'wrong_time':wt,'random_region':rand},{'wrong_H':miss1,'wrong_time':miss2}

def probe_risks(bank,h,a):
    p=ridge_predict(bank,h.reshape(len(h),-1));y=labels(a)
    return {name:float(np.mean((p[:,j:j+4]-y[:,j:j+4])**2)) for name,j in [('C_factor',0),('H_factor',4),('changed_C_factor',8),('U_vis_factor',12),('old_H_factor',16),('U_src_factor',20)]}

def contrast(a,d,p,r,floor=.0001):
    values=[np.asarray(v,float) for v in (a,d,p,r)]
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() or np.any(v<0) for v in values) or len({len(v) for v in values})!=1:
        raise ValueError('Paired finite nonnegative scene losses')
    if not np.isfinite(floor) or floor<=0:raise ValueError('Positive predefined damage floor')
    A,D,P,R=[float(v.mean()) for v in (a,d,p,r)];damage=D-A;rescue=D-R;benefit=A-P
    return {'R00':A,'R10':D,'R01':P,'R11':R,'damage':damage,'rescue':rescue,'patch_only_benefit':benefit,
        'risk_interaction':rescue-benefit,'recovery_ratio':rescue/damage if damage>floor else None,'damage_floor':floor}

def evaluate(output,pool,fit_directory=None):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);start=time.monotonic();fit=Path(fit_directory or ROOT/'evidence/fitted')
    f=read_json(fit/'freeze_contract.json');bank=load_tree(fit/'bank');assert tree_hash(bank)==f['bank_sha256']
    settings=f['pools'][pool];don=f['pools'][pool+'_donor'];a=batch(settings['n'],settings['first_id']);b=batch(don['n'],don['first_id'])
    neutral=replace_source(a,b,'H',neutral=True);wrong=replace_source(a,b,'H');mask=a['masks']['H'];arrays={};prefix={};clean_reports={};rows={};intervals=[];cells=0
    baselines={'current_only':latest_visible(a['x'][:,-1:]),'latest_valid_history':latest_visible(a['x'])}
    baseline_reports={k:summary(v,a['y'],a['masks'],True) for k,v in baselines.items()}
    for backbone,neuron in CONFIGS:
        key=config_key(backbone,neuron);m=model_for(backbone,neuron);before=parameter_hash(m);assert before==f['parameter_hashes'][key];paths=site_paths(m)
        clean=run(m,a['x'],sites=paths);nr=run(m,neutral['x'],sites=paths);wr=run(m,wrong['x'],sites=paths);cfgbank=bank[key];cr={}
        for view in VIEWS:
            features=aggregate(clean['features'],view);cr[view]={}
            for kind,reader in cfgbank['decoders'][view].items():
                pred=decode(reader,features);report=summary(pred,a['y'],a['masks'],True)
                report['reader_parameters']=reader['parameter_count'];report['feature_access_scalars']=reader['persistent_feature_scalars_per_scene'];cr[view][kind]=report
            cp=ridge_predict(cfgbank['class_probes'][view],features.mean((2,3)))
            cr[view]['auxiliary_factor_sign_accuracy']={source:float(np.mean(np.sign(cp[:,j])==np.sign(a['latents'][:,j,0]))) for j,source in enumerate(('C','H','changed_C','U_vis'))}
        cr['native_random_head']=summary(clean['prediction'],a['y'],a['masks']);clean_reports[key]=cr
        for path in paths:
            de=masked_replace(path,2,state(nr,path,2),mask);damaged=run(m,a['x'],[de],sites=paths)
            for t in f['patch_times']:
                vectors,miss=patch_vectors(clean,damaged,wr,path,t,mask);context=key+'::'+path+'::p'+str(t)
                row={'config':key,'site':path,'damage_time':2,'patch_time':t,'unmatchable_directions':miss,'methods':{},
                    'clean_probe':probe_risks(cfgbank['probes'][path][str(t)],state(clean,path,t),a),
                    'damaged_probe':probe_risks(cfgbank['probes'][path][str(t)],state(damaged,path,t),a)}
                for name,delta in vectors.items():
                    pe=additive(path,t,delta);patch=run(m,a['x'],[pe],sites=paths);repair=run(m,a['x'],[de,pe],sites=paths)
                    cells+=1;result={'views':{},'repaired_probe':probe_risks(cfgbank['probes'][path][str(t)],state(repair,path,t),a),
                        'patch_only_probe':probe_risks(cfgbank['probes'][path][str(t)],state(patch,path,t),a),
                        'delta_mean_L2':float(np.mean(np.linalg.norm(delta.reshape(len(delta),-1),axis=1))),
                        'terminal_current_readout_overwrite':bool(path==paths[-1] and t==5 and name=='owner_self')}
                    for view in VIEWS:
                        reader=cfgbank['decoders'][view]['spatial3'];preds=[decode(reader,aggregate(v['features'],view)) for v in (clean,damaged,patch,repair)]
                        losses=[regional_losses(np.clip(v,0,1),a['y'],a['masks']) for v in preds];viewrow={}
                        for region in ('full','H','C','U_vis'):
                            z=[v[region] for v in losses];stat=contrast(*z,floor=f['MSE_damage_floor']);viewrow[region]=stat
                            tag=context+'::'+name+'::'+view+'::'+region
                            for k,value in zip(('00','10','01','11'),z):arrays[tag+'::'+k]=value
                            if region in ('H','C'):
                                for label,values,h in [('rescue',z[1]-z[3],1.),('patch_only',z[0]-z[2],1.),('interaction',z[1]-z[3]-z[0]+z[2],2.)]:
                                    intervals.append({'id':tag+'::'+label,**bounded_interval(values,f['bounded_family_per_pool'],f['family_delta'],h)})
                        omega=(preds[3]-preds[1])-(preds[2]-preds[0]);viewrow['output_interaction_MSE']=float(np.mean(omega**2));result['views'][view]=viewrow
                        tag=context+'::'+name+'::'+view;prefix[tag]={'delta':delta[:8],'patch_prediction':preds[2][:8],'repair_prediction':preds[3][:8],
                            'repaired_state':state(repair,path,t)[:8]}
                    row['methods'][name]=result
                rows[context]=row
        assert parameter_hash(m)==before;print(pool,key,'four_group_cells',cells,'elapsed',round(time.monotonic()-start,1),flush=True)
    assert cells==144 and len(intervals)==f['bounded_family_per_pool']
    np.savez_compressed(out/'scene_losses.npz',**arrays);save_tree(out/'replay_prefix',prefix)
    report={'module':'L4.5 spatial reconstruction development','pool':pool,'independent_scene_pairs':len(a['x']),'input_sha256':input_hash(a['x']),
        'core_configurations':6,'core_training_updates':0,'selected_decoder_evaluations':36,'clean_reports':clean_reports,'baseline_reports':baseline_reports,
        'state_four_group_cells':cells,'view_four_group_units':cells*2,'results':rows,'scene_loss_arrays':len(arrays),'bounded_comparisons':intervals,
        'positive_bounded_lower_bounds':sum(v['lower']>0 for v in intervals),'elapsed_seconds':time.monotonic()-start,
        'scope':'Six untrained spatial cores, fixed initialization; independent fitted decoders/probes. Confirmation replicates scenes, not training. Clipped pixel risks are task scores; raw decoder risks retained for clean comparisons. No real data, native end-to-end training or hardware energy results.'}
    dump(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--pool',choices=('development','confirmation'),default='development');p.add_argument('--fit-directory');a=p.parse_args()
    r=evaluate(a.output,a.pool,a.fit_directory);print('Four-group cells',r['state_four_group_cells'],'scene arrays',r['scene_loss_arrays'])
