"""Source/time maps, stale history, alignment, spatial geometry, frequency and aux owners."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from scipy.fft import dctn,idctn
from scipy.ndimage import binary_erosion
from bridge45 import ROOT,CONFIGS,VIEWS,model_for,config_key,site_paths,parameter_hash,load_tree,tree_hash,save_tree,dump,read_json
from spatial_data45 import batch,replace_source,latest_visible,align_observations,render,input_hash
from spatial_runtime45 import run,state,masked_replace,additive
from spatial_readers45 import aggregate,decode
from spatial_metrics45 import summary,regional_losses,frequency_losses

def lds(features):
    f=np.asarray(features,float)[:,:,::2,::2];n,c,h,w=f.shape;t=f.reshape(n,c,-1).transpose(0,2,1);norm=np.linalg.norm(t,axis=-1)
    sim=np.einsum('nid,njd->nij',t,t)/np.maximum(norm[:,:,None]*norm[:,None,:],1e-24)
    coords=np.array(list(np.ndindex(h,w)));dist=np.abs(coords[:,None]-coords[None]).sum(-1)
    near=(dist>0)&(dist<h/2);far=dist>=h/2;valid=(norm[:,:,None]>1e-12)&(norm[:,None,:]>1e-12)
    values=[]
    for s,v in zip(sim,valid):
        a=v&near;b=v&far;values.append(float(s[a].mean()-s[b].mean()) if a.any() and b.any() else None)
    return {'mean_LDS':float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None,
        'valid_scenes':sum(v is not None for v in values),'token_grid':[h,w],'definition':'Cosine local-distance contrast; valid norm pairs; fixed stride2 grid, diagonal excluded.'}

def run_all(output,n=128):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);f=read_json(ROOT/'evidence/fitted/freeze_contract.json');bank=load_tree(ROOT/'evidence/fitted/bank');assert tree_hash(bank)==f['bank_sha256']
    p=f['pools']['confirmation'];d=f['pools']['confirmation_donor'];a=batch(n,p['first_id']);b=batch(n,d['first_id'])
    motion=batch(n,p['first_id'],motion=True);aligned=align_observations(motion);assert np.array_equal(aligned,a['x'])
    ood=batch(256,45200000,family='texture_ood');rows={};arrays={};prefix={};joint={};frequency={};start=time.monotonic()
    for backbone,neuron in CONFIGS:
        key=config_key(backbone,neuron);m=model_for(backbone,neuron);before=parameter_hash(m);paths=site_paths(m)
        clean=run(m,a['x'],sites=paths);variants={}
        for source in ('H','U_src','C'):variants['source_'+source]=run(m,replace_source(a,b,source)['x'],sites=())
        for t in (0,2,5):
            x=a['x'].copy();x[:,t]=0;variants['drop_frame_'+str(t)]=run(m,x,sites=())
        variants['motion_raw']=run(m,motion['x'],sites=());variants['motion_oracle_aligned']=run(m,aligned,sites=());variants['texture_ood']=run(m,ood['x'],sites=())
        row={'views':{}}
        for view in VIEWS:
            reader=bank[key]['decoders'][view]['spatial3'];baseline=decode(reader,aggregate(clean['features'],view));r={}
            for name,v in variants.items():
                data=ood if name=='texture_ood' else a;pred=decode(reader,aggregate(v['features'],view));entry=summary(pred,data['y'],data['masks'],True)
                if name!='texture_ood':
                    response=regional_losses(pred,baseline,a['masks']);entry['output_response_MSE']={k:float(z.mean()) for k,z in response.items() if z is not None}
                    if name in ('source_H','source_C'):
                        z=a['latents'].copy();j=1 if name=='source_H' else 0;z[:,j]=b['latents'][:,j]
                        entry['recomputed_counterfactual_target']=summary(pred,render(z),a['masks'])
                        entry['original_label_score_is_dependence_or_corruption_not_same_label_supervision']=True
                    if name=='source_U_src':entry['same_current_target_and_last_frame_verified']=bool(np.array_equal(a['x'][:,-1],replace_source(a,b,'U_src')['x'][:,-1]))
                    for region,z in regional_losses(np.clip(pred,0,1),data['y'],data['masks']).items():
                        if z is not None:arrays[key+'::'+view+'::'+name+'::'+region]=z
                r[name]=entry
                prefix[key+'::'+view+'::'+name]=pred[:8]
            feat=aggregate(clean['features'],view);perm=np.random.default_rng(450999).permutation(256)
            shuffled=feat.reshape(n,8,-1)[:,:,perm].reshape(feat.shape);inverse=shuffled.reshape(n,8,-1)[:,:,np.argsort(perm)].reshape(feat.shape)
            assert np.array_equal(inverse,feat)
            r['geometry']={'original':lds(feat),'shuffled':lds(shuffled),'inverse_permutation_feature_max_abs':float(np.max(np.abs(inverse-feat))),
                'shuffled_frozen_decoder':summary(decode(reader,shuffled),a['y'],a['masks']),
                'interpretation':'Permutation preserves full information but changes coordinates and the fixed spatial decoder; LDS is descriptive, not a semantic or causal score.'}
            q=regional_losses(np.clip(baseline,0,1),a['y'],a['masks']);area=.5*q['C']+.25*q['H']+.25*q['U_vis']
            boundary=a['masks']['H']&~binary_erosion(a['masks']['H']);interior=binary_erosion(a['masks']['H'])
            r['regional_accounting']={'full_partition_max_abs':float(np.max(np.abs(area-q['full']))),
                'boundary_interior':summary(baseline,a['y'],{'H_boundary':boundary,'H_interior':interior})}
            assert r['regional_accounting']['full_partition_max_abs']<1e-12
            row['views'][view]=r
        # Global spatial DCT of a region-limited state patch: low/high components need not remain local.
        path=paths[-1];neutral=run(m,replace_source(a,b,'H',neutral=True)['x'],sites=paths)
        de=masked_replace(path,2,state(neutral,path,2),a['masks']['H']);damaged=run(m,a['x'],[de],sites=paths)
        delta=(state(clean,path,3)-state(damaged,path,3))*a['masks']['H'][None,None]
        dct=dctn(delta,axes=(-2,-1),norm='ortho');low=np.zeros((16,16),bool);low[:4,:4]=1
        dl=idctn(np.where(low,dct,0),axes=(-2,-1),norm='ortho');dh=delta-dl
        runs={'none':damaged,'low':run(m,a['x'],[de,additive(path,3,dl)],sites=()),
            'high':run(m,a['x'],[de,additive(path,3,dh)],sites=()),'both':run(m,a['x'],[de,additive(path,3,delta)],sites=())}
        frequency[key]={'component_sum_max_abs':float(np.max(np.abs(dl+dh-delta))),
            'low_outside_H_energy_fraction':float(np.sum(dl[:,:,~a['masks']['H']]**2)/max(np.sum(dl**2),1e-24)),
            'mask_note':'Low DCT component spreads outside the original H footprint; named frequency intervention, not a selective H-only edit.',
            'views':{}}
        for view in VIEWS:
            reader=bank[key]['decoders'][view]['spatial3'];preds={k:decode(reader,aggregate(v['features'],view)) for k,v in runs.items()}
            gamma=preds['both']-preds['low']-preds['high']+preds['none']
            frequency[key]['views'][view]={'outputs':{k:summary(v,a['y'],a['masks'],True) for k,v in preds.items()},'Gamma_MSE':float(np.mean(gamma**2))}
        if neuron=='alif_ref':
            damages=[masked_replace(path,2,state(neutral,path,2,k),a['masks']['H'],label='damage_'+k,owner=k) for k in ('v','a')]
            jrow={'views':{},'subset_states':{}}
            for subset in ('none','v','a','va'):
                edits=list(damages)
                for owner in ('v','a'):
                    if owner in subset:
                        ref=state(clean,path,3,owner);edits.append(masked_replace(path,3,ref,np.ones((16,16),bool),label='restore_'+owner,owner=owner))
                rr=run(m,a['x'],edits,sites=paths);jrow['subset_states'][subset]={}
                for view in VIEWS:
                    pred=decode(bank[key]['decoders'][view]['spatial3'],aggregate(rr['features'],view));base=decode(bank[key]['decoders'][view]['spatial3'],aggregate(clean['features'],view))
                    jrow['views'].setdefault(view,{})[subset]={'risk':summary(pred,a['y'],a['masks']),
                        'clean_prediction_max_abs':float(np.max(np.abs(pred-base)))}
            assert jrow['views']['last']['va']['clean_prediction_max_abs']==0
            joint[key]=jrow
        assert parameter_hash(m)==before;rows[key]=row;print('supplement',key,'elapsed',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(out/'scene_losses.npz',**arrays);save_tree(out/'prediction_prefix',prefix)
    report={'module':'L4.5 source, geometry and transfer supplements','confirmation_subset_scenes':n,'independent_texture_OOD_scenes':len(ood['x']),
        'texture_OOD_first_id':45200000,'texture_OOD_input_sha256':input_hash(ood['x']),'configs':6,'source_and_frame_variants':6,
        'results':rows,'frequency_patch_factorials':frequency,'ALIF_joint_owner_factorials':joint,'saved_scene_loss_arrays':len(arrays),
        'motion_inverse_input_max_abs':float(np.max(np.abs(aligned-a['x']))),
        'scope':'Shared confirmation subset, except independent changed-texture OOD scenes. Alignment uses true synthetic shifts and is an explicit oracle. Geometry is descriptive; no method selection, training-seed confirmation, natural satellite/event data or end-to-end reconstruction training.'}
    dump(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=run_all(a.output);print('Supplement loss arrays',r['saved_scene_loss_arrays'])
