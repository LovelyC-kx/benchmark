"""Post-main exploratory comparison: equal learned decoder dimension, separately charged summaries."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from bridge45 import ROOT,CONFIGS,VIEWS,model_for,config_key,site_paths,parameter_hash,dump,save_tree,read_json
from spatial_data45 import batch,input_hash
from spatial_runtime45 import run
from spatial_readers45 import aggregate,ridge_fit,ridge_predict
from spatial_metrics45 import summary

def execute(output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);f=read_json(ROOT/'evidence/fitted/freeze_contract.json')
    tr=batch(f['pools']['train']['n'],f['pools']['train']['first_id']);va=batch(f['pools']['val']['n'],f['pools']['val']['first_id'])
    test=batch(512,45220000);rows={};bank={};losses={};start=time.monotonic()
    for b,n in CONFIGS:
        key=config_key(b,n);m=model_for(b,n);before=parameter_hash(m)
        runs={name:run(m,a['x'],sites=()) for name,a in [('train',tr),('val',va),('test',test)]};rows[key]={};bank[key]={}
        for view in VIEWS:
            features={k:aggregate(v['features'],view).reshape(len(v['features']),-1).astype(float) for k,v in runs.items()}
            mean=features['train'].mean(0);z=features['train']-mean
            _,_,vh=np.linalg.svd(z,full_matrices=False);pca=vh[:8].T
            random=np.random.default_rng(451212).normal(size=pca.shape);random=np.linalg.qr(random)[0]
            # GAP is itself a fixed 2048 -> 8 linear summary.
            gap=np.zeros_like(pca)
            for c in range(8):gap[c*256:(c+1)*256,c]=1/256
            rows[key][view]={};bank[key][view]={}
            for name,proj in [('GAP8',gap),('PCA8',pca),('random8',random)]:
                x={k:(v-mean)@proj for k,v in features.items()};candidates=[];models=[]
                for alpha in (.001,.01,.1):
                    reader=ridge_fit(x['train'],tr['y'].reshape(len(tr['x']),-1),alpha)
                    risk=float(np.mean((ridge_predict(reader,x['val'])-va['y'].reshape(len(va['x']),-1))**2))
                    candidates.append({'alpha':alpha,'validation_raw_MSE':risk});models.append(reader)
                j=int(np.argmin([v['validation_raw_MSE'] for v in candidates]));reader=models[j];pred=ridge_predict(reader,x['test']).reshape(test['y'].shape)
                report=summary(pred,test['y'],test['masks'],True);report.update({'selected_alpha':candidates[j]['alpha'],'validation_candidates':candidates,
                    'decoder_learned_parameters':int(reader['w'].size+reader['y_mean'].size),'summary_matrix_shape':list(proj.shape),
                    'dense_projection_coefficients':int(proj.size),'summary_cost':'GAP has pooling structure; PCA/random require full spatial access and a dense projection. Decoder equality is not total-cost equality.'})
                rows[key][view][name]=report;bank[key][view][name]={'projection':proj,'feature_mean':mean,'reader':reader}
                losses[key+'::'+view+'::'+name]=np.mean((np.clip(pred,0,1)-test['y'])**2,axis=(1,2,3))
        assert parameter_hash(m)==before;print('matched summaries',key,'elapsed',round(time.monotonic()-start,1),flush=True)
    save_tree(out/'bank',bank);np.savez_compressed(out/'scene_losses.npz',**losses)
    result={'module':'L4.5 matched decoder-dimension summaries','added_after_main_results':True,'analysis_type':'Exploratory; no reuse of main evaluation scenes',
        'independent_scene_count':512,'first_scene_id':45220000,'test_input_sha256':input_hash(test['x']),
        'fit_and_select_scene_pools':'Same independent original train/val; PCA fitted on train without labels; ridge alpha only val.',
        'configurations':6,'views':2,'summary_methods':3,'selected_decoders':36,'ridge_candidate_solves':108,
        'learned_decoder_parameters_each':9216,'results':rows,'new_SNN_training_updates':0,
        'scope':'All cores untrained, one initialization. Equal decoder parameters only; summary access/storage/compute differs. No independent trained-model or equal-total-budget design confirmation.'}
    dump(out/'report.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();execute(a.output)
