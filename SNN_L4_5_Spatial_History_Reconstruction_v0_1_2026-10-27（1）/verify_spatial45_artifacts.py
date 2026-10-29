"""Recompute every main loss/interval and replay every saved state-patch cell prefix."""
import argparse,ast,hashlib
from pathlib import Path
import numpy as np
import torch
from bridge45 import ROOT,CONFIGS,VIEWS,model_for,config_key,site_paths,parameter_hash,load_tree,tree_hash,read_json,dump
from spatial_data45 import batch,replace_source,input_hash
from spatial_runtime45 import run,state,masked_replace,additive
from spatial_readers45 import aggregate,decode
from spatial_metrics45 import bounded_interval,regional_losses
from run_spatial_main45 import patch_vectors,contrast

def verify(replay=True):
    torch.set_num_threads(1);py=0;js=0
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text());py+=1
    for p in ROOT.rglob('*.json'):
        import json
        json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)));js+=1
    prov=read_json(ROOT/'references/project_provenance.json')
    for name,h in prov['inherited_files'].items():assert hashlib.sha256((ROOT/'parent_l44'/name).read_bytes()).hexdigest()==h,name
    f=read_json(ROOT/'evidence/fitted/freeze_contract.json');bank=load_tree(ROOT/'evidence/fitted/bank');assert tree_hash(bank)==f['bank_sha256']
    array_count=0;unit_count=0;int_count=0;prefix_units=0;gaps={'payload':0.,'state':0.,'prediction':0.};pools={}
    for pool in ('development','confirmation'):
        root=ROOT/'evidence'/pool;r=read_json(root/'report.json');loss=np.load(root/'scene_losses.npz');array_count+=len(loss.files)
        assert len(loss.files)==r['scene_loss_arrays']==4608;intervals={v['id']:v for v in r['bounded_comparisons']};ratio_entries=0;unique_damage=set()
        for context,row in r['results'].items():
            for method,mr in row['methods'].items():
                for view in VIEWS:
                    for region in ('full','H','C','U_vis'):
                        tag=context+'::'+method+'::'+view+'::'+region;values=[loss[tag+'::'+g] for g in ('00','10','01','11')]
                        recomputed=contrast(*values,floor=f['MSE_damage_floor']);stored=mr['views'][view][region]
                        for k,value in recomputed.items():
                            if value is None:assert stored[k] is None
                            else:assert abs(value-stored[k])<1e-12
                        if recomputed['recovery_ratio'] is not None:
                            ratio_entries+=1;unique_damage.add((row['config'],row['site'],view,region))
                        if region in ('H','C'):
                            a,d,p,z=values
                            for label,v,h in [('rescue',d-z,1.),('patch_only',a-p,1.),('interaction',d-z-a+p,2.)]:
                                got=bounded_interval(v,f['bounded_family_per_pool'],f['family_delta'],h);old=intervals[tag+'::'+label]
                                for k,value in got.items():assert abs(value-old[k])<1e-12
                                int_count+=1
                    unit_count+=1
        pools[pool]={'independent_scene_pairs':r['independent_scene_pairs'],'qualified_ratio_method_view_region_entries':ratio_entries,
            'unique_qualified_damage_config_site_view_region':len(unique_damage),'positive_bounded_lower_bounds':r['positive_bounded_lower_bounds']}
        if not replay:continue
        p=f['pools'][pool];d=f['pools'][pool+'_donor'];a=batch(64,p['first_id']);b=batch(64,d['first_id']);assert input_hash(batch(p['n'],p['first_id'])['x'])==r['input_sha256']
        neutral=replace_source(a,b,'H',neutral=True);wrong=replace_source(a,b,'H');saved=load_tree(root/'replay_prefix')
        for backbone,neuron in CONFIGS:
            key=config_key(backbone,neuron);m=model_for(backbone,neuron);assert parameter_hash(m)==f['parameter_hashes'][key];paths=site_paths(m)
            clean=run(m,a['x'],sites=paths);nr=run(m,neutral['x'],sites=paths);wr=run(m,wrong['x'],sites=paths)
            for path in paths:
                de=masked_replace(path,2,state(nr,path,2),a['masks']['H']);damaged=run(m,a['x'],[de],sites=paths)
                for t in f['patch_times']:
                    vectors,_=patch_vectors(clean,damaged,wr,path,t,a['masks']['H']);context=key+'::'+path+'::p'+str(t)
                    for method,delta in vectors.items():
                        pe=additive(path,t,delta);patch=run(m,a['x'],[pe],sites=paths);repair=run(m,a['x'],[de,pe],sites=paths)
                        for view in VIEWS:
                            savedrow=saved[context+'::'+method+'::'+view];reader=bank[key]['decoders'][view]['spatial3']
                            gaps['payload']=max(gaps['payload'],float(np.max(np.abs(savedrow['delta']-delta[:8]))))
                            gaps['state']=max(gaps['state'],float(np.max(np.abs(savedrow['repaired_state']-state(repair,path,t)[:8]))))
                            for label,result in [('patch_prediction',patch),('repair_prediction',repair)]:
                                prediction=decode(reader,aggregate(result['features'],view))[:8]
                                gaps['prediction']=max(gaps['prediction'],float(np.max(np.abs(prediction-savedrow[label]))))
                            prefix_units+=1
            print('replayed',pool,key,flush=True)
    assert max(gaps.values())<1e-8
    supplements=read_json(ROOT/'evidence/supplements/report.json');sl=np.load(ROOT/'evidence/supplements/scene_losses.npz');assert len(sl.files)==supplements['saved_scene_loss_arrays']==480
    for name,v in supplements['ALIF_joint_owner_factorials'].items():assert v['views']['last']['va']['clean_prediction_max_abs']==0
    quality=read_json(ROOT/'evidence/quality/report.json');ssim=np.load(ROOT/'evidence/quality/scene_SSIM.npz')
    for name,v in quality['SSIM'].items():assert abs(float(ssim[name].mean())-v['mean_sequence_SSIM'])<1e-12
    matched=read_json(ROOT/'evidence/matched_summaries/report.json');ml=np.load(ROOT/'evidence/matched_summaries/scene_losses.npz')
    for config,row in matched['results'].items():
        for view,methods in row.items():
            for method,v in methods.items():assert abs(float(ml[config+'::'+view+'::'+method].mean())-v['clipped_MSE']['full'])<1e-12
    return {'module':'L4.5 artifact verification','python_AST_files':py,'strict_JSON_files':js,'unchanged_parent_files_verified':len(prov['inherited_files']),
        'main_scene_loss_arrays_recomputed':array_count,'main_view_four_group_units_recomputed':unit_count,'bounded_intervals_recomputed':int_count,
        'main_prefix_view_units_replayed':prefix_units,'scenes_saved_per_prefix':8,'original_replay_batch_size':64,'maximum_replay_absolute_gaps':gaps,
        'supplement_scene_loss_arrays_checked':len(sl.files),'SSIM_arrays_checked':len(ssim.files),'matched_summary_arrays_checked':len(ml.files),'pool_summaries':pools,
        'joint_ALIF_last_full_owner_restoration_verified':True,'new_SNN_training_updates':0,'formal_benchmark_completed':0,
        'scope':'All main saved scene risks/intervals recomputed; every main saved prefix replayed. Supplement arrays and exact owner contracts checked, not all supplementary forward passes rerun. No trained-core or real-data conclusion.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--skip-replay',action='store_true');a=p.parse_args();v=verify(not a.skip_replay);dump(a.output,v);print(v)
