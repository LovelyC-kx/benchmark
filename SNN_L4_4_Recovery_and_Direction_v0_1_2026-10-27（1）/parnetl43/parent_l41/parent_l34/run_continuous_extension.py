"""Original-head annotations plus fresh continuous-target finite-p checks.

The completed classification protocol is not changed. Native classifications
are evaluated on its original scenes. The continuous extension fits/fixes all
maps and scalar gains before a different scene pool (default 5,700,000).
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from bridge import load_model,collect,flatten_contents
from parent_sampling import fixed_batch
from factor_sampling import factor_schema
from continuous_risk import continuous_moments,exact_continuous_risk
from task_moments import classification_error,labels
from run_generalization import serial


def continuous_loss(weights,x,y,gain=1.):
    scene=np.zeros(len(x));per_reader=[]
    for w in weights:
        error=np.mean((gain*(x@w)-y)**2,axis=1);scene+=error/len(weights);per_reader.append(float(error.mean()))
    return {'MSE_per_target':float(scene.mean()),'per_scene_MSE':scene,'per_frozen_reader_MSE':per_reader,
        'aggregation':'Mean individual-reader risk; no deployment ensemble and no bounded classification-loss confidence interval.'}


def run(args):
    import torch
    torch.set_num_threads(1);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);source=Path(args.classification_analysis);model,run,meta=load_model(args.run_directory)
    checkpoint=Path(args.run_directory)/'best.pt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    paths=sorted(source.glob('*/report.json'));results=[];total_fits=0
    if not paths:raise ValueError('Completed paired classification analysis required')
    for schedule in sorted({json.loads(p.read_text())['schedule'] for p in paths}):
        configs=[(p,json.loads(p.read_text())) for p in paths if json.loads(p.read_text())['schedule']==schedule]
        batches={name:fixed_batch(run.data,n,split,first,schedule) for name,n,split,first in [
            ('moment_a',1024,'probe_train',5100000),('moment_b',1024,'probe_train',5200000),('train',512,'probe_train',5400000)]}
        f={k:collect(model,b['x']) for k,b in batches.items()};z={k:flatten_contents(b,run.data) for k,b in batches.items()}
        prepared={}
        for path,report in configs:
            folder=out/path.parent.name;folder.mkdir();a=np.load(path.parent/'frozen_readers_and_scene_losses.npz')
            schema=factor_schema(run.data,report['focal'],batches['train']['source_present']);groups={};freeze={}
            for group,roles in [('F',('focal',)),('G',('cotarget',)),('FG',('focal','cotarget'))]:
                g=report['target_groups'][group];columns=[i for i,r in enumerate(schema['coordinate_roles']) if r['role'] in roles]
                origin=np.array(g['observation_origin']);scale=g['shared_geometry_fit_scalar_scale'];obs=report['observation']
                x={k:(v[obs].astype(float)-origin)/scale for k,v in f.items()};target={k:v[:,columns] for k,v in z.items()}
                mx=np.concatenate((x['moment_a'],x['moment_b']));my=np.concatenate((target['moment_a'],target['moment_b']));mom=continuous_moments(mx,my)
                sizes=sorted(map(int,g['reader_results']['identity']));indices=a[group+'__train_indices'];fits={};pred={}
                for p in sizes:
                    h=[];r=[]
                    for idx in indices:
                        tx,ty=x['train'][idx[:p]],target['train'][idx[:p]];c=tx.T@ty/p
                        h.append(c);r.append(np.linalg.solve(tx.T@tx/p+1e-3*np.eye(tx.shape[1]),c))
                    fits[p]={'hebbian':np.stack(h),'ridge':np.stack(r)}
                    pred[p]=exact_continuous_risk(mom['M'],mom['C'],mom['T'],mom['target_squared_norm'],mom['target_dimension'],p)
                    total_fits+=2*len(indices)
                digest=hashlib.sha256()
                for p in sizes:
                    for v in fits[p].values():digest.update(v.tobytes())
                freeze[group]={'weight_sha256':digest.hexdigest(),'label_counts_per_reader':sizes,
                    'moment_selected_scalar_gain':{p:pred[p]['oracle_common_scalar_gain'] for p in sizes},
                    'extra_labelled_scenes_for_scalar_gain':len(mx)}
                groups[group]={'origin':origin,'scale':scale,'columns':columns,'fits':fits,'forecast':pred,
                    'theta':a[group+'__theta'],'observation':obs,'moment_pool_size':len(mx)}
            (folder/'continuous_selection_freeze.json').write_text(json.dumps(serial(freeze),sort_keys=True,allow_nan=False))
            prepared[path.parent.name]=(report,groups,folder)
        # Both test pools are generated only after continuous fits for all
        # priors/carriers in this schedule have been frozen.
        native_first=configs[0][1]['evaluation_scene_first_id'];n=configs[0][1]['evaluation_scenes']
        nb=fixed_batch(run.data,n,'development_test',native_first,schedule);nf=collect(model,nb['x']);nz=flatten_contents(nb,run.data)
        cb=fixed_batch(run.data,args.evaluation_scenes,'development_test',args.evaluation_first_id,schedule);cf=collect(model,cb['x']);cz=flatten_contents(cb,run.data)
        for name,(report,groups,folder) in prepared.items():
            rows={};arrays={}
            for group,g in groups.items():
                columns=g['columns'];theta=g['theta'];native_scores=nf['prediction'][:,columns]@theta
                native=classification_error(native_scores,labels(nz[:,columns],theta))
                native['permission']='Original regression head reads last_post. Different supervision budget; for a spike-only probe this includes additional carrier access.'
                x=(cf[g['observation']].astype(float)-g['origin'])/g['scale'];target=cz[:,columns];curves={}
                for p,fits in g['fits'].items():
                    ordinary=continuous_loss(fits['hebbian'],x,target);ridge=continuous_loss(fits['ridge'],x,target)
                    gain=g['forecast'][p]['oracle_common_scalar_gain'];calibrated=continuous_loss(fits['hebbian'],x,target,gain)
                    arrays[f'{group}__p{p}__raw_scene_mse']=ordinary.pop('per_scene_MSE')
                    arrays[f'{group}__p{p}__gain_scene_mse']=calibrated.pop('per_scene_MSE')
                    arrays[f'{group}__p{p}__ridge_scene_mse']=ridge.pop('per_scene_MSE')
                    arrays[f'{group}__p{p}__raw_weights']=fits['hebbian'];arrays[f'{group}__p{p}__ridge_weights']=fits['ridge']
                    curves[p]={'moment_plug_in_forecast':g['forecast'][p],'raw_Hebbian':ordinary,'moment_gain_Hebbian':calibrated,
                        'moment_selected_scalar_gain':gain,'ridge':ridge,'gain_fitting_extra_labelled_scenes':g['moment_pool_size']}
                rows[group]={'latent_coordinate_indices':columns,'native_classification_on_primary_scenes':native,
                    'native_continuous_MSE_on_fresh_scenes':float(np.mean((cf['prediction'][:,columns]-target)**2)),
                    'continuous_learning_curves':curves,
                    'scope':'Actual continuous labels in this role, not random threshold labels. Population exact formula is evaluated with estimated moments; predicted-vs-measured discrepancy includes moment-estimation and frozen-reader training variation.'}
            result={'module':'L3.4','version':'0.2','kind':'post-primary native annotation and separately frozen fresh continuous extension',
                'source_classification_view':name,'continuous_evaluation_first_id':args.evaluation_first_id,
                'continuous_evaluation_scenes':args.evaluation_scenes,'native_primary_scene_first_id':native_first,
                'groups':rows,'new_SNN_optimizer_updates':0,'audit_test_accessed':False}
            (folder/'report.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False));np.savez_compressed(folder/'continuous_scene_losses_and_weights.npz',**arrays)
            results.append({'view':name,'groups':{g:rows[g]['continuous_learning_curves'][max(rows[g]['continuous_learning_curves'])] for g in rows}})
        print(json.dumps({'completed_schedule':schedule,'views':len(configs)}),flush=True)
    if before!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():raise AssertionError('SNN checkpoint changed')
    summary={'semantic_views':len(results),'closed_form_regression_maps_fitted':total_fits,'new_SNN_training_runs':0,
        'new_SNN_optimizer_updates':0,'audit_test_accessed':False,'checkpoint_sha256':before,'checkpoint_unchanged':True,
        'continuous_evaluation_first_id':args.evaluation_first_id,'continuous_evaluation_scenes':args.evaluation_scenes,'views':results}
    (out/'summary.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('classification_analysis');p.add_argument('--output',required=True)
    p.add_argument('--evaluation-first-id',type=int,default=5700000);p.add_argument('--evaluation-scenes',type=int,default=8192);args=p.parse_args();s=run(args)
    print(json.dumps({'views':s['semantic_views'],'regression_map_fits':s['closed_form_regression_maps_fitted'],'checkpoint_unchanged':s['checkpoint_unchanged']}))
