"""Known control-variate check and nine frozen-SNN role diagnostics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from higher_order_controls import independent_control_comparison,contribution_stats
from task_structure import hermite3,KAPPA
from run_generalization import serial


def known_control():
    rng=np.random.default_rng(36501);z=rng.normal(size=(250000,1));x=.1*z+hermite3(z);theta=np.eye(1)
    optimal=.1+2*KAPPA*KAPPA/(np.sqrt(6)*(1-KAPPA*KAPPA))
    ordinary=np.array([[[0.,.1]]]);weighted=np.array([[[0.,optimal]]])
    a=contribution_stats(x,z,theta,ordinary);b=contribution_stats(x,z,theta,weighted)
    return {'exact_alignment':-KAPPA/np.sqrt(6),'exact_optimal_affine_slope':optimal,
        'ordinary_variance':a['mean_contribution_variance'],'optimal_variance':b['mean_contribution_variance'],
        'variance_reduction':1-b['mean_contribution_variance']/a['mean_contribution_variance'],
        'ordinary_mean':a['eta'],'optimal_mean':b['eta'],
        'scope':'Known nonlinear code, large independent contribution pool. The variance-optimal slope follows the weighted covariance identity, not a new neural architecture.'}


def run(args):
    from bridge import load_model,collect,flatten_contents
    from parent_sampling import fixed_batch
    from factor_sampling import factor_schema
    import torch
    torch.set_num_threads(1);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);model,cfg,meta=load_model(args.run_directory);checkpoint=Path(args.run_directory)/'best.pt'
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract=[('fit',512,'geometry_fit',5000000),('validation',512,'development_test',5300000),
        ('a',1024,'probe_train',5100000),('b',1024,'probe_train',5200000)]
    batches={k:fixed_batch(cfg.data,n,split,first,0) for k,n,split,first in contract}
    features={k:collect(model,b['x']) for k,b in batches.items()};z={k:flatten_contents(b,cfg.data) for k,b in batches.items()}
    results={}
    for focal,obs in [('C','last_post'),('C','joint_post'),('H1','last_post')]:
        source=Path(args.classification_analysis)/f'{focal}_{obs}_sphere_schedule_0';report=json.loads((source/'report.json').read_text());a=np.load(source/'frozen_readers_and_scene_losses.npz')
        schema=factor_schema(cfg.data,focal,batches['fit']['source_present']);groups={};arrays={}
        for group,roles in [('F',('focal',)),('G',('cotarget',)),('FG',('focal','cotarget'))]:
            g=report['target_groups'][group];columns=[i for i,row in enumerate(schema['coordinate_roles']) if row['role'] in roles]
            x={k:(v[obs].astype(float)-np.array(g['observation_origin']))/g['shared_geometry_fit_scalar_scale'] for k,v in features.items()}
            zz={k:v[:,columns] for k,v in z.items()};theta=a[group+'__theta']
            r=independent_control_comparison(x['fit'],zz['fit'],x['validation'],zz['validation'],x['a'],zz['a'],x['b'],zz['b'],theta)
            arrays.update({group+'__'+k:v for k,v in r.pop('coefficients').items()});groups[group]=r
        name=f'{focal}_{obs}';(out/(name+'.json')).write_text(json.dumps(serial(groups),indent=2,allow_nan=False));np.savez_compressed(out/(name+'_controls.npz'),**arrays);results[name]=groups
    if before!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():raise AssertionError('Checkpoint changed')
    final={'known_control':known_control(),'SNN_role_diagnostics':results,'pool_contract':contract,'checkpoint_sha256':before,
        'checkpoint_unchanged':True,'new_SNN_training_runs':0,'audit_test_accessed':False,
        'scope':'Estimator-variance and higher-order-alignment diagnostics only; no extra trained SNN or architecture performance comparison.'}
    (out/'summary.json').write_text(json.dumps(serial(final),indent=2,allow_nan=False));return final


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('classification_analysis');p.add_argument('--output',required=True);args=p.parse_args();j=run(args)
    print(json.dumps({'known_control_variance_reduction':j['known_control']['variance_reduction'],'SNN_groups':9,'checkpoint_unchanged':j['checkpoint_unchanged']}))
