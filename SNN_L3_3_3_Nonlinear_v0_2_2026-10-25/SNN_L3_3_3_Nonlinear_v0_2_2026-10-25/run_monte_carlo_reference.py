"""Repeated independent estimator calibration; no SNN fitting or evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from functional_anova import error_risk_budget,paired_bootstrap
from calibration_cases import function_lattice
from run_calibrations import serial


def run():
    names=('F','G','U','E');truth=np.array([.25,.49,.04,.78])
    rows=[]
    for n in (128,512,2048,8192):
        estimates=[]
        for seed in range(16):
            rng=np.random.default_rng(np.random.SeedSequence([333,seed,n]))
            a,b=rng.normal(size=(2,n,4))
            errors=function_lattice(lambda x:.5*x[:,1]+.7*x[:,0]*x[:,1]+.2*x[:,3],a,b)
            _,s=error_risk_budget(errors,names,('F',))
            p={row['mask']:row['power_per_output'] for row in s['component_rows']}
            estimates.append([p[2],p[3],p[8],s['target_contract']['non_target_conditional_variability_power']])
        estimates=np.array(estimates)
        rows.append({'independent_pairs':n,'independent_estimator_seeds':16,
            'quantity_order':['G','F_x_G','E','non_target_total'],
            'population_truth':truth,'mean_estimates':estimates.mean(0),
            'rmse':np.sqrt(np.mean((estimates-truth)**2,axis=0)),
            'negative_component_estimate_count':int(np.sum(estimates[:,:3]<0)),
            'all_estimates':estimates})
    rng=np.random.default_rng(3337);a,b=rng.normal(size=(2,4096,4))
    errors=function_lattice(lambda x:.5*x[:,1]+.7*x[:,0]*x[:,1]+.2*x[:,3],a,b)
    interval=paired_bootstrap(errors,names,('F',),draws=200,seed=3338)
    return {'kind':'known Gaussian product system; not SNN findings',
        'error_function':'0.5G+0.7FG+0.2E; all factors independent standard Gaussian',
        'sample_size_calibration':rows,'one_fixed_example_joint_bootstrap':interval,
        'scope':'Independent Monte Carlo seeds assess an estimator under one known function. They are not SNN training seeds and do not establish universal bootstrap coverage.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True);result=run()
    target.write_text(json.dumps(serial(result),indent=2))
    print(json.dumps({'output':str(target),'non_target_total_rmse_by_n':{r['independent_pairs']:float(r['rmse'][-1]) for r in result['sample_size_calibration']}}))
