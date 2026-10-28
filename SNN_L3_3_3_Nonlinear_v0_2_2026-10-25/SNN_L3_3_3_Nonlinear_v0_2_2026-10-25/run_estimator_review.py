"""Controlled estimator comparison: improvements and failure modes retained."""
import argparse,json
from pathlib import Path
import numpy as np
from functional_anova import decompose,error_risk_budget
from calibration_cases import function_lattice
from run_calibrations import serial


def run():
    names=('F','G','U','E');rows=[]
    cases={'additive':(lambda x:x[:,0]+.5*x[:,1]+.2*x[:,3],0.,.29,1.),
           'target_interaction':(lambda x:.5*x[:,1]+.7*x[:,0]*x[:,1]+.2*x[:,3],.49,.78,0.),
           'pure_third':(lambda x:x[:,0]*x[:,1]*x[:,2],0.,1.,0.),
           'binary_main':(lambda x:x[:,0],0.,0.,1.)}
    for n in (128,512,2048):
        for name,(fn,fg_truth,noise_truth,main_truth) in cases.items():
            estimates={method:[] for method in ('legacy_covariance','symmetric_difference')}
            for seed in range(32):
                rng=np.random.default_rng(np.random.SeedSequence([334,n,seed]))
                a,b=rng.choice((-1.,1.),size=(2,n,4)) if name=='binary_main' else rng.normal(size=(2,n,4))
                h=function_lattice(fn,a,b)
                for method in estimates:
                    fit=decompose(h,names,estimator=method)
                    _,risk=error_risk_budget(h,names,('F',),estimator=method)
                    fg=float(np.trace(fit['components'][3]))
                    high=float(np.sqrt(sum(np.sum(fit['components'][m]**2) for m in range(1,16) if m.bit_count()>1)))
                    estimates[method].append([fg,risk['target_contract']['non_target_conditional_variability_power'],high,float(np.trace(fit['components'][1]))])
            summary={}
            for method,e in estimates.items():
                e=np.array(e);summary[method]={'FG_estimator_mean':float(e[:,0].mean()),'FG_RMSE':float(np.sqrt(np.mean((e[:,0]-fg_truth)**2))),
                    'non_target_RMSE':float(np.sqrt(np.mean((e[:,1]-noise_truth)**2))),
                    'F_main_RMSE':float(np.sqrt(np.mean((e[:,3]-main_truth)**2))),
                    'mean_higher_order_estimated_norm':float(e[:,2].mean()),'all_estimates':e}
            rows.append({'case':name,'pairs':n,'estimator_seeds':32,'FG_truth':fg_truth,'methods':summary})
    return {'kind':'known independent Gaussian and binary input functions; not SNN findings','rows':rows,
        'scope':'The symmetric estimator enforces exact additive interaction zeros and uses A/B symmetrically. It is not universally lower-variance for every nonzero effect; both estimators remain available.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    d=run();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(serial(d),indent=2))
    print(json.dumps([{'case':r['case'],'pairs':r['pairs'],'legacy_FG_RMSE':r['methods']['legacy_covariance']['FG_RMSE'],'symmetric_FG_RMSE':r['methods']['symmetric_difference']['FG_RMSE']} for r in d['rows']]))
