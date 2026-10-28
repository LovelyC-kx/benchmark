"""Gaussian finite-dimensional calibration separates two theory errors."""
import argparse,json,math
from pathlib import Path
import numpy as np
from geometry_generalization import geometry_terms,predict_error
from task_moments import teachers,labels,hebbian_curves,empirical_task_moments,conditional_forecast
from run_generalization import serial


def run():
    rng=np.random.default_rng(34028);d,n=40,80;a=rng.normal(size=(n,d))/np.sqrt(d)
    noise=.4;om=np.eye(d);phi=a;psi=a@a.T+noise**2*np.eye(n)
    terms=geometry_terms(psi,phi,om);theta=teachers(d,96)
    pools=[]
    for count in (2048,2048,2048,2048):
        z=rng.normal(size=(count,d));x=z@a.T+noise*rng.normal(size=(count,n));pools.append((x,labels(z,theta)))
    (tx,ty),(ex,ey),(mx,my),(fx,fy)=pools;sizes=(4,16,64,256,1024)
    empirical=hebbian_curves(tx,ty,ex,ey,sizes,repeats=24)
    forecasts=[predict_error(terms,p) for p in sizes];mom=empirical_task_moments(mx,my)
    task_forecasts=[conditional_forecast(mom,fx,fy,p) for p in sizes]
    return {'kind':'Known Gaussian synthetic code; not a trained SNN',
        'dimension':{'responses':n,'latents':d,'tasks':96},'terms':terms,
        'four_statistic_forecast':forecasts,'empirical_Hebbian':empirical,'task_moment_normal_forecast':task_forecasts,
        'one_dimensional_noiseless_control':{'code':'x=z, z~N(0,1), y=sign(z)',
            'exact_error_for_every_positive_p':0.,
            'four_statistic_p1':predict_error(geometry_terms(np.eye(1),np.eye(1),np.eye(1)),1)['predicted_error'],
            'task_moment_normal_p1':.5*math.erfc(math.sqrt((2/math.pi)/(1-2/math.pi))/math.sqrt(2)),
            'reason':'The empirical Hebbian weight is a mean of |z| and is strictly positive. Gaussian approximations assign mass to impossible negative weights.'},
        'interpretation':'Finite dimension and Gaussian training-weight approximation already create errors without non-Gaussian neural responses. Higher-order claims require these controls.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=run();out.write_text(json.dumps(serial(result),indent=2,allow_nan=False));print(json.dumps([{'p':e['samples'],'actual':e['mean_error'],'four_stat':f['predicted_error']} for e,f in zip(result['empirical_Hebbian'],result['four_statistic_forecast'])]))
