"""Exact known cases, author-function parity and finite-p training calibration."""
import argparse,ast,hashlib,json,math
from pathlib import Path
import numpy as np
from geometry_generalization import geometry_terms,predict_error
from task_moments import (classification_error,empirical_task_moments,conditional_forecast,
    gaussian_task_mean,nonlinear_alignment)


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x


def run(author_file=None):
    a,b=.1,1.;variance=a*a+b*b;phi=a
    kappa=np.sqrt(2/np.pi);true_m=kappa*(a-b/np.sqrt(6))
    q=math.erfc(math.sqrt((3-a*np.sqrt(6)/b)/2))
    geom={'Psi':np.array([[variance]]),'Phi':np.array([[phi]]),'Omega':np.eye(1)}
    terms=geometry_terms(**geom)
    result={'kind':'known distributions and readout training calibration, not SNN findings',
        'matched_moments':{'nonlinear_code':'x=0.1z+(z^3-3z)/sqrt(6)',
            'Gaussian_code':'x=0.1z+epsilon, independent standard Gaussian epsilon',
            'Psi':variance,'Phi':phi,'Omega':1.,
            'actual_nonlinear_task_mean':true_m,'Gaussian_task_mean':kappa*a,
            'nonlinear_population_Hebbian_error':q,
            'Gaussian_population_Hebbian_error':math.atan(b/a)/math.pi,
            'same_four_statistics':{k:v for k,v in terms.items() if k not in ('linear_encoding','linear_remainder_covariance')},
            'exact_risk_identity':'R_p=q*P(w_p<0)+(1-q)*P(w_p>0)+0.5*P(w_p=0)'},
        'finite_p':[]}
    sizes=(1,2,4,8,16,32,64,128,512)
    counts={p:{'positive':0,'negative':0,'zero':0,'sum':0.,'sqsum':0.} for p in sizes}
    rng=np.random.default_rng(3401);repetitions=20000
    for start in range(0,repetitions,500):
        z=rng.normal(size=(min(500,repetitions-start),max(sizes)))
        x=a*z+b*(z**3-3*z)/np.sqrt(6);contribution=np.sign(z)*x
        prefix=np.cumsum(contribution,axis=1)
        for p in sizes:
            w=prefix[:,p-1]/p;c=counts[p]
            c['positive']+=int(np.sum(w>0));c['negative']+=int(np.sum(w<0));c['zero']+=int(np.sum(w==0));c['sum']+=float(w.sum());c['sqsum']+=float(w@w)
    # Forecast pool is distinct from the fresh training draws above.
    zr=rng.normal(size=(50000,1));xr=a*zr+b*(zr**3-3*zr)/np.sqrt(6);yr=np.sign(zr)
    moments={'M':np.array([[variance]]),'m':np.array([[true_m]])}
    for p in sizes:
        c=counts[p];positive=c['positive']/repetitions
        risk=q*c['negative']/repetitions+(1-q)*positive+.5*c['zero']/repetitions
        result['finite_p'].append({'samples':p,'training_repetitions':repetitions,
            'empirical_risk_using_exact_test_population':risk,'positive_weight_probability':positive,
            'four_statistic_forecast':predict_error(terms,p)['predicted_error'],
            'task_moment_normal_forecast':conditional_forecast(moments,xr,yr,p)['normal_weight_forecast'],
            'mean_training_weight':c['sum']/repetitions,'population_mean_weight':true_m,
            'training_weight_variance':c['sqsum']/repetitions-(c['sum']/repetitions)**2,
            'population_weight_variance':(variance-true_m*true_m)/p})
    zeros=np.zeros((100,8));y=np.tile(np.r_[-np.ones(50),np.ones(50)][:,None],(1,8))
    result['tie_control']={'ours_half_error':classification_error(zeros,y)['mean_error'],
        'legacy_strict_error':classification_error(zeros,y,'legacy_strict')['mean_error']}
    if author_file:
        path=Path(author_file);tree=ast.parse(path.read_text());wanted={'theta','get_geometry','theory_part','emp_error'}
        functions=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in wanted]
        ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
        dense_phi=np.diag([.8,.4,.2]);dense_psi=dense_phi@dense_phi.T+np.diag([.2,.3,.7]);om=np.eye(3)
        ours=geometry_terms(dense_psi,dense_phi,om);author=ns['get_geometry'](dense_psi,dense_phi,om,16)
        np.random.seed(3402);z=np.random.randn(1200,4)
        author_zero=ns['emp_error'](np.zeros((1200,16)),z,32,n_task=300,min_task=100,svc_analysis=False)[0]
        result['author_selected_function_execution']={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'functions_executed':sorted(wanted),'upstream_training_executed':False,
            'author_zero_code_error':float(author_zero),
            'named_geometry_max_absolute_difference':float(max(abs(ours['PR']-author[0]),abs(ours['c']-author[1]),abs(ours['s']-author[2]),abs(ours['f']-author[3]))),
            'prediction_error_difference':float(predict_error(ours,16)['predicted_error']-author[-1]),
            'warning':'Executing exact selected NumPy function definitions via AST; no upstream MLP training, imports or datasets executed. Tie behavior matters for this sparse-code application, not an assertion about the published continuous-model results.'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--author-utils');args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    d=run(args.author_utils);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(serial(d),indent=2))
    print(json.dumps({'output':str(out),'matched_population_errors':[d['matched_moments']['nonlinear_population_Hebbian_error'],d['matched_moments']['Gaussian_population_Hebbian_error']],
        'curve':[(r['samples'],r['empirical_risk_using_exact_test_population']) for r in d['finite_p']],'author':d.get('author_selected_function_execution')}))
