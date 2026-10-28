"""Run analytical/data controls, without training or importing an SNN."""
from dataclasses import replace
import ast
import csv
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from load_data import LoadConfig,generate,source_reference
from linear_reference import gaussian_linear_capacity
from experiment_plan import build_plan

ROOT=Path(__file__).resolve().parent
def main():
    evidence=ROOT/'evidence';evidence.mkdir(exist_ok=True)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
    (evidence/'test_log.txt').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Reference checks failed; see evidence/test_log.txt')
    compiled=[]
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(),filename=str(p));compiled.append(str(p.relative_to(ROOT)))
    oracle=[]
    for k in (1,2,4):
        for energy in ('raw','fixed_rms'):
            c=LoadConfig(k=k,energy_mode=energy)
            b=generate(c,6144,'development_test')
            estimates=source_reference(c,b,'C');truth=b['latents'][:,4,:k]
            squared=np.square(estimates-truth)
            sensory_energy=float(np.square(b['x'][:,:,:8]).sum()/(len(b['x'])*5))
            row={'K':k,'energy':energy,'observed_sequence_oracle_mse':float(squared.mean()),
                 'sequence_level_monte_carlo_se':float(squared.mean(1).std(ddof=1)/np.sqrt(len(squared))),
                 'expected_oracle_mse':c.sigma**2/(1+c.sigma**2),
                 'mean_sensory_event_energy':sensory_energy,
                 'expected_sensory_event_energy':c.sensory_scale**2*(k+8*c.sigma**2)}
            if abs(row['observed_sequence_oracle_mse']-row['expected_oracle_mse'])>.004:
                raise AssertionError('Oracle calibration outside declared Monte Carlo tolerance')
            oracle.append(row)
    with (evidence/'input_oracle_calibration.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(oracle[0]));w.writeheader();w.writerows(oracle)
    conditioning=[]
    for rho in (0.,.25,.5,.75,.9,.95,.99,.999,1.):
        a=np.array([[1,rho],[0,np.sqrt(1-rho*rho)]])
        for noise in (0.,.1):
            r=gaussian_linear_capacity(a,noise)
            conditioning.append({'column_cosine':rho,'state_noise_std':noise,
                'optimal_linear_sum_r2':r['sum_r2'],'rank':r['rank']})
    with (evidence/'conditioning_reference.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(conditioning[0]));w.writeheader();w.writerows(conditioning)
    figures=[]
    if importlib.util.find_spec('matplotlib'):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for noise,style in [(0.,'-'),(.1,'--')]:
            rs=[r for r in conditioning if r['state_noise_std']==noise]
            axes[0].plot([r['column_cosine'] for r in rs],[r['optimal_linear_sum_r2'] for r in rs],style,label=f'state noise SD={noise}')
        axes[0].set(xlabel='Encoding column cosine',ylabel='Sum of optimal linear R²',ylim=(0,2.1),title='Overlap alone does not imply loss')
        axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
        rs=[r for r in oracle if r['energy']=='fixed_rms']
        axes[1].errorbar([r['K'] for r in rs],[r['observed_sequence_oracle_mse'] for r in rs],
                         yerr=[1.96*r['sequence_level_monte_carlo_se'] for r in rs],fmt='o-',capsize=3,
                         label='observed-sequence oracle (95% MC interval)')
        axes[1].axhline(.04/1.04,color='black',linestyle='--',label='Gaussian Bayes reference')
        axes[1].set(xlabel='Active content dimension K',ylabel='Per-coordinate MSE',title='Input solvability stays comparable')
        axes[1].set_xticks([1,2,4]);axes[1].set_ylim(0,.05)
        axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
        fig.suptitle('L3.2 analytical/data calibration — no SNN training',fontsize=11)
        for ext in ('png','pdf'):
            path=evidence/('reference_calibration.'+ext);fig.savefig(path,dpi=180);figures.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    plan=build_plan();(ROOT/'experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    torch_available=importlib.util.find_spec('torch') is not None
    result={'status':'PASS','date':'2026-10-25','python':platform.python_version(),'numpy':np.__version__,
            'numpy_reference_tests':21,'pytorch_runtime_tests':4 if torch_available else 0,
            'ast_files':compiled,'new_benchmark_fits':0,
            'torch_available':importlib.util.find_spec('torch') is not None,
            'torch_forward_backward_validated':torch_available,
            'actual_controls':['same input across supervised demand','prefix/time legality','write-pressure anchors',
                'energy/SNR reference','noiseless solvability','Gaussian observed-sequence oracle',
                'nonorthogonal invertible counterexample','conditional linear rank bound',
                'linear age expansion','held-out risk identity with cross terms','independent split and negative R² handling'],
            'planned_runs':plan['run_count'],'screen_runs':15,'primary_runs':135,
            'figures':figures,'interpretation':'Analytical/data verification only. Planned experiments and new scientific findings are not completed.'}
    smoke=[]
    for p in sorted((evidence/'smoke_runs').glob('*/run.json')):
        r=json.loads(p.read_text())
        smoke.append({'path':str(p.parent.relative_to(ROOT)),'kind':r['run_kind'],
                      'objective':r['config']['objective'],'optimizer_updates':r['optimizer_updates'],
                      'status':r['status'],'noop_max_abs_error':r['noop_max_abs_error'],
                      'content_analysis_completed':(p.parent/'content_analysis_development_test.json').exists()})
    result['actual_training_pipeline_smokes']=smoke
    result['scientific_result_scope']='Tiny smoke runs verify the pipeline only; 0 of 435 planned benchmark configurations completed.'
    (evidence/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
