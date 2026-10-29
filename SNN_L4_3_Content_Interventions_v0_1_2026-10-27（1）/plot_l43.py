"""Scientific development figure; no formal architecture claim."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def run():
    s=json.loads((ROOT/'evidence/state_interventions/report.json').read_text());f=json.loads((ROOT/'evidence/stability_fitted/freeze_contract.json').read_text());r=json.loads((ROOT/'evidence/stability_followup/report.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(13,8.3),layout='constrained');keys=list(f['fit_diagnostics']);x=np.arange(len(keys))
    colors=['#c44943','#3765a8','#36866b']
    for color,name,label in zip(colors,['initial_unregularized','author_default','shrink_fixed'],['Unregularized reference','Author LEACE defaults','Fixed covariance shrinkage']):
        axes[0,0].plot(x,[f['fit_diagnostics'][k][name]['operator_norm'] for k in keys],marker='o',ms=3,lw=1.4,color=color,label=label)
    axes[0,0].set(yscale='log',ylabel='Operator spectral norm',title='A. Stability is a separate intervention axis')
    axes[0,0].set_xticks(x,[k.replace('_','/') for k in keys],rotation=75,fontsize=7);axes[0,0].legend(fontsize=7)
    cases=['H1_l0_t3','H4_l0_t12','C_l0_t12'];xx=np.arange(3);width=.23
    for k,(name,label,color) in enumerate([('baseline','Before: ridge','#626c77'),('ridge','After: refitted ridge','#3765a8'),('mlp','After: refitted MLP','#36866b')]):
        values=[]
        for case in cases:
            source=case.split('_')[0];rows=r['results'][case]
            pr=rows['no_op']['frozen_validation_probes']['ridge'] if name=='baseline' else rows['shrink_fixed']['refitted_after_erasure_probes'][name]
            values.append(pr[source]['original_label_MSE'])
        axes[0,1].bar(xx+(k-1)*width,values,width,label=label,color=color)
    axes[0,1].set_xticks(xx,['H1: layer0, t3','H4: layer0, t12','C: layer0, t12']);axes[0,1].set(ylabel='Factor recovery MSE',title='B. Fresh development scenes: target hit');axes[0,1].legend(fontsize=7)
    d=r['results']['H1_l0_t3']['shrink_transport']['operator_and_scope'];names=['Operator','Scope','Interaction','Total']
    values=[d['operator_fidelity_MSE'],d['scope_fidelity_MSE'],d['interaction'],d['natural_fidelity_MSE']]
    axes[1,0].bar(names,values,color=['#3765a8','#b8863b','#c44943','#36866b']);axes[1,0].axhline(0,color='black',lw=.5)
    axes[1,0].set(ylabel='Output-space squared discrepancy',title='C. Operator + scope + interaction = total\nH1/layer0/t3, content transfer')
    phase=s['phase_control'];values=[phase['before_G_zero_v_prediction_change_max_abs'],phase['after_final_step_zero_v_prediction_change_max_abs']]
    axes[1,1].bar(['Before observation/readout','After final step'],values,color=['#3765a8','#626c77'])
    axes[1,1].set(ylabel='Maximum absolute native output change',title='D. Same state, different intervention phase')
    for i,value in enumerate(values):axes[1,1].text(i,value+.025,f'{value:.4g}',ha='center',fontsize=9)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('L4.3 development evidence — one old checkpoint with one optimizer update\nAdaptive stability follow-up: n=1024 pairs; no architecture or real-data conclusion',fontsize=12)
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'L4_3_intervention_evidence.png',dpi=170);fig.savefig(folder/'L4_3_intervention_evidence.svg');plt.close(fig)

if __name__=='__main__':run()
