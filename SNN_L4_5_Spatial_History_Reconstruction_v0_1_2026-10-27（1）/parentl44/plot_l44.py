"""Scientific development figure from actual saved records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text())

def run():
    main=read('evidence/confirmation/report.json');supp=read('evidence/supplements/report.json');path=read('evidence/path_reblocking/report.json');geo=read('evidence/event_geometry/report.json')
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.1),layout='constrained');colors=['#657383','#c34948','#35866c'];x=np.arange(3);width=.23
    keys=['H1_l0_t3_p4','H4_l0_t12_p13','C_l0_t12_p13'];values=[[],[],[]]
    for key in keys:
        c=main['results'][key];source=c['context']['source'];d=c['damaged_state_validation']['ridge'][source];r=c['methods']['target_self_delta']['repaired_state_validation']['ridge'][source]
        values[0].append(d['clean_probe_MSE']);values[1].append(d['MSE']);values[2].append(r['MSE'])
    for k,(v,name,color) in enumerate(zip(values,['Clean','Damaged','Target patch'],colors)):axes[0,0].bar(x+(k-1)*width,v,width,label=name,color=color)
    axes[0,0].set_xticks(x,['H1: t3 to t4','H4: t12 to t13','C: t12 to t13']);axes[0,0].set(ylabel='Independent ridge factor MSE',title='A. Content readability can be restored\nLayer0, confirmation n=2048');axes[0,0].legend(fontsize=8)
    maxima=[]
    for regime in ('natural','fixed_clean_events'):
        maxima.append(max(v[regime]['output_interaction_max_abs'] for c in supp['event_results'].values() for v in c.values()))
    axes[0,1].bar(['Natural events','Clean-event replay'],maxima,color=['#c34948','#3765a8']);axes[0,1].set(yscale='log',ylabel='Maximum absolute output interaction',title='B. Event-policy diagnostic\n12 contexts, n=256 shared confirmation scenes')
    for i,v in enumerate(maxima):axes[0,1].text(i,v*1.5,f'{v:.2g}',ha='center',fontsize=9)
    case=path['results']['C_l0_t12_p13::full'];v=[case['unblocked_repair_vs_damage_prediction_MSE'],0.,case['reblocked_vs_repaired_prediction_MSE']]
    axes[1,0].bar(['Restored vs damaged','Reblocked vs damaged','Reblocked vs restored'],v,color=['#35866c','#657383','#3765a8']);axes[1,0].set(ylabel='Native output discrepancy MSE',title='C. Future messages mediate this repair\nC/layer0/t12 to t13, n=256')
    for i,value in enumerate(v):axes[1,0].text(i,value+max(v)*.025,f'{value:.3g}',ha='center',fontsize=8)
    curve=geo['results']['C_l1_t12_p13']['dose_curve'];aa=[v['alpha'] for v in curve];risk=[v['native_H1_C_MSE'] for v in curve]
    axes[1,1].plot(aa,risk,'o-',color='#3765a8',lw=1.7);axes[1,1].axvline(0,color='#657383',ls=':',lw=.8);axes[1,1].axvline(1,color='#35866c',ls=':',lw=.8)
    axes[1,1].set(xlabel='Fixed patch dose alpha',ylabel='Original H1/C task MSE',title='D. Finite dose response\nC/layer1/t12 to t13, n=256; no dose selection')
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    axes[1,0].tick_params(axis='x',labelsize=8)
    fig.suptitle('L4.4 development evidence — same old checkpoint with one optimizer update\nContent restoration, task benefit and output interaction are separate measurements',fontsize=12)
    out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'L4_4_recovery_evidence.png',dpi=170);fig.savefig(out/'L4_4_recovery_evidence.svg');plt.close(fig)

if __name__=='__main__':run()
