"""Scientific figures from recorded data; no synthesized model findings."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def run():
    r=json.loads((ROOT/'evidence/reader_ladder/report.json').read_text())
    s=json.loads((ROOT/'evidence/task_subspaces/report.json').read_text())
    c=json.loads((ROOT/'evidence/known_latent_controls.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    views=['output__last','output__mean','output__trajectory','post__last','post__mean','post__trajectory']
    labels=['spike\nlast','spike\nmean','spike\ntrajectory','post\nlast','post\nmean','post\ntrajectory']
    colors=['#356d9a','#d08238','#54917b']
    for ax,source in zip(axes[0],['H1','C']):
        for kind,color,label in zip(['ridge','budget_mlp','residual'],colors,['Ridge','MLP (parameter cap)','Ridge + residual MLP']):
            vals=np.array([[r['risks'][v+'__draw'+str(d)+'__p512__'+kind][source]['MSE'] for d in (0,1)] for v in views])
            ax.plot(np.arange(6),vals.mean(1),'o-',color=color,label=label)
            ax.fill_between(np.arange(6),vals.min(1),vals.max(1),color=color,alpha=.1)
        ax.set_xticks(range(6),labels);ax.set_ylabel('Native continuous-target MSE')
        ax.set_title(source+': 512 fit + 128 selection labels per reader')
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=9)
    ax=axes[1,0];names=['row','null'];vals=[];randoms=[]
    for kind in names:
        vals.append(np.mean([s['risks']['requested8__'+kind+'__draw'+str(d)]['C']['MSE'] for d in (0,1)]))
        randoms.append([np.mean([s['risks']['requested8__random_'+kind+'_'+str(k)+'__draw'+str(d)]['C']['MSE'] for d in (0,1)]) for k in range(5)])
    ax.scatter([0,1],vals,marker='D',s=90,color='#356d9a',label='Actual current-head subspace',zorder=4)
    for j,a in enumerate(randoms):ax.scatter(j+np.linspace(-.12,.12,len(a)),a,color='black',s=24,label='Five matched-rank random subspaces' if j==0 else None,zorder=3)
    baseline=np.mean([v['C']['MSE'] for v in s['constant_risks'].values()]);ax.axhline(baseline,ls='--',color='gray',label='Train-mean baseline')
    ax.set_xticks([0,1],['row (8 dimensions)','null (24 dimensions)']);ax.set_ylim(.84,max(1.05,baseline+.03))
    ax.set_ylabel('C MSE on separate 4,096-scene pool');ax.set_title('Content recoverability versus the requested 8-output head')
    ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];parts=c['risk_chain']['pieces'];bottom=0.
    for i,(name,value) in enumerate(parts.items()):
        ax.bar([0],value,bottom=bottom,color=plt.get_cmap('Set2')(i),label=name.replace('_',' '))
        ax.text(0,bottom+value/2,f'{value:.1f}',ha='center',va='center');bottom+=value
    ax.set_xticks([0],['Explicit nested Gaussian reference']);ax.set_xlim(-.6,.6);ax.set_ylabel('Population MSE components')
    ax.set_title('Known-answer risk decomposition; total = 1.1');ax.legend(fontsize=9,loc='upper right')
    fig.suptitle('L4.2: task-conditioned readability and native-head geometry\nOne frozen SNN checkpoint with one original optimizer update; shading is two reader draws, not confidence intervals',fontsize=13)
    dst=ROOT/'figures';dst.mkdir(exist_ok=True)
    fig.savefig(dst/'L4_2_task_geometry_and_readers.png',dpi=150)
    fig.savefig(dst/'L4_2_task_geometry_and_readers.svg')
    plt.close(fig)

if __name__=='__main__':run()
