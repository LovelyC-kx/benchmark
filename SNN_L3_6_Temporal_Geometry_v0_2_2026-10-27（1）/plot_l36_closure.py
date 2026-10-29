"""Scientific figure for the v0.2 closure; no simulated result substitution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent

def run():
    c=json.loads((ROOT/'evidence/temporal_consistency/report.json').read_text())
    p=json.loads((ROOT/'evidence/phase_readers/report.json').read_text())
    e=json.loads((ROOT/'evidence/event_scale_scan/report.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0]
    arrays=[[max(r['heldout_prediction_MSE'],1e-12) for r in c['cycles'] if r['family']==f] for f in ('orthogonal','affine')]
    ax.boxplot(arrays,showfliers=False);ax.set_xticks([1,2],['O(D)','Affine ridge'])
    ax.set_yscale('log');ax.set_ylabel('Heldout composed-vs-direct MSE')
    ax.set_title('(a) Cycles include offsets')
    ax.text(.03,.03,'180 triplets per family; 40 full rank\nCenter cycles are an algebraic zero control',transform=ax.transAxes,fontsize=9)
    ax=axes[0,1]
    colors={'center':'#7c7c7c','orthogonal':'#276f9e','affine':'#bf5b24'}
    for f,color in colors.items():
        for v,style in [('first_post','-'),('last_post','--')]:
            a=[r for r in c['equal_lag_transfers'] if r['family']==f and r['view']==v]
            ax.plot([r['evaluation_pair'][1] for r in a],[r['heldout_excess_prediction_MSE'] for r in a],style+'o',color=color,label=f'{f}, {v.split("_")[0]}')
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks([14,15]);ax.set_xlabel('Target step; reference map 12 -> 13')
    ax.set_ylabel('Fixed-map MSE minus local-map MSE');ax.set_title('(b) Equal-lag map transfer');ax.legend(fontsize=8,ncol=2)
    ax=axes[1,0];views=list(p['results']);sources=['H1','H2','H3','H4','C']
    gains=np.array([[p['results'][v][s]['paired_bound_selected_vs_pooled']['gain']*100 for s in sources] for v in views])
    lim=max(1.,np.max(np.abs(gains)));im=ax.imshow(gains,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(5),sources);ax.set_yticks(range(len(views)),views)
    ax.set_title('(c) Selected phase/clock reader vs pooled')
    for i in range(len(views)):
        for j in range(5):ax.text(j,i,f'{gains[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Error reduction (percentage points)',shrink=.8)
    ax.set_xlabel('0 / 25 positive simultaneous lower bounds')
    ax=axes[1,1]
    a=[r for r in e['rows'] if r['view']=='last_post' and r['source']=='C' and r['threshold_multiplier']==1 and r['measurement_sigma']==.2 and r['observer_bits_reference'] is None]
    a.sort(key=lambda r:r['delta']);delta=[r['delta'] for r in a]
    ax.plot(delta,[r['finite_scale_metric_trace'] for r in a],'o-',label='Mean across 128 scenes',color='#9c342c')
    ax.plot(delta,[r['scene_metric_trace_quantiles_50_95_99'][0] for r in a],'s--',label='Median across scenes',color='#285f8f')
    ax.set_xscale('log');ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(-.0005,2e4);ax.set_xlabel('Content perturbation delta')
    ax.set_ylabel('Finite-scale metric trace');ax.set_title('(d) Rare crossings dominate small-scale means')
    ax.legend(fontsize=9);ax.text(.03,.46,'last_post / C / threshold x1 / sigma .2\nPath crossings and quantiles must be read together',transform=ax.transAxes,fontsize=8)
    fig.suptitle('L3.6 closure: diagnostics on one old frozen SNN checkpoint\nOriginal training: one optimizer update; no architecture ranking',fontsize=12)
    directory=ROOT/'figures';directory.mkdir(exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(directory/f'L3_6_closure_evidence.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':run()
