"""Plot actual L4.1 development risks and the measured stream registry."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def run():
    r=json.loads((ROOT/'evidence/frozen_access_review/report.json').read_text());grid=json.loads((ROOT/'evidence/interface_grid.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    modes=['last','mean','sum','exponential','trajectory'];phases=['output','pre','post']
    for ax,source,label in [(axes[0,0],'H1','(a)'),(axes[0,1],'C','(b)')]:
        a=np.array([[r['risks'][p+'__'+m]['512'][source]['classification'] for p in phases] for m in modes])
        im=ax.imshow(a,cmap='viridis_r',vmin=.3,vmax=.5,aspect='auto')
        ax.set_xticks(range(3),phases);ax.set_yticks(range(5),modes);ax.set_title(f'{label} {source}: auxiliary classification error')
        for i in range(5):
            for j in range(3):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if a[i,j]>.435 else 'black')
        fig.colorbar(im,ax=ax,shrink=.85,label='Error; 512 unique training scenes')
    ax=axes[1,0];names=['output__last','output__mean','output__trajectory','post__last','pre__last'];xx=np.arange(len(names));width=.35
    for source,offset,color in [('H1',-.5,'#276f9e'),('C',.5,'#c36929')]:
        ax.bar(xx+offset*width,[r['risks'][n]['512'][source]['MSE'] for n in names],width,label=source,color=color)
        ax.axhline(r['constant_reader_risks']['512'][source]['MSE'],color=color,linestyle='--',linewidth=1,label=source+' train-mean baseline')
    ax.set_xticks(xx,['output\nlast','output\nmean','output\ntrajectory','post\nlast','pre\nlast']);ax.set_ylim(0,1.3)
    ax.set_ylabel('Latent reconstruction MSE');ax.set_title('(c) A richer access need not improve finite-label MSE');ax.legend()
    ax=axes[1,1];values=[]
    for mode in modes:
        row=next(x for x in grid['cases'] if x['objective']=='classification' and x['backbone']=='mlp' and x['cell']=='lif_hard' and x['phase']=='post' and x['readout']==mode)
        values.append(row['stream_registry']['unique_storage_bytes_within_registered_operational_tensors']/2)
    ax.bar(range(5),values,color=['#777','#447b9c','#447b9c','#447b9c','#ba632b']);ax.set_xticks(range(5),['last','mean','sum','exp','trajectory'])
    ax.set_ylabel('Registered unique tensor bytes / scene');ax.set_title('(d) State includes the readout owner')
    ax.text(.03,.95,'Untrained MLP, D=2, T=3, post\nCell/cache/readout only; excludes diagnostics,\nparameters, allocator and peak/device energy',transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle('L4.1 access audit: one old checkpoint with one optimizer update\nTwo of 12 simultaneous classification comparisons have positive lower bounds; no architecture ranking',fontsize=11)
    for suffix in ('png','svg'):fig.savefig(ROOT/'figures'/f'L4_1_access_evidence.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':run()
