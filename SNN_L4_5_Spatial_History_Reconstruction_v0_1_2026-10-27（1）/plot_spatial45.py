"""Scientific figures from saved development evidence; no selected best/worst scene."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bridge45 import ROOT,read_json
from spatial_data45 import batch

def main():
    r=read_json(ROOT/'evidence/confirmation/report.json');m=read_json(ROOT/'evidence/matched_summaries/report.json');s=read_json(ROOT/'evidence/supplements/report.json')
    a=batch(1,45060000);fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    fig.suptitle('L4.5 development evidence: frozen untrained spatial cores and offline fitted decoders',fontsize=15)
    strips=[]
    for t in (0,2,5):strips.append(np.moveaxis(a['x'][0,t,[2,1,0]],0,-1))
    strips.append(np.moveaxis(a['y'][0,[2,1,0]],0,-1));combined=np.concatenate(strips,axis=1)
    ax=axs[0,0];ax.imshow(combined,vmin=0,vmax=1);ax.set_xticks([7.5,23.5,39.5,55.5],['past 0','past 2','current 5','target 5']);ax.set_yticks([])
    ax.set_title('A  Same scene: latest H is hidden at the deadline',loc='left',fontsize=11)
    ax.set_xlabel('Four-band toy data shown with channels 2, 1, 0; first fixed confirmation scene')
    ax=axs[0,1];keys=list(r['clean_reports']);labels=['current only','latest valid']+[k.replace('__',' / ').replace('continuous','linear') for k in keys]
    values=[r['baseline_reports'][k]['clipped_MSE']['H'] for k in ['current_only','latest_valid_history']]+[r['clean_reports'][k]['last']['spatial3']['clipped_MSE']['H'] for k in keys]
    ax.barh(np.arange(len(values)),values,color=['#999999','#2a9d8f']+['#577590']*6);ax.set_yticks(np.arange(len(values)),labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel('H region MSE, clipped prediction');ax.set_title('B  Pixel recovery needs more than a current-content label',loc='left',fontsize=11)
    ax=axs[1,0];conditions=[(k,v) for k in keys for v in ('last','mean')];colors={'GAP8':'#577590','PCA8':'#e76f51','random8':'#8c8c8c'}
    for method,color in colors.items():
        x=[m['results'][k][v][method]['clipped_MSE']['H'] for k,v in conditions];ax.plot(x,np.arange(len(x)),'.-',color=color,label=method,lw=.6,ms=6)
    ax.set_yticks(np.arange(len(conditions)),[k.replace('__',' / ').replace('continuous','linear')+' / '+v for k,v in conditions],fontsize=7);ax.invert_yaxis();ax.set_xlabel('H region MSE; 512 fresh exploratory scenes');ax.legend(fontsize=8)
    ax.set_title('C  Equal learned decoder parameters; summary costs still differ',loc='left',fontsize=11)
    ax=axs[1,1];xs=np.arange(4);subsets=['none','v','a','va']
    for key,j in s['ALIF_joint_owner_factorials'].items():
        for view,style in [('last','-'),('mean','--')]:
            vals=[max(j['views'][view][k]['clean_prediction_max_abs'],1e-12) for k in subsets]
            ax.plot(xs,vals,style,marker='o',label=key.split('__')[0]+' / '+view)
    ax.set_xticks(xs,['none','membrane','adaptation','both']);ax.set_yscale('log');ax.set_ylabel('Maximum prediction gap to clean output');ax.set_xlabel('Restoration at step 3 after a joint state corruption at step 2')
    ax.set_title('D  Core restoration leaves earlier mean-readout history',loc='left',fontsize=11);ax.legend(fontsize=8);ax.text(.03,.04,'Exact zero shown at 1e-12 for display',transform=ax.transAxes,fontsize=8)
    p=ROOT/'figures';p.mkdir(exist_ok=True);fig.savefig(p/'L4_5_evidence_overview.png',dpi=160);fig.savefig(p/'L4_5_evidence_overview.svg');plt.close(fig)
    # Source-response diagram is an input intervention map, not a local storage map.
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');key='cnn2d__continuous';v=s['results'][key]['views']['last']
    sources=['source_H','source_U_src','source_C','drop_frame_0','drop_frame_2','drop_frame_5'];regions=['H','C','changed_C','U_vis']
    arr=np.array([[v[q]['output_response_MSE'][z] for z in regions] for q in sources]);im=axs[0].imshow(arr,aspect='auto',cmap='magma');fig.colorbar(im,ax=axs[0],label='Output response MSE')
    axs[0].set_xticks(range(4),regions);axs[0].set_yticks(range(6),sources);axs[0].set_title('Source / output-region response, CNN linear, last')
    freq=s['frequency_patch_factorials'];x=np.arange(6);values=[freq[k]['low_outside_H_energy_fraction'] for k in keys]
    axs[1].bar(x,values,color='#577590');axs[1].set_xticks(x,[k.replace('__','\n') for k in keys],fontsize=8);axs[1].set_ylim(0,.25);axs[1].set_ylabel('Fraction of low-band state-patch energy outside H');axs[1].set_title('Spatial frequency projection spreads a local state patch')
    fig.savefig(p/'L4_5_source_and_frequency.png',dpi=160);fig.savefig(p/'L4_5_source_and_frequency.svg');plt.close(fig)

if __name__=='__main__':main()
