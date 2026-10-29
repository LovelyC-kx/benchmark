"""Scientific figures from bundled saved evidence; no model fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parent

def load(path):return json.loads((ROOT/path).read_text())
def save(fig,name):
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    for extension in ('png','svg'):fig.savefig(folder/(name+'.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)

def run():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    r=load('evidence/frozen_temporal_review/report.json');e=load('evidence/temporal_extensions/report.json');k=load('evidence/known_temporal_controls.json')
    colors={'first_pre':'#197f85','first_post':'#af7a2f','first_spike':'#8e5aa0','last_pre':'#2d6a9d','last_post':'#b75e64','last_spike':'#777777'}
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    fig.suptitle('L3.6 | One frozen one-update SNN; formal benchmark 0/435',fontsize=17)
    for ax,source,onset,letter in ((axes[0,0],'H1',2,'A'),(axes[0,1],'C',12,'B')):
        for view in ('first_post','last_post','first_spike','last_spike'):
            diag=np.diag(r['risk_matrices'][view]['raw'][source]['classification'])
            ax.plot(np.arange(onset,16),diag[onset:],'.-',color=colors[view],label=view.replace('_',' '))
        ax.axhline(.5,color='gray',ls=':',lw=1);ax.set_ylim(0,.57);ax.set_xlabel('Observation step');ax.set_ylabel('Within-time classification error')
        ax.set_title(f'{letter}  Same content, readers fitted at each time: {source}');ax.legend(fontsize=8)
    ax=axes[0,2];m=np.array(r['risk_matrices']['first_post']['raw']['H1']['classification'])
    im=ax.imshow(m,vmin=0,vmax=1,origin='lower',cmap='viridis',aspect='auto');ax.axvline(1.5,color='white',ls=':',lw=1);ax.axhline(1.5,color='white',ls=':',lw=1)
    ax.set_xlabel('State observation time');ax.set_ylabel('Reader training time');ax.set_title('C  First post / H1: fixed-reader transfer');fig.colorbar(im,ax=ax,label='Error',shrink=.8)
    ax=axes[1,0];t=np.arange(2,16);ax.plot(t,m[15,2:],'o-',label='Fixed reader trained at step 15',color='#b75e64');ax.plot(t,np.diag(m)[2:],'o-',label='Reader fitted at each time',color='#197f85')
    ax.set_ylim(0,.57);ax.set_xlabel('Observation step');ax.set_ylabel('H1 classification error');ax.set_title('D  Reading earlier states; memory is not extended');ax.legend(fontsize=8)
    ax=axes[1,1];times=(13,15,19,23);ages=(1,3,7,11)
    for family,color in (('raw','#b75e64'),('center','#af7a2f'),('orthogonal','#4979a8'),('affine','#197f85')):
        values=[e['blank_delay_scan']['first_post'][str(t)][family]['C']['classification_error'] for t in times]
        ax.plot(ages,values,'o-',label=family,color=color)
    ax.set_xlabel('Blank steps after C write');ax.set_ylabel('C classification error');ax.set_ylim(0,.57);ax.set_title('E  Same step-15 head, extra unlabelled calibration');ax.legend(fontsize=8)
    ax=axes[1,2];rows=[v for v in e['finite_scale_task_metrics'] if v['source']=='C' and v['view']=='last_post'];ds=[v['delta'] for v in rows]
    values=[np.trace(np.array(v['mean_Euclidean_metric'])[-1]) for v in rows];ax.plot(ds,values,'o-',color='#197f85');ax.set_xscale('log');ax.set_xlabel('Content perturbation scale delta');ax.set_ylabel('Trace of mean finite-scale metric at step 15')
    ax.set_title('F  Pathwise content response and spike boundaries');ax.text(.04,.85,'Global C decoding error ≈ 0.43\nFixed-noise paths; conditional-mean geometry differs',transform=ax.transAxes,fontsize=9)
    save(fig,'L3_6_frozen_temporal_evidence')
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);fig.suptitle('L3.6 | Known-law controls; not trained-SNN or hardware results',fontsize=16)
    ax=axes[0,0];rows=k['pure_rotation'];angles=[v['angle']*180/np.pi for v in rows]
    ax.plot(angles,[v['fixed_binary_error'] for v in rows],'o-',label='Frozen-direction error',color='#b75e64');ax.plot(angles,[v['theory_error'] for v in rows],':',label='Exact Gaussian law',color='black');ax.plot(angles,np.zeros(len(angles)),'o-',label='Bayes error (all orientations)',color='#197f85')
    ax.set_xlabel('Rotation (degrees)');ax.set_ylabel('Classification error');ax.set_title('A  Orientation can break a perfect fixed reader');ax.legend(fontsize=9)
    ax=axes[0,1];row=k['dominant_nuisance_alignment'];ax.bar([0,1],[row['raw_target_error'],row['aligned_target_error']],color=['#af7a2f','#197f85']);ax.set_xticks([0,1],['Raw target reader','Whole-state O(D) alignment']);ax.set_ylabel('Target classification error');ax.set_title('B  Better state fit can harm target reading')
    ax.text(.04,.82,f"State MSE: {row['raw_state_MSE']:.3f} → {row['aligned_state_MSE']:.3f}\nExact calibration moments;\nindependent Gaussian evaluation",transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax=axes[1,0];rows=k['threshold_scale_dependent_geometry'];ds=[v['delta'] for v in rows]
    ax.loglog(ds,[v['expected_squared_central_difference'] for v in rows],'o-',label='Expected squared finite difference',color='#197f85');ax.loglog(ds,[v['event_boundary_crossing_probability'] for v in rows],'o-',label='Probability of event crossing',color='#af7a2f');ax.set_xlabel('Perturbation scale delta');ax.set_ylabel('Value (log scale)');ax.set_title('C  A threshold has zero derivative almost everywhere');ax.legend(fontsize=9)
    ax.text(.04,.5,'Binary target remains perfectly decodable',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1];row=k['serial_recall_loss'];values=[row['exact_general_moment'],row['Monte_Carlo_iid'],row['correlated_exact'],row['correlated_Monte_Carlo'],row['incorrect_if_independence_assumed']]
    ax.bar(np.arange(5),values,color=['#197f85','#7eafb2','#af7a2f','#d4b27c','#b75e64']);ax.set_xticks(np.arange(5),['IID exact','IID sampled','Correlated\nexact','Correlated\nsampled','Incorrect IID\nassumption']);ax.set_ylabel('Continuous recall MSE');ax.set_title('D  Temporal covariance changes interference')
    save(fig,'L3_6_known_temporal_controls')
    print('Saved two PNG/SVG figure pairs')

if __name__=='__main__':run()
