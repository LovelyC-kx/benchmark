"""Rebuild figures from saved v0.2 evidence only; no new model fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/'figures'
OUT.mkdir(exist_ok=True)
def read(p):return json.loads((BASE/p).read_text())
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.18)
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.facecolor':'white',
    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False})
known=read('evidence/known_controls_v0_2_continuous.json')
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
ax=axes[0,0]
for mean,marker in ((0.,'o'),(.5,'s')):
    rows=[r for r in known['finite_Gaussian'] if abs(r['response_mean'][0]-mean)<1e-10]
    if not rows:continue
    p=[r['p'] for r in rows]
    ax.errorbar(p,[r['direct_risk'] for r in rows],yerr=[2*r['direct_training_MC_standard_error'] for r in rows],
        fmt=marker+'-',label='Direct, zero mean' if mean==0 else 'Direct, nonzero mean',capsize=3)
    ax.plot(p,[r['conditional_exact_law']['predicted_error'] for r in rows],'kx',ms=8,label='Conditional exact law' if mean==0 else None)
ax.set(xscale='log',xlabel='Training labels p',ylabel='Classification error',title='A  Finite-p Gaussian law')
ax.legend(fontsize=8)
ax=axes[0,1]
rows=known['task_support_Hermite']['rows'];x=[r['support'] for r in rows]
ax.plot(x,[r['exact_teacher_projection'] for r in rows],'-',label='Exact')
ax.errorbar(x,[r['empirical_teacher_projection'] for r in rows],
    yerr=[2*r['exact_projection_MC_standard_error'] for r in rows],fmt='o',capsize=3,label='Sample, ±2 MCSE')
ax.axhline(0,color='.3',lw=.8);ax.axvline(known['task_support_Hermite']['critical_equal_support'],ls='--',color='.4')
ax.set(xlabel='Equal-support task size k',ylabel='Teacher-direction projection',title='B  Same code, task-dependent direction')
ax.legend(fontsize=8)
ax=axes[0,2]
ax.bar(['Target energy','Target linear capacity'],[known['task_energy_spectrum']['target_energy_fraction'],known['task_energy_spectrum']['target_capacity']],color=['#8094b0','#198b80'])
ax.set(ylim=(0,1.1),ylabel='Fraction / capacity',title='C  Low energy can hold the target')
ax.text(0,.06,'0.99%',ha='center')
ax=axes[1,0]
rows=known['task_energy_spectrum']['PCR_noise_control']['rows'];x=[r['added_observation_noise_variance'] for r in rows]
ax.plot(x,[r['unrestricted_capacity'][0] for r in rows],'o-',label='Unrestricted')
ax.plot(x,[r['PCR_capacity'][0] for r in rows],'s--',label='Absolute-cutoff PCR')
ax.set(xlabel='Added observation-noise variance',ylabel='Linear capacity',title='D  PCR support changes under noise')
ax.legend(fontsize=8)
ax=axes[1,1];rows=known['task_energy_spectrum']['paired_finite_p_curves']
ax.plot([r['p'] for r in rows],[r['raw_Hebbian'] for r in rows],'o-',label='Raw Hebbian')
ax.plot([r['p'] for r in rows],[r['whitened_Hebbian'] for r in rows],'s-',label='Known whitening')
ax.set(xscale='log',xlabel='Training labels p',ylabel='Classification error',title='E  Capacity and finite learning differ')
ax.legend(fontsize=8)
ax=axes[1,2];p=np.array([1,2,8,32,128])
ax.plot(p,2/p,'o-',label='Identity-code regression')
ax.plot(p,2/(p+2),'s-',label='Oracle scalar gain')
ax.axhline(0,ls='--',color='.3',label='Same-code classification')
ax.set(xscale='log',xlabel='Training labels p',ylabel='MSE / classification error',title='F  Easy sign does not fix value recovery')
ax.legend(fontsize=8)
for ax in axes.flat:style(ax)
fig.suptitle('L3.4 v0.2 | Known-answer controls, not trained-SNN findings',fontsize=15)
save(fig,'L3_4_v0_2_known_controls')

fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
names=['C_last_post_sphere_schedule_0','C_joint_post_sphere_schedule_0','H1_last_post_sphere_schedule_0']
labels=['C final post','C two-layer post','H1 final post']
methods=['identity','spectral_eta_0.1','ridge'];colors=['#8296ac','#138d80','#d18b38']
ax=axes[0];positions=np.arange(3)
for i,(m,col) in enumerate(zip(methods,colors)):
    values=[read(f'evidence/validated_v0_2_thirteen_views/{n}/report.json')['target_groups']['F']['reader_results'][m]['128']['mean_error'] for n in names]
    ax.bar(positions+(i-1)*.24,values,.23,label=['Raw Hebbian','Selected spectral Hebbian','Fixed ridge'][i],color=col)
ax.set(xticks=positions,xticklabels=labels,ylim=(0,.55),ylabel='F-task classification error',title='A  Matched p=128; one frozen SNN')
ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8,loc='lower right')
ax=axes[1]
s=read('evidence/validated_v0_2_thirteen_views/summary.json')
g=[v['groups'][group]['four_statistic'] for v in s['views'] for group in ('F','G','FG')]
order=np.argsort([v['mean_error_reduction'] for v in g])
gain=np.array([g[i]['mean_error_reduction'] for i in order]);lower=np.array([g[i]['simultaneous_Hoeffding_interval'][0] for i in order]);upper=np.array([g[i]['simultaneous_Hoeffding_interval'][1] for i in order])
ax.errorbar(np.arange(len(g)),gain,yerr=np.stack([gain-lower,upper-gain]),fmt='.',ms=4,color='#406b89',alpha=.7,lw=.6)
ax.scatter(np.flatnonzero(lower>0),gain[lower>0],color='#138d80',zorder=3,label='Positive simultaneous lower bound')
ax.axhline(0,color='.3',lw=.8)
ax.set(xlabel='39 correlated view/role combinations',ylabel='Baseline error − selected error',title='B  Keep gains, failures and uncertainty')
ax.legend(fontsize=7)
ax=axes[2]
cont=[read(f'evidence/validated_v0_2_continuous_native/{n}/report.json')['groups']['F']['continuous_learning_curves']['128'] for n in names]
for i,(m,col) in enumerate(zip(['raw_Hebbian','moment_gain_Hebbian','ridge'],colors)):
    ax.bar(positions+(i-1)*.24,[v[m]['MSE_per_target'] for v in cont],.23,color=col,label=['Raw Hebbian','Moment-fitted scalar gain','Fixed ridge'][i])
ax.set(xticks=positions,xticklabels=labels,ylabel='Continuous-target MSE per coordinate',title='C  Value recovery needs scale calibration\nScalar gain uses +2048 moment labels')
ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
for ax in axes:style(ax)
fig.suptitle('L3.4 v0.2 | One-update checkpoint integration; formal benchmark 0/435',fontsize=14)
save(fig,'L3_4_v0_2_frozen_SNN_evidence')
