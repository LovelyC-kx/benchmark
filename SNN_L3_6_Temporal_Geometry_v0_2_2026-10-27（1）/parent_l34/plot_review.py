"""Standalone scientific calibration figure; no generated illustration."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(review_directory,root):
    root=Path(root);review=Path(review_directory)
    known=json.loads((root/'evidence/known_theory.json').read_text())
    gaussian=json.loads((root/'evidence/gaussian_reference.json').read_text())
    estimator=json.loads((review/'evidence/estimator_review_v0_2_full.json').read_text())
    follow=json.loads((root/'evidence/followup_3_5_to_3_7.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.5),constrained_layout=True)
    ax=axes[0,0]
    e=gaussian['empirical_Hebbian'];p=[r['samples'] for r in e]
    ax.semilogx(p,[r['mean_error'] for r in e],'o-',label='Empirical Hebbian')
    ax.semilogx(p,[r['predicted_error'] for r in gaussian['four_statistic_forecast']],'s--',label='Four-statistic forecast')
    ax.set(title='A  Known Gaussian code (80 responses, 40 latents)',xlabel='New-task labels p',ylabel='Classification error');ax.legend(frameon=False,fontsize=9)
    ax=axes[0,1];rows=known['finite_p'];p=[r['samples'] for r in rows]
    for key,label,style in [('empirical_risk_using_exact_test_population','Empirical weight signs; exact test risk','o-'),('four_statistic_forecast','Four-statistic forecast','s--'),('task_moment_normal_forecast','Task-moment normal approximation','^-')]:
        ax.semilogx(p,[r[key] for r in rows],style,label=label)
    ax.axhline(known['matched_moments']['nonlinear_population_Hebbian_error'],color='grey',lw=1,ls=':',label='Nonlinear population limit')
    ax.set(title='B  Matched second moments; nonlinear H3 code',xlabel='New-task labels p',ylabel='Classification error',ylim=(.06,.54));ax.legend(frameon=False,fontsize=8)
    ax=axes[1,0]
    rows=[r for r in estimator['rows'] if r['case']=='additive']
    for method,label,style in [('legacy_covariance','Legacy FG estimate','o-'),('symmetric_difference','Difference FG estimate','s--')]:
        # Fixed display floor only. Raw values remain unchanged in JSON.
        ax.loglog([r['pairs'] for r in rows],[max(r['methods'][method]['FG_RMSE'],1e-16) for r in rows],style,label=label)
    ax.set(title='C  Additive function: true FG interaction = 0',xlabel='Independent A/B pairs',ylabel='FG estimator RMSE');ax.legend(frameon=False,fontsize=9)
    ax.text(.03,.14,'Structural zero is useful; main-effect variance\ncan be worse (see binary-main control).',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1];control=follow['controls']['rotating_code']
    data=np.c_[np.array(control['train_time_by_test_time_mse']).reshape(-1),np.array(control['orthogonally_aligned_mse']).reshape(-1)]
    im=ax.imshow(data,aspect='auto',cmap='viridis',vmin=0);ax.set_xticks([0,1],['Raw fixed reader','Train-only alignment']);ax.set_yticks(range(4),['t0 -> t0','t0 -> t1','t1 -> t0','t1 -> t1']);ax.set_title('D  Known rotating code: recovery without forgetting')
    for i in range(4):
        for j in range(2):ax.text(j,i,f'{data[i,j]:.3g}',ha='center',va='center',color='white' if data[i,j]<.7 else 'black',fontsize=10)
    fig.colorbar(im,ax=ax,label='Regression MSE',shrink=.8)
    for ax in axes.flat:
        if ax is not axes[1,1]:ax.grid(alpha=.2)
    fig.suptitle('Calibration and failure discrimination — synthetic controls, not SNN benchmark findings',fontsize=13)
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'review_and_generalization_controls.png',dpi=180)
    fig.savefig(folder/'review_and_generalization_controls.svg')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review-directory',required=True);a=p.parse_args();plot(a.review_directory,Path(__file__).resolve().parent)
