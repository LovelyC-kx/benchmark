"""v0.2 review: reuse frozen responses, no SNN forward/training required."""
import argparse,json
from pathlib import Path
import numpy as np
from functional_anova import decompose,power_summary,error_risk_budget,paired_bootstrap
from run_calibrations import serial


def gated_specialist_predictions(contexts,group,reader_name='linear',shift=0):
    """Exactly transport six existing specialist maps to an explicit gate.

    No new labels/parameters are gained. A cyclic wrong-gate control has the
    SAME parameter blocks. Schedule access remains an analysis permission.
    """
    output=[]
    for i,c in enumerate(contexts):
        selected=contexts[(i+shift)%len(contexts)]['readers'][group][reader_name]
        raw_w=selected['coef']/selected['input_scale']
        raw_b=selected['bias']-selected['input_mean']@raw_w
        output.append(c['evaluation_r']@raw_w+raw_b)
    return np.stack(output)


def augment_folder(folder,draws=64):
    root=Path(folder);summaries=[]
    for path in sorted(root.glob('schedule_*/report.json')):
        report=json.loads(path.read_text());names=tuple(report['factor_schema']['names'])
        with np.load(path.parent/'analysis_arrays.npz') as data:arrays={k:data[k] for k in data.files}
        state=decompose(arrays['lattice_r'],names,estimator='symmetric_difference')
        review={'primary_estimator':'symmetric_difference','legacy_estimator':'legacy_covariance',
            'representation':power_summary(state),'readers':{},
            'scope':'Same stored frozen responses. The primary covariance reference and risk use paired A/B, while v0.1 used A-only. Distinct finite estimates of the same population quantities.'}
        arrays['review_primary_state_component_covariances']=state['components']
        for group,gdata in report['target_readers'].items():
            columns=gdata['target_coordinate_indices'];readers={}
            for reader in gdata['fixed_readers']:
                p=arrays[group+'__'+reader+'__lattice_predictions']
                errors=p-arrays['lattice_z'][:,:,columns]
                pf=decompose(p,names,estimator='symmetric_difference')
                ef,budget=error_risk_budget(errors,names,(group,),estimator='symmetric_difference')
                old=gdata['fixed_readers'][reader]['error_anova_risk_budget']
                readers[reader]={'prediction_anova':power_summary(pf,(group,)),
                    'error_risk':budget,'uncertainty':paired_bootstrap(errors,names,(group,),draws=draws,estimator='symmetric_difference'),
                    'legacy_A_only_mse':old['direct_mse'],
                    'primary_pooled_A_B_mse':budget['direct_mse'],
                    'independent_repeat_variability':gdata['fixed_readers'][reader]['independent_conditioned_repeat_risk']['conditional_variability'],
                    'non_target_variability_estimator_difference':budget['target_contract']['non_target_conditional_variability_power']-old['target_contract']['non_target_conditional_variability_power']}
                arrays['review__'+group+'__'+reader+'__error_components']=ef['components']
            review['readers'][group]=readers
        if 'original_whole_task_objective' in report:
            obj=report['original_whole_task_objective']['objective'];selected=obj['visible_latent_indices']
            pred=np.empty((len(arrays['lattice_r']),len(arrays['lattice_r'][0]),len(selected)))
            for k,column in enumerate(selected):
                for group,gdata in report['target_readers'].items():
                    columns=gdata['target_coordinate_indices']
                    if column in columns:
                        pred[:,:,k]=arrays[group+'__original_task_head__lattice_predictions'][:,:,columns.index(column)]
                        break
            error=(pred-arrays['lattice_z'][:,:,selected])*np.sqrt(len(selected)*np.array(obj['coordinate_loss_weights']))
            _,review['whole_task_weighted_error_risk']=error_risk_budget(error,names,('F','G'),estimator='symmetric_difference')
        report['review_v0_2']=review
        path.write_text(json.dumps(serial(report),ensure_ascii=False,indent=2))
        np.savez_compressed(path.parent/'analysis_arrays.npz',**arrays)
        summaries.append({'schedule_index':report['schedule_index'],'primary_state_closure':state['covariance_closure_max_error'],
            'max_primary_risk_closure':max(abs(r['error_risk']['mse_identity_error']) for g in review['readers'].values() for r in g.values())})
    if not summaries:raise ValueError('Validated frozen-response reports required')
    sp=root/'summary.json';s=json.loads(sp.read_text());s['review_v0_2']=summaries
    sp.write_text(json.dumps(serial(s),ensure_ascii=False,indent=2));return summaries


def review_stored_gates(folder):
    root=Path(folder);contexts=[]
    for path in sorted(root.glob('schedule_*/report.json')):
        r=json.loads(path.read_text());a=np.load(path.parent/'analysis_arrays.npz')
        context={'schedule_index':r['schedule_index'],'evaluation_r':a['evaluation_r'],
            'evaluation_y':a['evaluation_z'],'readers':{},'groups':{}}
        for group,data in r['target_readers'].items():
            prefix='probe__'+group+'__linear__'
            fit={key:a[prefix+key] for key in ('coef','bias','input_mean')}
            fit['input_scale']=data['learning_curves']['linear'][-1]['fit']['input_scale']
            context['readers'][group]={'linear':fit}
            context['groups'][group]=data['target_coordinate_indices']
        contexts.append(context)
    if len(contexts)!=6:raise ValueError('All six stored schedule views required')
    contexts.sort(key=lambda c:c['schedule_index']);result={}
    if any(not np.array_equal(contexts[0]['evaluation_y'],c['evaluation_y']) for c in contexts[1:]):raise ValueError('Unpaired schedule targets')
    for group,columns in contexts[0]['groups'].items():
        p=gated_specialist_predictions(contexts,group);wrong=gated_specialist_predictions(contexts,group,shift=1)
        y=np.stack([c['evaluation_y'][:,columns] for c in contexts])
        result[group]={'known_gate_mse':float(np.mean((p-y)**2)),
            'wrong_gate_mse':float(np.mean((wrong-y)**2)),
            'same_six_parameter_blocks':True,'new_fits':0,
            'scope':'Explicit context-dependent slopes from the existing six linear specialist maps; access to schedule is analysis-only.'}
    sp=root/'summary.json';s=json.loads(sp.read_text());s['review_v0_2_explicit_gate']=result
    sp.write_text(json.dumps(serial(s),indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('analysis_directory');p.add_argument('--draws',type=int,default=64);args=p.parse_args()
    print(json.dumps(augment_folder(args.analysis_directory,args.draws)))
