"""Add exact task-objective accounting to previously validated frozen views."""
import argparse,json
from pathlib import Path
import numpy as np
from bridge import load_model
from task_objective import requested_objective,whole_task_report
from run_calibrations import serial


def annotate(run_directory,output_directory):
    model,run,_=load_model(run_directory,'cpu')
    folder=Path(output_directory)
    rows=[]
    for report_path in sorted(folder.glob('schedule_*/report.json')):
        report=json.loads(report_path.read_text());array_path=report_path.parent/'analysis_arrays.npz'
        with np.load(array_path) as data:arrays={k:data[k] for k in data.files}
        roles=report['factor_schema']['coordinate_roles']
        obj=requested_objective(run.data,roles,run.loss_mode,run.lambda_extra)
        w=model.core.head.weight.detach().numpy()[obj['native_head_rows']].astype(float)
        b=model.core.head.bias.detach().numpy()[obj['native_head_rows']].astype(float)
        if report['observation']=='joint_post':w=np.concatenate((np.zeros_like(w),w),axis=1)
        lr,er=arrays['lattice_r'],arrays['evaluation_r']
        target=arrays['lattice_z'][:,:,obj['visible_latent_indices']]
        et=arrays['evaluation_z'][:,obj['visible_latent_indices']]
        audit,c=whole_task_report(lr @ w.T+b,target,er @ w.T+b,et,tuple(report['factor_schema']['names']),obj)
        report['original_whole_task_objective']=audit
        report_path.write_text(json.dumps(serial(report),ensure_ascii=False,indent=2))
        arrays['original_whole_task_weighted_error_component_covariances']=c
        np.savez_compressed(array_path,**arrays)
        rows.append({'schedule_index':report['schedule_index'],'natural_objective':audit['natural_objective'],
            'lattice_A_objective':audit['lattice_A_direct_objective'],'weighted_risk_identity_error':audit['weighted_error_anova']['mse_identity_error']})
    if not rows:raise ValueError('No validated schedule reports found')
    summary_path=folder/'summary.json';summary=json.loads(summary_path.read_text())
    summary['whole_task_objective_audit']=rows
    summary_path.write_text(json.dumps(serial(summary),ensure_ascii=False,indent=2))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('output_directory');args=p.parse_args()
    print(json.dumps(annotate(args.run_directory,args.output_directory)))
