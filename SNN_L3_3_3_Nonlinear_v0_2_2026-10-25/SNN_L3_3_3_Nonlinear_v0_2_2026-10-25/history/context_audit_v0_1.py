"""Presentation-order conditioning; no cross-time or hidden-state intervention."""
import numpy as np
from decoding import fit_probe, predict, mse, probe_metadata


def audit_contexts(contexts):
    """Contexts share base scene IDs but train/selection/evaluation are disjoint.

    Shared and specialized systems use the same total presented training rows:
    six schedules x n scene IDs. Rows from one latent scene are correlated;
    report unique scenes separately and never bootstrap them as iid rows.
    """
    if len(contexts) != 6 or sorted(row['schedule_index'] for row in contexts) != list(range(6)):
        raise ValueError('All six paired schedules required')
    contexts = sorted(contexts,key=lambda row:row['schedule_index'])
    for name in ('train_y','select_y','evaluation_y'):
        if any(not np.array_equal(contexts[0][name], row[name]) for row in contexts[1:]):
            raise ValueError('Schedule views must use paired latent targets')
    groups = contexts[0]['target_slices']
    output = {}
    for group, columns in groups.items():
        rows = {}
        for reader_name in contexts[0]['readers'][group]:
            transfer = np.array([[mse(predict(source['readers'][group][reader_name], destination['evaluation_r']), destination['evaluation_y'][:, columns]) for destination in contexts] for source in contexts])
            rows[reader_name] = {'source_schedule_by_evaluation_schedule_mse':transfer,
                'diagonal_same_schedule_mse':np.diag(transfer),
                'mean_off_diagonal_minus_diagonal_mse':float((transfer.sum()-np.trace(transfer))/30-np.trace(transfer)/6),
                'scope':'Fixed trained reader transferred across schedules. Loss can reflect mapping mismatch; not evidence of lost information.'}
        shared = {}
        train_rows = sum(len(c['train_r']) for c in contexts)
        for extra_context in (False,True):
            def combine(key):
                values=[]
                for i,c in enumerate(contexts):
                    r=c[key]
                    if extra_context:r=np.concatenate((r,np.tile(np.eye(6)[i],(len(r),1))),axis=1)
                    values.append(r)
                return np.concatenate(values)
            tx, sx, ex = [combine(key) for key in ('train_r','select_r','evaluation_r')]
            ty=np.concatenate([c['train_y'][:,columns] for c in contexts])
            sy=np.concatenate([c['select_y'][:,columns] for c in contexts])
            ey=np.concatenate([c['evaluation_y'][:,columns] for c in contexts])
            shared_rows={}
            for family in ('linear','quadratic'):
                fit=fit_probe(tx,ty,sx,sy,family=family)
                pe=predict(fit,ex)
                # Aggregate schedules WITHIN an independent evaluation scene.
                per_scene=np.mean((pe-ey).reshape(6,len(contexts[0]['evaluation_r']),-1)**2,axis=(0,2))
                shared_rows[family]={'fit':probe_metadata(fit),'evaluation_mse':mse(pe,ey),
                    'per_independent_scene_schedule_averaged_mse':per_scene,
                    'extra_analysis_schedule_one_hot':extra_context,
                    'permission':'observation plus analysis-only schedule identity' if extra_context else 'original observation only'}
            shared['with_extra_schedule' if extra_context else 'legal_observation_only']=shared_rows
        output[group]={'fixed_reader_transfer':rows,'shared_readers':shared,
            'total_training_presentations':train_rows,'unique_latent_training_scenes':len(contexts[0]['train_r']),
            'specialist_system_head_count':6,'shared_system_head_count':1,
            'specialists_total_parameter_counts':{key:sum(c['readers'][group][key]['parameter_count'] for c in contexts) for key in contexts[0]['readers'][group]},
            'note':'Paired scene identities are not extra independent data. Extra context is a diagnostic permission. No schedule chosen from evaluation performance.'}
    return output
