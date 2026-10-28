"""Preserve the parent's anchored/uniform task objective exactly."""
import numpy as np
from functional_anova import error_risk_budget


def requested_objective(cfg, coordinate_roles, loss_mode, lambda_extra):
    if loss_mode not in ('anchored','uniform') or not np.isfinite(lambda_extra) or lambda_extra<0:
        raise ValueError('Known task loss and nonnegative auxiliary weight required')
    selected=[i for i,r in enumerate(coordinate_roles) if r['requested']]
    rows=[coordinate_roles[i] for i in selected]
    if not rows:raise ValueError('Requested task coordinates required')
    if loss_mode=='uniform':weights=np.full(len(rows),1/len(rows))
    else:
        weights=np.array([1/(2*cfg.k) if r['source'] in ('H1','C') else lambda_extra/((cfg.required_history-1)*cfg.k) for r in rows])
    return {'visible_latent_indices':np.array(selected,int),
            'native_head_rows':np.array([r['source_id']*4+r['coordinate'] for r in rows],int),
            'coordinate_loss_weights':weights,'weight_sum':float(weights.sum()),
            'loss_mode':loss_mode,'lambda_extra':float(lambda_extra)}


def whole_task_report(lattice_prediction,lattice_target,natural_prediction,natural_target,names,objective):
    """Scale errors so the generic per-output MSE equals the real objective.

    sqrt(q*w) preserves sum_j w_j error_j² despite the generic 1/q.
    F+G contain all requested target factors; U/E are unrequested/measurement.
    Target-only ERROR includes target-dependent mistakes; it is not usefulness.
    """
    p,y=np.asarray(lattice_prediction,float),np.asarray(lattice_target,float)
    weights=objective['coordinate_loss_weights'];q=len(weights)
    if p.shape!=y.shape or p.ndim!=3 or p.shape[-1]!=q:raise ValueError('Matching requested prediction/target lattice required')
    scaled=(p-y)*np.sqrt(q*weights)
    fit,budget=error_risk_budget(scaled,names,('F','G'))
    natural=np.sum((np.asarray(natural_prediction)-np.asarray(natural_target))**2*weights,axis=-1)
    direct=float(np.mean(np.sum((p[-1]-y[-1])**2*weights,axis=-1)))
    if abs(direct-budget['direct_mse'])>1e-10:raise AssertionError('Weighted objective transport failed')
    return {'objective':objective,'natural_objective':float(natural.mean()),
            'lattice_A_direct_objective':direct,'weighted_error_anova':budget,
            'note':'Exact original anchored/uniform regression objective; separate F/G means must not simply be added. Target-only components here are ERROR, not useful signal.'},fit['components']
