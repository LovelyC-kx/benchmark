"""Four-cell risk contrasts and output interactions are distinct estimands."""
import numpy as np

def finite(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all():raise ValueError('Finite array required')
    return x

def four_losses(a,d,p,r,damage_floor=.01):
    a,d,p,r=map(finite,(a,d,p,r))
    if a.ndim!=1 or not len(a) or not a.shape==d.shape==p.shape==r.shape:raise ValueError('Paired scalar scene losses')
    damage=d-a;rescue=d-r;patch_only=a-p;interaction=rescue-patch_only
    denominator=float(damage.mean());eligible=damage_floor is not None and denominator>damage_floor
    return {'R00':float(a.mean()),'R10':float(d.mean()),'R01':float(p.mean()),'R11':float(r.mean()),
        'damage':denominator,'rescue':float(rescue.mean()),'patch_only_gain':float(patch_only.mean()),
        'risk_interaction':float(interaction.mean()),'recovery_ratio':float(rescue.mean()/denominator) if eligible else None,
        'ratio_defined':bool(eligible),'damage_floor':damage_floor,'ratio_clipped':False}

def output_decomposition(a,d,p,r,y):
    a,d,p,r,y=map(finite,(a,d,p,r,y))
    if a.ndim!=2 or not a.shape==d.shape==p.shape==r.shape==y.shape:raise ValueError('Same scene/output coordinates')
    e=a-y;delta=d-a;b0=p-a;b1=r-d;omega=b1-b0
    risk=np.mean((d-y)**2-(r-y)**2-(a-y)**2+(p-y)**2)
    alignment=-2*np.mean(delta*b1);response=2*np.mean(e*(b0-b1));energy=np.mean(b0*b0-b1*b1)
    old_fidelity=float(np.mean((d-a)**2));new_fidelity=float(np.mean((r-a)**2))
    return {'output_interaction_MSE':float(np.mean(omega**2)),'output_interaction_max_abs':float(np.max(np.abs(omega))),
        'risk_interaction':float(risk),'damage_patch_alignment':float(alignment),'baseline_error_response':float(response),
        'patch_energy_difference':float(energy),'risk_identity_absolute_residual':float(abs(risk-alignment-response-energy)),
        'damaged_clean_prediction_MSE':old_fidelity,'repaired_clean_prediction_MSE':new_fidelity,
        'prediction_recovery_ratio':float(1-new_fidelity/old_fidelity) if old_fidelity>1e-8 else None,
        'scope':'Output interaction measures context dependence of the declared patch. Positive squared-risk interaction can occur with zero output interaction in an affine model.'}

def bounded_interval(values,family_size,half_width=1.,delta=.05):
    v=finite(values)
    if v.ndim!=1 or not len(v) or np.max(np.abs(v))>half_width+1e-12:raise ValueError('Independent bounded scene contrasts')
    if family_size<1 or not 0<delta<1:raise ValueError('Declared positive family and delta')
    radius=half_width*np.sqrt(2*np.log(2*family_size/delta)/len(v));mean=float(v.mean())
    return {'mean':mean,'interval':[float(mean-radius),float(mean+radius)],'family_size':int(family_size),
        'independent_scene_pairs':len(v),'half_width':float(half_width),'delta':delta}

def norm_match(direction,reference):
    v,r=map(finite,(direction,reference))
    if v.ndim!=2 or v.shape!=r.shape:raise ValueError('Same scene/feature coordinates')
    nv=np.linalg.norm(v,axis=1,keepdims=True);nr=np.linalg.norm(r,axis=1,keepdims=True)
    usable=(nv[:,0]>1e-14)|(nr[:,0]<=1e-14)
    result=np.divide(v,nv,out=np.zeros_like(v),where=nv>1e-14)*nr
    return result,{'unmatchable_nonzero_reference_scenes':int(np.sum(~usable)),
        'maximum_norm_gap_on_matchable_scenes':float(np.max(np.abs(np.linalg.norm(result[usable],axis=1)-nr[usable,0]),initial=0))}

def random_dose(reference,seed):
    return norm_match(np.random.default_rng(seed).normal(size=reference.shape),reference)[0]

def mobius_outputs(outputs):
    if set(outputs)!={0,1,2,3}:raise ValueError('All two-owner subsets required')
    a,b,c,d=(finite(outputs[k]) for k in range(4))
    interaction=d-b-c+a
    reconstructed=a+(b-a)+(c-a)+interaction
    return {'joint_output_interaction_MSE':float(np.mean(interaction**2)),
        'joint_output_interaction_max_abs':float(np.max(np.abs(interaction))),
        'reconstruction_max_abs':float(np.max(np.abs(reconstructed-d))),
        'scope':'Exact two-owner factorial contrast; not unique percentages of causal responsibility.'}
