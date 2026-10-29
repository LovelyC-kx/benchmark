"""Known-answer analytic and spatial counterexamples, independent of learned scores."""
import argparse
import numpy as np
from scipy.fft import dctn,idctn
from scipy.signal import convolve2d
from spatial_data45 import batch,replace_source,render,region_masks,latest_visible,align_observations,basis
from spatial_metrics45 import regional_losses,frequency_losses,psnr_summary,spectral_angle
from bridge45 import dump

def controls():
    out={};a=batch(128,45300000,noise=0);b=batch(128,45310000,noise=0);m=a['masks']
    assert np.array_equal(m['C'].astype(int)+m['H']+m['U_vis'],np.ones((16,16),int))
    out['coverage_partition']={'C_pixels':int(m['C'].sum()),'H_pixels':int(m['H'].sum()),'U_vis_pixels':int(m['U_vis'].sum()),'changed_C_is_subset_of_C':bool(np.all(~m['changed_C']|m['C']))}
    hist=latest_visible(a['x']);current=latest_visible(a['x'][:,-1:]);rh=regional_losses(hist,a['y'],m);rc=regional_losses(current,a['y'],m)
    assert max(rh['C'].max(),rh['H'].max())==0 and rc['H'].mean()>0
    out['exact_current_and_history_reference']={'history_H_MSE':float(rh['H'].mean()),'current_H_MSE':float(rc['H'].mean()),'never_seen_MSE':float(rh['U_vis'].mean())}
    # Known prior floor for each unobserved independent uniform coefficient.
    z=np.zeros((4,4,4))
    for k in range(4):z[k,3,k]=1
    response=render(z)[:,:,8:,8:].astype(float)-.5;floor=float(np.mean(np.sum(response**2,axis=0))/3)
    out['unseen_Bayes_floor']={'analytic_expected_MSE':floor,'finite_scene_MSE':float(rh['U_vis'].mean()),'information_condition':'All U_vis factors independent and absent from every model input.'}
    changed=a['latents'].copy();changed[:,3]=b['latents'][:,3]
    same=batch(128,45300000,noise=0);same['latents']=changed;same['y']=render(changed)
    assert np.array_equal(a['x'],same['x']) and not np.array_equal(a['y'],same['y'])
    out['unseen_input_identical_target_changes']=True
    stale=replace_source(a,b,'U_src');assert np.array_equal(stale['x'][:,-1],a['x'][:,-1]) and np.array_equal(stale['y'],a['y'])
    out['stale_source_pair_same_target_current']=True
    replaced=replace_source(a,b,'H');assert np.array_equal(replaced['x'][:,-1],a['x'][:,-1]) and not np.array_equal(replaced['x'][:,2],a['x'][:,2])
    newz=a['latents'].copy();newz[:,1]=b['latents'][:,1];newy=render(newz)
    assert not np.array_equal(newy,a['y'])
    out['informative_source_requires_recomputed_supervised_target']=True
    current_tile=a['x'][:,5,:4,:8,:8].mean(axis=(1,2,3))
    acc=float(np.mean(np.sign(current_tile-.5)==np.sign(a['latents'][:,0,0])))
    assert acc==1.
    out['classification_can_ignore_history_details']={'current_factor_sign_accuracy':acc,'H_counterfactual_changes_pixel_target_but_not_this_class':bool(np.array_equal(a['latents'][:,0],newz[:,0])),
        'scope':'Exact toy current-content class; not a real semantic classifier or a learned SNN accuracy result.'}
    mot=batch(128,45300000,noise=0,motion=True);aligned=align_observations(mot)
    assert np.array_equal(aligned,a['x'])
    out['periodic_alignment_oracle_exact']=True
    # Pixel-uniform full risk is area-weighted within each scene.
    g=np.random.default_rng(45400);p=g.uniform(0,1,a['y'].shape);rl=regional_losses(p,a['y'],m)
    gap=rl['full']-.5*rl['C']-.25*rl['H']-.25*rl['U_vis'];assert np.max(np.abs(gap))<1e-14
    out['area_weighted_full_risk']={'max_abs':float(np.max(np.abs(gap)))}
    empty=regional_losses(p,a['y'],{'empty':np.zeros((16,16),bool)});assert empty['empty'] is None
    out['empty_regions_are_undefined']=True
    f=frequency_losses(p,a['y'],m);residual=max(np.max(np.abs(f['parseval_residual'])),np.max(np.abs(rl['H']-f['H_low']-f['H_high']-f['H_cross'])))
    assert residual<1e-14 and np.max(np.abs(f['H_cross']))>1e-5
    out['frequency_and_region_commutator']={'identity_max_abs':float(residual),'regional_cross_max_abs':float(np.max(np.abs(f['H_cross'])))}
    per=psnr_summary(np.array([.001,.1]));assert per['mean_scene_PSNR']>per['PSNR_of_mean_MSE']
    out['PSNR_aggregation_order']=per
    zero=psnr_summary(np.array([0.]));assert zero['PSNR_of_mean_MSE'] is None
    out['zero_MSE_infinite_PSNR_is_flagged']=zero
    sam=spectral_angle(np.zeros((1,4,2,2)),np.ones((1,4,2,2)));assert sam['scene_mean_radians']==[None]
    out['SAM_zero_norm_is_undefined']=True
    # Exact same mean/global code, distinct spatial targets. No function of this code distinguishes them.
    image=np.array([[1.,0.],[0.,1.]]);flipped=image[:,::-1]
    assert image.mean()==flipped.mean() and not np.array_equal(image,flipped)
    out['GAP_spatial_ambiguity']={'same_mean':float(image.mean()),'target_MSE':float(np.mean((image-flipped)**2))}
    feat=g.normal(size=(2,8,16,16));perm=g.permutation(256);shuffle=feat.reshape(2,8,-1)[:,:,perm];back=shuffle[:,:,np.argsort(perm)]
    assert np.array_equal(feat.reshape(2,8,-1),back)
    out['permutation_is_coordinate_change_not_information_loss']=True
    tokens=g.normal(size=(8,64,4));shifted=tokens+np.array([1.,2.,3.,4.])[None,None]
    mu=tokens.mean(1,keepdims=True);sd=tokens.std(1,ddof=1,keepdims=True)
    normalized=(tokens-mu)/(sd+1e-6);other=(shifted-shifted.mean(1,keepdims=True))/(shifted.std(1,ddof=1,keepdims=True)+1e-6)
    gap=float(np.max(np.abs(normalized-other)));back=normalized*(sd+1e-6)+mu
    assert gap<1e-12 and np.max(np.abs(tokens-back))<1e-12
    out['spatial_normalization_absolute_content']={'shift_invariance_max_abs':gap,'mean_scale_restore_max_abs':float(np.max(np.abs(tokens-back))),
        'meaning':'A sole normalized code loses additive channel offsets. Saving channel mean/scale restores this map; counts 2C extra scalars. This does not refute normalization as an auxiliary alignment loss.'}
    impulse=np.zeros((16,16));impulse[7,8]=1;response=convolve2d(impulse,np.ones((3,3))/9,mode='same')
    assert np.sum(response[~m['H']]**2)>0
    out['local_state_edit_can_affect_other_regions']={'outside_H_energy':float(np.sum(response[~m['H']]**2))}
    # Bounded repeated noise, exact variance formula; scenes are statistical units.
    signs=g.choice([-1.,1.],size=(2048,3,4,16,16));sigma=.02
    independent=sigma*signs;repeated=np.repeat(independent[:,:1],3,axis=1)
    vi=float(np.mean(independent.mean(1)**2));vr=float(np.mean(repeated.mean(1)**2))
    assert abs(vi/(sigma*sigma/3)-1)<.02 and abs(vr/sigma**2-1)<1e-12
    out['repeated_noise_effective_information']={'independent_mean_MSE':vi,'independent_theory':sigma*sigma/3,'repeated_mean_MSE':vr,'repeated_theory':sigma*sigma}
    # Sufficiently competent hand-specified image-state system, no optimizer training.
    state=np.full_like(a['y'],.5)
    for t in range(6):state=np.where(a['x'][:,t,4:5]>0,a['x'][:,t,:4],state)
    damaged=state.copy();damaged[:,:,m['H']]=.5;delta=(state-damaged)*m['H'][None,None];repaired=damaged+delta
    assert np.max(np.abs(repaired-state))==0
    out['competent_image_recovery_known_answer']={'H_damage_MSE':float(regional_losses(damaged,a['y'],m)['H'].mean()),'H_repaired_MSE':float(regional_losses(repaired,a['y'],m)['H'].mean()),'C_prediction_change_max_abs':float(np.max(np.abs((repaired-state)[:,:,m['C']]))) ,'scope':'Hand-specified last-valid image memory, not learned SNN or energy advantage.'}
    return {'module':'L4.5 known-answer controls','control_families':len(out),'results':out,'new_SNN_training_updates':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=controls();dump(a.output,r);print('Control families',r['control_families'])
