"""Local-only adapter/validation boundary; no automatic dataset download."""
import argparse,json
from pathlib import Path
import numpy as np

def validate_groups(train,val,test):
    groups=[set(map(str,v)) for v in (train,val,test)]
    if any(not g for g in groups):raise ValueError('Nonempty explicit source/ROI groups')
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:raise ValueError('Original source/ROI leakage across splits')
    return {'source_groups_train':len(groups[0]),'source_groups_val':len(groups[1]),'source_groups_test':len(groups[2]),'disjoint':True}

def from_utilise_sample(sample,*,target_index,mode='causal',band_indices=(0,1,2,3),direct_acquisition=False,compatible_with_target=None):
    """UT-style masks use 1=occluded; output model x uses 1=visible.

    Real temporal observations are NOT assumed to show the same current content.
    compatible_with_target is analysis-only and must be justified externally.
    """
    if mode not in ('causal','offline'):raise ValueError('Explicit causal/offline protocol')
    required={'x','y','masks','position_days','source_id','target_is_verified_clean'}
    if not required<=set(sample):raise ValueError('Need '+str(sorted(required-set(sample))))
    if sample['target_is_verified_clean'] is not True:raise ValueError('Quantitative pixel evaluation requires a verified clean target; real cloudy references need a separate validity protocol')
    if not isinstance(sample['source_id'],str) or not sample['source_id']:raise ValueError('Original source/ROI identity')
    x=np.asarray(sample['x'],float);y=np.asarray(sample['y'],float);mask=np.asarray(sample['masks'],float);days=np.asarray(sample['position_days'],float)
    if x.ndim!=4 or y.shape!=x.shape or mask.shape!=(len(x),1,*x.shape[-2:]) or days.shape!=(len(x),):raise ValueError('TCHW / T1HW / T dimensions')
    if any(not np.isfinite(v).all() for v in (x,y,mask,days)) or np.any(np.diff(days)<0):raise ValueError('Finite ordered times and values')
    if not np.isin(mask,[0.,1.]).all():raise ValueError('Explicit binary occlusion mask')
    if len(set(band_indices))!=len(band_indices) or min(band_indices)<0 or max(band_indices)>=x.shape[1]:raise ValueError('Explicit unique band indices')
    if isinstance(target_index,bool) or not isinstance(target_index,int) or not 0<=target_index<len(x):raise ValueError('Target index')
    if 'availability_days' in sample:availability=np.asarray(sample['availability_days'],float)
    elif direct_acquisition:availability=days.copy()
    else:raise ValueError('Supply availability of derived observations, or explicitly assert direct acquisitions')
    if availability.shape!=days.shape or not np.isfinite(availability).all() or np.any(availability<days):raise ValueError('Acquisition/derived availability contract')
    deadline=float(days[target_index]);chosen=np.arange(target_index+1) if mode=='causal' else np.arange(len(x))
    if mode=='causal' and np.any(availability[chosen]>deadline):raise ValueError('Future-supported preprocessing unavailable at deadline')
    visible=1-mask[chosen];frames=x[chosen][:,band_indices];payload=np.concatenate([frames,visible],axis=1).astype('float32')
    current_visible=(1-mask[target_index,0]).astype(bool);past_seen=np.any(visible[:,0]>0,axis=0)
    analysis={'C_visible':current_visible,'past_or_current_seen_without_semantic_equivalence':past_seen,'never_visible_in_allowed_input':~past_seen,
        'H_validity':'Unknown without target-compatible content and alignment evidence.'}
    if compatible_with_target is not None:
        c=np.asarray(compatible_with_target,bool)
        if c.shape!=(len(x),*x.shape[-2:]):raise ValueError('Explicit per-frame target-compatible mask')
        ever=np.any((visible[:,0]>0)&c[chosen],axis=0)
        analysis['H_target_compatible']=(~current_visible)&ever
        analysis['U_vis_target_unobserved']=(~current_visible)&(~ever)
        analysis['compatibility_is_external_analysis_oracle']=True
    return {'x':payload,'y':y[target_index,band_indices].astype('float32'),'position_days':days[chosen].copy(),
        'source_id':sample['source_id'],'analysis':analysis,'contract':{'mode':mode,'target_index':target_index,'target_time':deadline,
        'selected_indices':chosen.tolist(),'model_fields':['x'],'position_days_returned_but_not_consumed_by_current_reference':True,
        'band_indices':list(band_indices),'mask_semantics':'input visible1, original cloud occluded1','future_inputs_allowed':mode=='offline'}}

def audit_npz(path,target_index,source_id,mode='causal',verified_clean_target=False):
    with np.load(path,allow_pickle=False) as z:
        sample={k:z[k] for k in ('x','y','masks','position_days','availability_days') if k in z.files}
    sample['source_id']=source_id;sample['target_is_verified_clean']=bool(verified_clean_target)
    out=from_utilise_sample(sample,target_index=target_index,mode=mode)
    return {'x_shape':list(out['x'].shape),'y_shape':list(out['y'].shape),'source_id':source_id,'contract':out['contract']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True);p.add_argument('--target-index',required=True,type=int);p.add_argument('--source-id',required=True);p.add_argument('--mode',choices=('causal','offline'),default='causal');p.add_argument('--verified-clean-target',action='store_true');a=p.parse_args()
    print(json.dumps(audit_npz(a.sample,a.target_index,a.source_id,a.mode,a.verified_clean_target),ensure_ascii=False,indent=2))
