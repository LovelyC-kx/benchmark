"""Event-stratum finite geometry and dose curves; post-main diagnostic analysis."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from bridge44 import ROOT,RUN_DIRECTORY,load_model,fixed_batch,load_tree,write_json,save_tree
from recovery_runtime44 import scheduled_run,trace_site,damage_edit,additive_edit

ALPHAS=(-1.,0.,.25,.5,1.,1.5)
def agree(runs,time):
    result=np.ones(len(runs[0]['prediction']),dtype=bool)
    for path in ('backbone.cells.0','backbone.cells.1'):
        a=runs[0]['traces'][path+'/emit_event'][:,time:]
        for r in runs[1:]:result &= np.all(a==r['traces'][path+'/emit_event'][:,time:],axis=(1,2))
    return result

def run(output,n=256):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);f=json.loads((ROOT/'evidence/fitted/freeze_contract.json').read_text());bank=load_tree(ROOT/'evidence/fitted/frozen_bank')
    model,cfg,_=load_model(RUN_DIRECTORY);a=fixed_batch(cfg.data,n,'development_test',f['evaluation_pools']['confirmation']['recipient'],0);clean=scheduled_run(model,a['x'])
    columns=list(range(4))+list(range(16,20));y=a['latents'][:,[0,4]].reshape(n,8);rows={};saved={}
    for c in f['recovery_contexts']:
        if c['time']!=12 or c['patch_time'] not in (13,14):continue
        de=damage_edit(c,bank['damage_erasers'][c['key']]);d=scheduled_run(model,a['x'],[de]);h=trace_site(clean,c['layer'],c['patch_time']);dh=trace_site(d,c['layer'],c['patch_time'])
        delta=(h-dh)@(np.eye(32)-bank['patch_erasers'][c['patch_key']]['matrix']);runs={0.:d}
        for alpha in ALPHAS:
            if alpha!=0:runs[alpha]=scheduled_run(model,a['x'],[de,additive_edit(c['layer'],c['patch_time'],delta*alpha)])
        mask=agree([runs[0.],runs[1.]],c['patch_time']);pred0=runs[0.]['prediction'][:,columns].astype(float);b=runs[1.]['prediction'][:,columns].astype(float)-pred0;curve=[]
        e=pred0-y
        for alpha in ALPHAS:
            p=runs[alpha]['prediction'][:,columns].astype(float);linear=pred0+alpha*b;error=np.max(np.abs(p-linear),axis=1)
            certified=mask if 0<=alpha<=1 else np.zeros_like(mask)
            scale=np.maximum(1.,np.max(np.abs(p),axis=1)+np.max(np.abs(linear),axis=1))
            if np.any(certified):assert np.max(error[certified]/scale[certified])<2e-6
            mse=np.mean((p-y)**2,axis=1);quadratic=np.mean(e*e+2*alpha*e*b+alpha*alpha*b*b,axis=1)
            curve.append({'alpha':alpha,'native_H1_C_MSE':float(mse.mean()),'same_events_as_zero_fraction':float(agree([runs[0.],runs[alpha]],c['patch_time']).mean()),
                'endpoint_certified_scene_count':int(certified.sum()),'certified_output_affine_max_abs':float(error[certified].max(initial=0)),
                'certified_MSE_quadratic_max_abs':float(np.max(np.abs(mse[certified]-quadratic[certified]),initial=0)),
                'noncertified_output_affine_mean_squared_error':float(np.mean((p[~mask]-linear[~mask])**2)) if np.any(~mask) else None})
            saved[c['key44']+'::'+str(alpha)]=p[:16]
        # Four-group output interaction on naturally identical event-pattern vertices.
        p=scheduled_run(model,a['x'],[additive_edit(c['layer'],c['patch_time'],delta)]);r=runs[1.]
        same=agree([clean,d,p,r],c['patch_time']);omega=r['prediction'][:,columns].astype(float)-d['prediction'][:,columns]-p['prediction'][:,columns]+clean['prediction'][:,columns]
        scale=np.maximum(1.,sum(np.max(np.abs(v['prediction'][:,columns]),axis=1) for v in (clean,d,p,r)))
        if np.any(same):assert np.max(np.max(np.abs(omega[same]),axis=1)/scale[same])<2e-6
        rows[c['key44']]={'endpoint_same_pattern_scenes':int(mask.sum()),'dose_curve':curve,
            'four_vertex_same_pattern_scenes':int(same.sum()),'same_pattern_output_interaction_max_abs':float(np.max(np.abs(omega[same]),initial=0)),
            'different_pattern_output_interaction_MSE':float(np.mean(omega[~same]**2)) if np.any(~same) else None}
    save_tree(out/'prediction_prefix',saved)
    record={'module':'L4.4 event-conditioned recovery geometry','confirmation_subset_scene_pairs':n,'contexts':len(rows),'alpha_grid':ALPHAS,'dose_conditions':len(rows)*len(ALPHAS),'results':rows,
        'analysis_added_after_main_development':True,'no_method_or_alpha_selected_using_these_outcomes':True,
        'scope':'Post-main diagnostic on first256 confirmation scenes. Fixed weights, linear MLP operations, no normalization, fixed-beta hard-LIF. Equal endpoint event patterns certify the intervening convex alpha interval in exact arithmetic; FP32 checked separately. No tanh/attention/general-SNN theorem is claimed.'}
    write_json(out/'report.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=run(a.output);print('Event-geometry dose conditions',r['dose_conditions'])
