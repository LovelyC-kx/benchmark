"""Known-answer examples distinguish restoration, risk geometry and mechanism."""
import argparse
from pathlib import Path
import numpy as np
import torch
from bridge44 import write_json,BenchmarkModel,ModelConfig
from recovery_math44 import four_losses,output_decomposition,norm_match,mobius_outputs,bounded_interval
from recovery_runtime44 import scheduled_run,Edit,additive_edit,trace_site

def run():
    torch.set_num_threads(1);out={}
    a=np.zeros((4,1));d=np.ones_like(a);p=-np.ones_like(a);r=np.zeros_like(a)
    out['affine_risk_interaction_without_output_interaction']=output_decomposition(a,d,p,r,a)
    out['zero_damage_ratio']=four_losses(np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4))
    out['uncropped_recovery_ratio']=four_losses(np.full(4,.2),np.full(4,.8),np.full(4,.2),np.full(4,.1))
    for name,fn in [('AND',lambda x,y:x*y),('OR',lambda x,y:max(x,y)),('XOR',lambda x,y:x!=y)]:
        values={k:np.array([[float(fn(bool(k&1),bool(k&2)))]]) for k in range(4)}
        out[name+'_owner_interaction']={**mobius_outputs(values),'outputs':[float(values[k][0,0]) for k in range(4)]}
    zero=np.zeros((3,2));reference=np.ones_like(zero);_,budget=norm_match(zero,reference)
    out['zero_direction_matching_failure']=budget
    # Actual hard-reset function at four finite input points, not a surrogate derivative.
    pre=np.array([.999,1.001,.997,.999]);post=pre*(pre<1)
    omega=post[3]-post[1]-post[2]+post[0]
    out['finite_event_boundary']={'natural_output_interaction':float(omega),'all_events_fixed_off_output_interaction':float(pre[3]-pre[1]-pre[2]+pre[0])}
    # Independent F and G: any deterministic code/patch of G alone lacks F.
    F=np.array([-1,-1,1,1]);G=np.array([-1,1,-1,1]);prediction=np.array([F[G==g].mean() for g in G])
    out['external_information_permission']={'MSE_from_surviving_G':float(np.mean((F-prediction)**2)),'MSE_after_clean_F_patch':0.,'autonomous_recovery_claim':False}
    b=np.arange(24).reshape(8,3)/10;aa=np.zeros_like(b);dd=np.ones_like(b);pp=aa+b;rr=dd+b
    out['additive_owner_factorial']=mobius_outputs({0:aa,1:aa+b,2:aa+2*b,3:aa+3*b})
    out['same_output_can_have_wrong_content']={'clean_prediction':1,'wrong_donor_prediction':-1,'clean_label_error':1,'donor_label_error':0}
    stable=np.ones((6,2));out['time_invariant_signal_wrong_time_patch']={'correct_time_patch_max_abs':0.,'wrong_time_patch_max_abs':float(np.max(np.abs(stable-stable.copy())))}
    # A declared, competent subthreshold LIF reference; manually set, not trained.
    torch.manual_seed(44044);cfg=ModelConfig(neuron='lif_hard',width=2,depth=1,gain=1.,beta=.8,observation='post',readout='last')
    model=BenchmarkModel((2,),(2,),4,cfg,objective='regression').eval()
    with torch.no_grad():
        model.backbone.layers[0].weight.copy_(torch.eye(2));model.backbone.layers[0].bias.zero_()
        model.head.weight.copy_(torch.eye(2)/(.8**3));model.head.bias.zero_()
    x=np.zeros((128,4,2),np.float32);x[:,0]=np.random.default_rng(44044).uniform(-.4,.4,size=(128,2))
    clean=scheduled_run(model,x);erase=Edit('backbone.cells.0',1,lambda h,start:np.c_[np.zeros(len(h)),h[:,1]],'erase')
    damaged=scheduled_run(model,x,[erase]);delta=trace_site(clean,0,2)-trace_site(damaged,0,2)
    repaired=scheduled_run(model,x,[erase,additive_edit(0,2,delta)])
    wrong=scheduled_run(model,x,[erase,additive_edit(0,2,-delta)])
    out['competent_LIF_known_answer']={'clean_target_MSE':float(np.mean((clean['prediction']-x[:,0])**2)),
        'damaged_F_MSE':float(np.mean((damaged['prediction'][:,0]-x[:,0,0])**2)),
        'repaired_F_MSE':float(np.mean((repaired['prediction'][:,0]-x[:,0,0])**2)),
        'reverse_patch_F_MSE':float(np.mean((wrong['prediction'][:,0]-x[:,0,0])**2)),
        'cotarget_prediction_change_max_abs':float(np.max(np.abs(repaired['prediction'][:,1]-clean['prediction'][:,1]))),
        'natural_spike_count':int(sum(v.sum() for k,v in clean['traces'].items() if k.endswith('/emit_event'))),
        'scope':'Hand-calibrated subthreshold LIF reference; no optimizer training and no spike-coding advantage claim.'}
    z=np.array([-.5,.5,-.5,.5]);out['bounded_scene_interval']=bounded_interval(z,10)
    return {'module':'L4.4 known-answer controls','control_families':len(out),'results':out,'new_SNN_training':0,
        'scope':'Analytic, logical and manually calibrated reference controls; no architecture or formal task-performance conclusion.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=run();write_json(a.output,r);print('Known control families',r['control_families'])
