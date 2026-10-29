"""Known answers for access bottlenecks and readout-owned history."""
import argparse
from pathlib import Path
import numpy as np
import torch
from access_operators import aggregate,observability_stack,output_subspaces
from temporal_probes import fit_head
from bridge35 import BenchmarkModel,ModelConfig,StreamSession
from artifacts35 import write_json

def run():
    torch.set_num_threads(1);rng=np.random.default_rng(41001)
    z=rng.normal(size=(4096,1));features=np.stack([z,-z],1)
    cancellation={'mean_max_abs':float(np.abs(aggregate(features,'mean')).max()),
                  'sum_max_abs':float(np.abs(aggregate(features,'sum')).max()),
                  'last_recovers_z_MSE':float(np.mean((-aggregate(features,'last')-z)**2))}
    x=rng.normal(size=(512,3));y=x@np.array([[1.,2.],[-1.,.5],[.2,.7]])+.01*rng.normal(size=(512,2));T=7
    mean=fit_head(x,y,.01);summed=fit_head(T*x,y,.01*T*T)
    a=(x-mean[1])@mean[0]+mean[2];b=(T*x-summed[1])@summed[0]+summed[2]
    gauge={'fixed_T':T,'mean_alpha':.01,'sum_alpha':.01*T*T,'prediction_max_abs':float(np.max(np.abs(a-b)))}
    w=np.array([[0.,1.]]);h=np.c_[z[:,0],rng.normal(size=len(z))]
    target=z[:,0];null=output_subspaces(w);changed=h+np.array([3.,0.])
    unused={'target_decodable_MSE':float(np.mean((h[:,0]-target)**2)),
            'native_output_change_max_abs':float(np.max(np.abs(changed@w.T-h@w.T))),
            'target_shift':3.,'head_null_dimension':null['null_basis'].shape[1]}
    A=np.array([[0.,1.],[0.,0.]]);C=np.array([[1.,0.]]);delta=np.array([0.,1.])
    delayed={'current_output_change':float((C@delta)[0]),'next_output_change':float((C@A@delta)[0]),
             'one_step_observability_rank':int(np.linalg.matrix_rank(observability_stack(A,C,1))),
             'two_step_observability_rank':int(np.linalg.matrix_rank(observability_stack(A,C,2)))}
    cfg=ModelConfig(backbone='mlp',neuron='lif_hard',width=1,depth=1,gain=1.,threshold=.5,
                    gradient_mode='notd',observation='output',readout='mean')
    model=BenchmarkModel((1,),(1,),2,cfg).eval()
    with torch.no_grad():
        model.backbone.layers[0].weight.fill_(1);model.backbone.layers[0].bias.zero_();model.head.weight.fill_(1);model.head.bias.zero_()
        seq=torch.tensor([[[1.],[0.]],[[0.],[0.]]]);pred=model(seq).numpy()
        cell_state=model.backbone.cells[0].state['v'].numpy().copy()
        s=StreamSession(model);s.step(seq[:,0]);snapshot=s.snapshot()
        baseline=s.run(seq[:,1:]).numpy()
        s.restore(snapshot);s.clear('core');core_cleared=s.run(seq[:,1:]).numpy()
        s.restore(snapshot);s.clear('readout');readout_cleared=s.run(seq[:,1:]).numpy()
    notd={'identical_current_input':True,'final_cell_state_equal':bool(np.array_equal(cell_state[0],cell_state[1])),
          'native_mean_outputs':pred.ravel(),'baseline_stream':baseline.ravel(),'clear_core':core_cleared.ravel(),
          'clear_readout':readout_cleared.ravel(),'memory_location':'readout accumulator, not the notd cell'}
    logits=np.array([[20.,0.],[0.,2.],[0.,2.]])
    def softmax(a):
        a=a-a.max(axis=-1,keepdims=True);p=np.exp(a);return p/p.sum(axis=-1,keepdims=True)
    prob_of_mean=softmax(logits.mean(0));mean_prob=softmax(logits).mean(0)
    aggregation={'softmax_mean_logits':prob_of_mean,'mean_softmax_probabilities':mean_prob,
        'prediction_from_mean_logits':int(prob_of_mean.argmax()),'prediction_from_mean_probabilities':int(mean_prob.argmax()),
        'mean_per_step_CE_target0':float(np.mean(np.logaddexp.reduce(logits,axis=1)-logits[:,0])),
        'CE_mean_logits_target0':float(-np.log(prob_of_mean[0]))}
    Z=np.array([0,0,1,1]);Y=np.array([0,1,0,1]);S1=2*Z+Y
    def entropy(*columns):
        _,counts=np.unique(np.stack(columns,1),axis=0,return_counts=True);p=counts/counts.sum()
        return float(-np.sum(p*np.log2(p)))
    redundancy={'exact_uniform_cases':4,'I_previous_later_bits':entropy(Z)+entropy(S1)-entropy(Z,S1),
        'I_target_later_given_previous_bits':entropy(Y,Z)+entropy(S1,Z)-entropy(Z)-entropy(Y,S1,Z),
        'scope':'Exact finite-distribution control, not a high-dimensional empirical MI estimator.'}
    order=np.array([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]])
    permutation={'mean_same_after_frame_permutation':bool(np.array_equal(aggregate(order,'mean')[0],aggregate(order,'mean')[1])),
        'exponential_differs_after_permutation':bool(not np.array_equal(aggregate(order,'exponential')[0],aggregate(order,'exponential')[1])),
        'assumptions':'Shared frame-local encoder; no positional/time cue or upstream cross-time cache. Symmetric mean/sum cannot separate order labels on the same multiset.'}
    hidden_y=np.array([0.,1.]);constant_history=np.tile(np.array([1.,-1.]),(2,1))
    adaptive=hidden_y*constant_history[:,0]+(1-hidden_y)*constant_history[:,1]
    side_channel={'final_feature_history_identical':bool(np.array_equal(constant_history[0],constant_history[1])),
        'adaptive_outputs':adaptive,'target_recovered':bool(np.array_equal((adaptive>0).astype(float),hidden_y)),
        'scope':'Weights explicitly observe an auxiliary target-bearing variable. No information is created from the constant final features; the observer has extra access. This is a control, not a claim of label leakage in MI-TRQR.'}
    return {'module':'L4.1 known access controls','temporal_cancellation':cancellation,'sum_mean_ridge_gauge':gauge,
            'decodable_but_head_unused':unused,'instantaneous_null_not_future_null':delayed,'notd_with_readout_history':notd,
            'logits_probabilities_and_losses':aggregation,'redundancy_can_coexist_with_new_target_information':redundancy,
            'frame_local_symmetric_readout_order_control':permutation,
            'adaptive_weighting_has_its_own_observation_permission':side_channel,
            'new_SNN_optimizer_updates':0,'scope':'Known-answer controls. No learned-model mechanism or causal content-selectivity result is inferred.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    write_json(out,run());print('Known access controls completed')
