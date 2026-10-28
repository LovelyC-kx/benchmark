"""Execute L3.5--3.7 known controls and one frozen-checkpoint integration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from followup_diagnostics import *
from run_generalization import serial


def controls():
    r,y=count_matched_timing(256);counts=r.sum(1);timing=(r*np.arange(r.shape[1])[None,:,None]).sum(1)
    f=ridge_fit(counts,y);t=ridge_fit(timing,y)
    rng=np.random.default_rng(360);z=rng.normal(size=(500,2));e=rng.normal(size=(200,2));q=np.array([[0.,-1.],[1.,0.]])
    temporal=temporal_generalization(np.stack([z,z@q],1),np.stack([e,e@q],1),z,e,alpha=1e-8,align=True)
    # Independent target and shared nuisance. Nearly perfect communication
    # prediction can transmit no information about target coordinate zero.
    x=rng.normal(size=(1800,2));receiver=10*x[:,1:2]+.1*rng.normal(size=(1800,1))
    communication=selected_communication(x[:600],receiver[:600],x[600:1000],receiver[600:1000],x[1000:],receiver[1000:],ranks=(0,1))
    return {'kind':'known-answer controls, not SNN findings',
        'carrier':{'every_channel_exactly_one_spike':bool(np.all(counts==1)),
            'count_reader_training_MSE':float(np.mean((ridge_predict(f,counts)-y)**2)),
            'time_reader_training_MSE':float(np.mean((ridge_predict(t,timing)-y)**2)),
            'training_only_demo':True,'zero_jitter_identity':bool(np.array_equal(count_preserving_circular_jitter(r,0),r)),
            'jitter_count_preservation':bool(np.array_equal(count_preserving_circular_jitter(r,4).sum(1),counts)),
            'hard_reset_collision':{'pre':[1.2,2.0],'spike':[1,1],'post':[0.,0.],
                'consequence':'Fixed-threshold pre determines post/spike, while post/spike can lose overshoot.'}},
        'rotating_code':temporal,'nuisance_only_communication':communication,
        'known_operator_target_information':gaussian_information_transfer(np.array([1.,0.]),np.eye(2),np.array([[0.,10.]]),np.array([[.01]]))}


def collect_traces(model,x):
    import torch
    result={}
    with torch.no_grad():
        for start in range(0,len(x),128):
            model(torch.as_tensor(x[start:start+128]),capture=True)
            traces=model.core.traces()
            for name,trace in traces.items():
                for key in ('pre','post','output','base_current','a','syn','previous_a','previous_syn'):
                    if key in trace:result.setdefault(name+'/'+key,[]).append(trace[key].detach().cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def integration(run_directory):
    from bridge import load_model,flatten_contents
    from parent_sampling import fixed_batch
    import torch
    torch.set_num_threads(1);model,run,meta=load_model(run_directory)
    before=hashlib.sha256((Path(run_directory)/'best.pt').read_bytes()).hexdigest()
    pools={name:fixed_batch(run.data,n,split,first,0) for name,n,split,first in [
        ('train',256,'probe_train',4500000),('selection',128,'probe_val',4600000),('evaluation',256,'development_test',4700000)]}
    traces={k:collect_traces(model,b['x']) for k,b in pools.items()};z={k:flatten_contents(b,run.data) for k,b in pools.items()}
    first='backbone.cells.0/';last='backbone.cells.1/';evaluation=traces['evaluation']
    inventory=carrier_inventory({name:evaluation[last+name] for name in ('pre','post','output')})
    times=sorted(set([2,5,8,11,12,run.data.deadline_steps-1]));temporal={}
    for source,columns in [('H1',list(range(4))),('C',list(range(16,20)))]:
        temporal[source]=temporal_generalization(traces['train'][last+'post'][:,times],evaluation[last+'post'][:,times],z['train'][:,columns],z['evaluation'][:,columns],align=True)
    # Match the true same-step inter-layer interface; post is not the
    # transmitted carrier. Prior receiver state affects pre/post separately.
    current=evaluation[last+'base_current'];spike=evaluation[first+'output']
    backbone=model.core.backbone
    linears=[(name,m) for name,m in backbone.named_modules() if isinstance(m,torch.nn.Linear)]
    candidates=[]
    for name,layer in linears:
        w=layer.weight.detach().numpy();b=layer.bias.detach().numpy() if layer.bias is not None else np.zeros(len(w))
        if spike.shape[-1]==w.shape[1] and current.shape[-1]==w.shape[0]:
            for gain in (1.,2.):
                error=float(np.max(np.abs(gain*(spike@w.T+b)-current)))
                candidates.append({'path':name,'gain':gain,'max_error':error})
    operator=min(candidates,key=lambda row:row['max_error'])
    if operator['max_error']>1e-5:raise AssertionError('Could not verify declared same-step inter-layer current')
    t=12;keys=['train','selection','evaluation']
    xs=[traces[k][first+'output'][:,t] for k in keys];ys=[traces[k][last+'base_current'][:,t] for k in keys]
    communication=selected_communication(xs[0],ys[0],xs[1],ys[1],xs[2],ys[2],ranks=(0,1,2,4,8,16,32))
    after=hashlib.sha256((Path(run_directory)/'best.pt').read_bytes()).hexdigest()
    if before!=after:raise AssertionError('Checkpoint mutated')
    return {'kind':'one frozen one-update-checkpoint integration','checkpoint_id':meta['effective_id'],
        'checkpoint_sha256':before,'checkpoint_unchanged':True,'new_SNN_training_runs':0,'audit_test_accessed':False,
        'inventory':inventory,'selected_time_indices':times,'time_generalization':temporal,
        'verified_actual_inter_layer_operator':operator,'spike_to_receiver_current_RRR':communication,
        'scope':'Current mapping equality is code-path validation. Statistical RRR and observational decoding do not establish causal task transmission; layer-4 interventions remain separate.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-directory');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    r={'controls':controls()}
    if a.run_directory:r['integration']=integration(a.run_directory)
    out.write_text(json.dumps(serial(r),indent=2,allow_nan=False));print(json.dumps({'output':str(out),'integration':a.run_directory is not None}))
