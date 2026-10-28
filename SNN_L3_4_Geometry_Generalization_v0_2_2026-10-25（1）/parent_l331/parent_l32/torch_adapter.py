"""PyTorch adapter to the delivered L2 reference.

See evidence for actual CPU runtime checks; this is not a completed benchmark.
"""
from pathlib import Path
import sys
import copy
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent/'parent_l2'))
from snnbench.models import BenchmarkModel, ModelConfig
from snnbench.l2_runtime import StreamSession

class LoadModel(nn.Module):
    def __init__(self,run):
        super().__init__();run.validate();self.run_config=run
        self.encoder=nn.Sequential(nn.Linear(14,64),nn.ReLU(),nn.Linear(64,64))
        cfg=ModelConfig(backbone='mlp',neuron=run.cell,width=run.width,depth=2,
                        beta=.8,threshold=1.,gain=2.,dc_normalize=False,
                        gradient_mode='full',surrogate='atan',sg_scale=2.,
                        detach_reset=True,observation='post',readout='last',
                        normalization='none',push_pull=False,reset_strength=1.)
        self.core=BenchmarkModel((64,),(run.output_dim,),run.data.deadline_steps,cfg,
                                 objective=run.objective)

    def encode(self,x):
        return self.encoder(x)*x[...,13:14]

    def forward(self,x,capture=False):
        return self.core(self.encode(x),capture=capture)

def task_loss(pred,targets,run):
    l=run.data.required_history;k=run.data.k
    if run.objective=='regression':
        error=(pred.reshape(-1,5,4)-targets.reshape(-1,5,4)).square()
        anchor=error[:,[0,4],:k].mean()
        extra=error[:,1:l,:k].mean() if l>1 else anchor.new_zeros(())
    else:
        error=nn.functional.binary_cross_entropy_with_logits(pred,targets,reduction='none')
        anchor=error[:,[0,4]].mean()
        extra=error[:,1:l].mean() if l>1 else anchor.new_zeros(())
    if run.loss_mode=='uniform':
        loss=error[:,list(range(l))+[4],:k].mean() if run.objective=='regression' else error[:,list(range(l))+[4]].mean()
    else:loss=anchor+run.lambda_extra*extra
    return loss,{'anchor':anchor.detach(),'extra':extra.detach()}

@torch.no_grad()
def collect_features(model,x,batch_size=128):
    model.eval();dev=next(model.parameters()).device
    result={};stats={}
    for start in range(0,len(x),batch_size):
        xb=torch.as_tensor(x[start:start+batch_size],device=dev)
        pred=model(xb,capture=True); traces=model.core.traces()
        first=traces['backbone.cells.0'];last=traces['backbone.cells.1']
        values={'prediction':pred,'last_post':last['post'][:,-1],
                'joint_post':torch.cat([first['post'][:,-1],last['post'][:,-1]],dim=1),
                'last_pre':last['pre'][:,-1],'last_output':last['output'][:,-1]}
        for key,value in values.items():result.setdefault(key,[]).append(value.detach().cpu().numpy())
        for layer,trace in traces.items():
            for key in ('base_current','pre','margin'):
                arr=trace[key]
                stats.setdefault(layer+'/'+key+'_mean_per_sample',[]).append(arr.mean((1,2)).cpu().numpy())
                stats.setdefault(layer+'/'+key+'_rms_per_sample',[]).append(arr.square().mean((1,2)).sqrt().cpu().numpy())
            if 'natural_event' in trace:
                stats.setdefault(layer+'/natural_event_fraction_per_sample',[]).append(trace['natural_event'].mean((1,2)).cpu().numpy())
                stats.setdefault(layer+'/natural_event_count_per_sample',[]).append(trace['natural_event'].sum((1,2)).cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}, {k:np.concatenate(v) for k,v in stats.items()}

@torch.no_grad()
def frozen_operations(model,x,cut=12):
    """After step 11, before C at step 12. Disjoint no-op/scope checks."""
    if len(x)==0 or not 0<cut<x.shape[1]:raise ValueError('Cut')
    model.eval();dev=next(model.parameters()).device
    encoded=model.encode(torch.as_tensor(x,device=dev))
    session=StreamSession(model.core)
    for t in range(cut):session.step(encoded[:,t])
    saved=session.snapshot()
    baseline=session.run(encoded[:,cut:]).detach().cpu().numpy()
    session.restore(saved);noop=session.run(encoded[:,cut:]).detach().cpu().numpy()
    session.restore(saved);session.clear('cell',paths=['backbone.cells.0'])
    clear_first=session.run(encoded[:,cut:]).detach().cpu().numpy()
    session.restore(saved);session.clear('core')
    clear_core=session.run(encoded[:,cut:]).detach().cpu().numpy()
    result={'baseline':baseline,'noop':noop,'clear_first_cell':clear_first,'clear_core':clear_core}
    if model.run_config.cell in ('analog_hard_ref','lif_hard','lif_soft'):
        # Fresh predictor/config; never restore an incompatible old snapshot.
        control=copy.deepcopy(model)
        control.core.cfg.reset_strength=0.
        for c in control.core.backbone.cells:c.reset_strength=0.
        result['reset_feedback_disabled_full_sequence']=control(torch.as_tensor(x,device=dev)).cpu().numpy()
    return result
