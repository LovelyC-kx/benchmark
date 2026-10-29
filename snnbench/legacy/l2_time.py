"""Explicit physical-time reference, separated from legacy index-time networks.

Held input: dv/dt=(-v+gain*drive)/tau, exact subthreshold interval update.
Event input: v decays during interval, then receives gain*event_amount.
Thresholds are checked at interval ends: this is not exact continuous spike-time
integration and is not a reproduction of an official physical neuron model.
"""
import torch


def timed_reference(values,dt,*,tau=.3,gain=1.,threshold=1.,reset='none',v0=None):
    if values.ndim!=3:raise ValueError('values must have [B,T,C] shape')
    if tau<=0 or gain<0 or threshold<=0:raise ValueError('time/reference parameters')
    dt=torch.as_tensor(dt,device=values.device,dtype=values.dtype)
    if dt.ndim!=1 or len(dt)!=values.shape[1] or not torch.all(torch.isfinite(dt)&(dt>0)):
        raise ValueError('dt must contain positive finite intervals, one per step')
    if reset not in ('none','hard','soft'):raise ValueError('reset')
    return _run(values,dt,tau,gain,threshold,reset,v0,held=True)


def timed_events(values,dt,*,tau=.3,gain=1.,threshold=1.,reset='none',v0=None):
    if values.ndim!=3:raise ValueError('values must have [B,T,C] shape')
    if tau<=0 or gain<0 or threshold<=0:raise ValueError('time/reference parameters')
    dt=torch.as_tensor(dt,device=values.device,dtype=values.dtype)
    if dt.ndim!=1 or len(dt)!=values.shape[1] or not torch.all(torch.isfinite(dt)&(dt>0)):
        raise ValueError('dt must contain positive finite intervals, one per step')
    if reset not in ('none','hard','soft'):raise ValueError('reset')
    return _run(values,dt,tau,gain,threshold,reset,v0,held=False)


def _run(values,dt,tau,gain,threshold,reset,v0,held):
    v=torch.zeros_like(values[:,0]) if v0 is None else v0.clone()
    pre=[];post=[];events=[]
    for t in range(values.shape[1]):
        b=torch.exp(-dt[t]/tau)
        u=b*v+gain*values[:,t]*((1-b) if held else 1.)
        s=(u>=threshold).to(values.dtype)
        v=u if reset=='none' else u*(1-s) if reset=='hard' else u-threshold*s
        pre.append(u);post.append(v);events.append(s)
    return {'pre':torch.stack(pre,1),'post':torch.stack(post,1),'events':torch.stack(events,1),
            'dt':dt,'input_semantics':'piecewise_constant_drive' if held else 'end_of_interval_event_amount',
            'spike_time_resolution':'interval-end checks only','neural_model_training':False}
