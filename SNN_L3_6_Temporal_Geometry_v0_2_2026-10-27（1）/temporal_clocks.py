"""Past-only phase diagnostics and explicitly scoped rhythmic state updates."""
import numpy as np

def causal_complex_signal(x,frequency,decay=8.,support=24):
    x=np.asarray(x,float)
    if x.ndim!=2 or not 0<frequency<.5 or decay<=0 or support<1 or not np.isfinite(x).all():
        raise ValueError('Finite scene x time, frequency in cycles/step')
    lag=np.arange(support);kernel=np.exp(-lag/decay+2j*np.pi*frequency*lag)
    kernel/=np.linalg.norm(kernel)
    return np.stack([np.convolve(row,kernel,'full')[:len(row)] for row in x])

def rhythmic_step(v,a,current,mask,beta=.8,rho=.9,threshold=1.,aux_clock='all'):
    if aux_clock not in ('all','membrane_only'):raise ValueError('Auxiliary clock policy')
    v,a,i,m=map(np.asarray,(v,a,current,mask))
    if not np.isin(m,[0,1]).all():raise ValueError('Binary update mask')
    pre=beta*v+i-a;event=(pre>=threshold)*m
    post=np.where(m>0,pre*(1-event),v)
    new_a=rho*a+.1*event
    if aux_clock=='all':new_a=np.where(m>0,new_a,a)
    return post,new_a,event

def finite_prefix_phase(x,frequency,minimum_amplitude=1e-8):
    signal=causal_complex_signal(x,frequency)
    return {'phase':np.angle(signal),'amplitude':np.abs(signal),
            'valid':np.abs(signal)>minimum_amplitude,
            'scope':'Past-only complex filter; fixed frequency, no future-defined band selection'}
