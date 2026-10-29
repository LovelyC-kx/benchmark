"""Explicit independent variants; NOT official reproductions of named papers."""
import math
import torch
from torch import nn

NEURONS=('lif_hard','lif_soft','threshold','if_hard','plif_ref','alif_ref',
         'adlif_ref','hetero_lif','two_state_ref','continuous','tanh','analog_hard_ref')

class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kind, scale):
        ctx.save_for_backward(x);ctx.kind=kind;ctx.scale=scale
        return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad):
        x,=ctx.saved_tensors;s=ctx.scale
        if ctx.kind=='triangle': g=(1-(x*s).abs()).clamp(min=0)*s
        elif ctx.kind=='atan': g=s/(1+(math.pi*s*x).square())
        elif ctx.kind=='sigmoid': p=torch.sigmoid(s*x);g=s*p*(1-p)
        elif ctx.kind=='rectangle': g=(x.abs()<=1/s).to(x.dtype)*s/2
        else: raise ValueError(ctx.kind)
        return grad*g,None,None

class Cell(nn.Module):
    def __init__(self, channels, kind='lif_hard', beta=.8, threshold=1., gain=2.,
                 dc_normalize=False, mode='full', surrogate='atan', sg_scale=2.,
                 detach_reset=True, adapt_decay=.9, adapt_gain=.1, reset_strength=1.):
        super().__init__()
        if kind not in NEURONS: raise ValueError(kind)
        if mode not in ('full','detach','notd'): raise ValueError(mode)
        if not 0<=beta<=1 or threshold<=0: raise ValueError('beta/threshold')
        if dc_normalize and (kind=='if_hard' or beta==1):
            raise ValueError('dc_normalize requires beta<1; IF has no finite DC gain')
        if not 0<=reset_strength<=1: raise ValueError('reset_strength must be in [0,1]')
        if not 0<=adapt_decay<1 or adapt_gain<0: raise ValueError('adaptation parameters')
        self.kind=kind;self.mode=mode;self.threshold=threshold;self.gain=gain
        self.dc_normalize=dc_normalize;self.surrogate=surrogate
        self.sg_scale=sg_scale;self.detach_reset=detach_reset
        self.adapt_decay=adapt_decay;self.adapt_gain=adapt_gain
        self.reset_strength=reset_strength
        self.event_controls={}
        b=torch.full((channels,), beta) if kind=='hetero_lif' else torch.tensor(beta)
        if kind=='hetero_lif': b=torch.linspace(max(.01,beta-.2),min(.99,beta+.1),channels)
        if kind=='if_hard':b=torch.tensor(1.)
        if kind in ('plif_ref','hetero_lif'):
            self.raw_beta=nn.Parameter(torch.logit(b.clamp(.0001,.9999)))
        else:self.register_buffer('fixed_beta',b)
        self.state={};self.trace={};self.capture=False;self.last={}

    def beta_value(self,x):
        b=torch.sigmoid(self.raw_beta) if hasattr(self,'raw_beta') else self.fixed_beta
        return b if b.ndim==0 else b.view(1,-1,*([1]*(x.ndim-2)))

    def reset(self): self.state={};self.trace={};self.last={};self.event_controls={}

    def force_events(self, **controls):
        """Diagnostic branch override; retained until cleared/reset. Not natural input."""
        if set(controls)-{'emit','reset','aux'}:raise ValueError('event branch')
        if self.kind in ('continuous','tanh') and controls:
            raise ValueError('event overrides require an event-based cell')
        for value in controls.values():
            if not isinstance(value,torch.Tensor) or not torch.all((value==0)|(value==1)):
                raise ValueError('forced events must be binary tensors')
        self.event_controls=dict(controls)

    def _branch(self,name,natural):
        value=self.event_controls.get(name,natural)
        if value.shape!=natural.shape or value.device!=natural.device or value.dtype!=natural.dtype:
            raise ValueError('forced event shape/device/dtype mismatch')
        return value

    def forward(self, x):
        if self.mode=='notd': self.state={}
        if self.mode=='detach': self.state={k:v.detach() for k,v in self.state.items()}
        v=self.state.get('v',torch.zeros_like(x));b=self.beta_value(x)
        I=self.gain*x*((1-b) if self.dc_normalize else 1.)
        base_current=I
        a=self.state.get('a',torch.zeros_like(x))
        old_syn=self.state.get('syn',torch.zeros_like(x))
        if self.kind=='two_state_ref':
            syn=self.adapt_decay*old_syn+I
            self.state['syn']=syn;I=(1-self.adapt_decay)*syn
        if self.kind=='adlif_ref': I=I-a
        pre=b*v+I
        theta=self.threshold+(a if self.kind=='alif_ref' else 0.)
        if self.kind=='continuous':out=pre;post=pre
        elif self.kind=='tanh':out=torch.tanh(pre);post=out
        else:
            event=Spike.apply(pre-theta,self.surrogate,self.sg_scale)
            emitted=self._branch('emit',event)
            reset_event=self._branch('reset',event)
            aux_event=self._branch('aux',event)
            out=pre if self.kind=='analog_hard_ref' else emitted
            if self.kind=='analog_hard_ref' and 'emit' in self.event_controls:out=emitted
            r=reset_event.detach() if self.detach_reset else reset_event
            if self.kind=='threshold':post=pre
            elif self.kind=='lif_soft':post=pre-self.reset_strength*self.threshold*r
            else:post=pre*(1-self.reset_strength*r)
        self.state['v']=post
        if self.kind=='alif_ref': self.state['a']=self.adapt_decay*a+self.adapt_gain*aux_event
        if self.kind=='adlif_ref': self.state['a']=self.adapt_decay*a+self.adapt_gain*pre+.1*aux_event
        self.last={'pre':pre,'post':post,'output':out,'margin':pre-theta,
                   'base_current':base_current,'effective_current':I,'previous_v':v,'theta':torch.zeros_like(pre)+theta}
        if self.kind not in ('continuous','tanh'):
            self.last.update(natural_event=event,emit_event=emitted,reset_event=reset_event,aux_event=aux_event)
        if self.kind in ('alif_ref','adlif_ref'):self.last.update(previous_a=a,a=self.state['a'])
        if self.kind=='two_state_ref':self.last.update(previous_syn=old_syn,syn=self.state['syn'])
        if self.capture:
            for k,val in self.last.items():self.trace.setdefault(k,[]).append(val)
        return out

    def state_scalars(self): return sum(v.numel() for v in self.state.values())

def hard_kernel(inputs, spikes, beta):
    """Exact realized-path post-reset decomposition, scalar fixed beta, zero state."""
    T=inputs.shape[1]; terms=[]
    for k in range(T):
        survival=(1-spikes[:,k:]).prod(1)
        terms.append(inputs[:,k]*(beta**(T-1-k))*survival)
    return torch.stack(terms,1)
