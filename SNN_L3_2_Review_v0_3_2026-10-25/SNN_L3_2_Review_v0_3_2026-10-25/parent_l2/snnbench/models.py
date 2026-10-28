"""Eight backbone families with shared cell injection and explicit state scope."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from .neurons import Cell

BACKBONES=('mlp','recurrent_mlp','temporal_conv','cnn1d','cnn2d',
           'sew_resnet18','spike_attention','unet')

@dataclass
class ModelConfig:
    backbone:str='mlp'
    neuron:str='lif_hard'
    width:int=32
    depth:int=2
    beta:float=.8
    threshold:float=1.
    gain:float=2.
    dc_normalize:bool=False
    gradient_mode:str='full'
    surrogate:str='atan'
    sg_scale:float=2.
    detach_reset:bool=True
    observation:str='output'
    readout:str='mean'
    readout_decay:float=.8
    normalization:str='none'
    push_pull:bool=False
    reset_strength:float=1.
    adapt_decay:float=.9
    adapt_gain:float=.1

def norm(ch, cfg):
    if cfg.normalization=='none':return nn.Identity()
    if cfg.normalization=='group':return nn.GroupNorm(1,ch)
    raise ValueError('only none/group (sample-local) are implemented')

class Backbone(nn.Module):
    def __init__(self,cfg):super().__init__();self.cfg=cfg;self.last_cell=None;self.cache=[]
    def cell(self,ch):
        c=Cell(ch,kind=self.cfg.neuron,beta=self.cfg.beta,threshold=self.cfg.threshold,
               gain=self.cfg.gain,dc_normalize=self.cfg.dc_normalize,
               mode=self.cfg.gradient_mode,surrogate=self.cfg.surrogate,
               sg_scale=self.cfg.sg_scale,detach_reset=self.cfg.detach_reset,
               reset_strength=self.cfg.reset_strength,adapt_decay=self.cfg.adapt_decay,adapt_gain=self.cfg.adapt_gain)
        # Non-owning alias: avoid duplicate module/state_dict registration.
        self.__dict__['last_cell']=c;return c
    def reset_cache(self):self.cache=[]
    def cache_scalars(self):return sum(x.numel() for x in self.cache)

class MLP(Backbone):
    def __init__(self,dim,cfg):
        super().__init__(cfg);self.layers=nn.ModuleList();self.cells=nn.ModuleList()
        for k in range(cfg.depth):
            self.layers.append(nn.Linear(dim if k==0 else cfg.width,cfg.width))
            self.cells.append(self.cell(cfg.width))
        self.rec=nn.Linear(cfg.width,cfg.width,bias=False) if cfg.backbone=='recurrent_mlp' else None
    def step(self,x):
        x=x.flatten(1)
        for k,(lin,c) in enumerate(zip(self.layers,self.cells)):
            I=lin(x)
            if k==0 and self.rec is not None and self.cache and self.cfg.gradient_mode!='notd':
                past=self.cache[0].detach() if self.cfg.gradient_mode=='detach' else self.cache[0]
                I=I+self.rec(past)
            x=c(I)
        self.cache=[x] if self.rec is not None else []
        return x

class TemporalConv(Backbone):
    """Causal 3-tap feature convolution; cache detach/notd includes explicit delays."""
    def __init__(self,dim,cfg):
        super().__init__(cfg);self.embed=nn.Linear(dim,cfg.width)
        self.taps=nn.ModuleList([nn.Linear(cfg.width,cfg.width,bias=False) for _ in range(3)])
        self.act=self.cell(cfg.width)
    def step(self,x):
        e=self.embed(x.flatten(1))
        if self.cfg.gradient_mode=='notd':self.cache=[]
        if self.cfg.gradient_mode=='detach':self.cache=[p.detach() for p in self.cache]
        I=self.taps[0](e)
        for k,p in enumerate(reversed(self.cache),1):I=I+self.taps[k](p)
        self.cache=(self.cache+[e])[-2:]
        return self.act(I)

class CNN1D(Backbone):
    """Per-step spatial/sensor-axis convolution; time recursion is in cells."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.conv1=nn.Conv1d(1,cfg.width,3,padding=1)
        self.conv2=nn.Conv1d(cfg.width,cfg.width,3,padding=1)
        self.a1=self.cell(cfg.width);self.a2=self.cell(cfg.width)
    def step(self,x):return self.a2(self.conv2(self.a1(self.conv1(x.flatten(1).unsqueeze(1)))))

class ConvUnit(nn.Module):
    def __init__(self, owner,inc,outc,stride=1):
        super().__init__();self.conv=nn.Conv2d(inc,outc,3,stride,padding=1)
        self.norm=norm(outc,owner.cfg);self.act=owner.cell(outc)
    def forward(self,x):return self.act(self.norm(self.conv(x)))

class CNN2D(Backbone):
    def __init__(self,channels,cfg):
        super().__init__(cfg);self.a=ConvUnit(self,channels,cfg.width)
        self.b=ConvUnit(self,cfg.width,cfg.width)
    def step(self,x):return self.b(self.a(x))

class SEWBlock(nn.Module):
    def __init__(self,owner,inc,outc,stride=1):
        super().__init__();self.a=ConvUnit(owner,inc,outc,stride);self.b=ConvUnit(owner,outc,outc)
        self.skip=ConvUnit(owner,inc,outc,stride) if inc!=outc or stride!=1 else nn.Identity()
    def forward(self,x):return self.b(self.a(x))+self.skip(x)

class SEWResNet(Backbone):
    """ResNet18 2/2/2/2 stage skeleton, SEW ADD; independent small-width variant."""
    def __init__(self,channels,cfg):
        super().__init__(cfg);self.stem=ConvUnit(self,channels,cfg.width)
        blocks=[];inc=cfg.width
        for stage in range(4):
            out=cfg.width*2**stage
            blocks.append(SEWBlock(self,inc,out,1 if stage==0 else 2))
            blocks.append(SEWBlock(self,out,out));inc=out
        self.blocks=nn.Sequential(*blocks);self.proj=nn.Conv2d(inc,cfg.width,1)
        self.final=self.cell(cfg.width)
    def step(self,x):return self.final(self.proj(self.blocks(self.stem(x))))

class SpikeAttention(Backbone):
    """Per-frame binary Q/K/V spatial attention. Dense tensor implementation."""
    def __init__(self,channels,cfg):
        super().__init__(cfg);w=cfg.width
        self.embed=ConvUnit(self,channels,w,2)
        self.qkv=nn.Conv2d(w,3*w,1);self.q=self.cell(w);self.k=self.cell(w);self.v=self.cell(w)
        self.proj=nn.Conv2d(w,w,1);self.final=self.cell(w)
    def step(self,x):
        e=self.embed(x);q,k,v=self.qkv(e).chunk(3,1)
        q=self.q(q).flatten(2).transpose(1,2)
        k=self.k(k).flatten(2).transpose(1,2)
        v=self.v(v).flatten(2).transpose(1,2)
        a=q@(k.transpose(1,2)@v)/max(1,q.shape[1])
        a=a.transpose(1,2).reshape_as(e)
        return self.final(self.proj(a)+e)

class UNet(Backbone):
    """Two-scale causal-in-time U-Net, stateful cells at every block."""
    def __init__(self,channels,cfg):
        super().__init__(cfg);w=cfg.width
        self.e1=ConvUnit(self,channels,w);self.e2=ConvUnit(self,w,2*w,2)
        self.b=ConvUnit(self,2*w,2*w)
        self.up=nn.ConvTranspose2d(2*w,w,2,stride=2)
        self.d=ConvUnit(self,2*w,w)
    def step(self,x):
        e=self.e1(x);b=self.b(self.e2(e));u=self.up(b)
        if u.shape[-2:]!=e.shape[-2:]:u=nn.functional.interpolate(u,e.shape[-2:])
        return self.d(torch.cat([u,e],1))

class BenchmarkModel(nn.Module):
    def __init__(self,input_shape,output_shape,T,cfg:ModelConfig,objective='classification'):
        super().__init__();self.cfg=cfg;self.objective=objective;self.T=T
        if cfg.backbone not in BACKBONES or cfg.depth<1 or cfg.width<1:raise ValueError('backbone/size')
        if cfg.normalization!='none' and cfg.backbone in ('mlp','recurrent_mlp','temporal_conv','cnn1d'):
            raise ValueError('group normalization is implemented only for spatial image backbones')
        if cfg.observation not in ('output','pre','post'):raise ValueError('observation')
        if cfg.readout not in ('last','mean','sum','exponential','trajectory'):raise ValueError('readout')
        shape=list(input_shape)
        if cfg.push_pull:shape[0]*=2
        dim=1
        for s in shape:dim*=s
        if cfg.backbone in ('mlp','recurrent_mlp'):self.backbone=MLP(dim,cfg)
        elif cfg.backbone=='temporal_conv':self.backbone=TemporalConv(dim,cfg)
        elif cfg.backbone=='cnn1d':self.backbone=CNN1D(cfg)
        else:
            if len(shape)!=3:raise ValueError('image backbones require C,H,W input')
            cls={'cnn2d':CNN2D,'sew_resnet18':SEWResNet,'spike_attention':SpikeAttention,'unet':UNet}[cfg.backbone]
            self.backbone=cls(shape[0],cfg)
        if objective=='reconstruction' and len(output_shape)==3:
            if cfg.backbone not in ('cnn2d','unet'):raise ValueError('spatial reconstruction uses cnn2d/unet')
            if cfg.readout=='trajectory':raise ValueError('trajectory image decoder not defined; use mean/last or seq2seq')
            self.head=nn.Conv2d(cfg.width,output_shape[0],1);self.image_output=True
        else:
            out=output_shape[0];self.image_output=False
            self.head=nn.Linear(cfg.width*(T if cfg.readout=='trajectory' else 1),out)
        self.last_features=None

    def reset(self,capture=False):
        self.backbone.reset_cache()
        for m in self.modules():
            if isinstance(m,Cell):m.reset();m.capture=capture

    def forward(self,x,capture=False,return_sequence=False):
        if x.ndim not in (3,5):raise ValueError('input must B,T,F or B,T,C,H,W')
        self.reset(capture);fs=[]
        for t in range(x.shape[1]):
            xt=x[:,t]
            if self.cfg.push_pull:xt=torch.cat([xt.relu(),(-xt).relu()],dim=1)
            f=self.backbone.step(xt)
            if self.cfg.observation!='output':f=self.backbone.last_cell.last[self.cfg.observation]
            if not self.image_output and f.ndim>2:f=f.flatten(2).mean(-1)
            fs.append(f)
        features=torch.stack(fs,1);self.last_features=features
        if self.cfg.readout=='trajectory':
            if x.shape[1]!=self.T:raise ValueError('trajectory head is fixed-T; condition not compatible')
            pred=self.head(features.flatten(1))
        else:
            if self.cfg.readout=='last':a=features[:,-1]
            elif self.cfg.readout=='mean':a=features.mean(1)
            elif self.cfg.readout=='sum':a=features.sum(1)
            else:
                w=self.cfg.readout_decay**torch.arange(x.shape[1]-1,-1,-1,device=x.device,dtype=x.dtype)
                a=(features*w.view(1,-1,*([1]*(features.ndim-2)))).sum(1)/w.sum()
            pred=self.head(a)
        if return_sequence:
            if self.cfg.readout=='trajectory':raise ValueError('trajectory head has no per-step output')
            if self.image_output:
                seq=self.head(features.flatten(0,1)).reshape(x.shape[0],x.shape[1],*pred.shape[1:])
            else:seq=self.head(features)
            return pred,seq
        return pred

    def traces(self):
        return {name:{k:torch.stack(v,1) for k,v in m.trace.items()}
                for name,m in self.named_modules() if isinstance(m,Cell) and m.trace}

    def resources(self):
        state=sum(m.state_scalars() for m in self.modules() if isinstance(m,Cell))
        return {'parameters':sum(p.numel() for p in self.parameters()),
                'cell_state_scalars_batch':state,'explicit_cache_scalars_batch':self.backbone.cache_scalars(),
                'config':asdict(self.cfg),
                'implementation':'dense PyTorch; no asynchronous hardware claim',
                'real_value_interfaces':['input current','membrane','classifier/regression head'],
                'reference_variant_not_official_reproduction':True}
