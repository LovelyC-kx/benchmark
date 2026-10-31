"""Causal reference architectures with explicit state, never official replicas.

S4D/LRU/Mamba equations motivate the named references; widths, input/output
adapters and training protocols remain ours. Official reproductions are tracked
separately in resources/model_sources.json. No fused/event-driven speed claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .models import MemoryBackbone,tree_map
from .reference.models import Backbone,ConvUnit,SEWBlock,norm

SPIKING_VECTOR=('dilated_spiking_tcn','causal_spike_attention_ref')
SPIKING_IMAGE=('sew_resnet34','ms_resnet18_ref','spiking_vgg11_ref','conv_lif_rnn')
CONTINUOUS_VECTOR=('rnn_tanh','dilated_tcn','transformer','conformer','s4d_ref','lru_ref','mamba_ref')
CONTINUOUS_IMAGE=('cnn_lstm','causal_conv3d','ann_unet','ann_resnet18_ref','convnext_ref','vit_ref')
SPATIAL_DECODERS=('causal_conv3d','ann_unet')


class ExplicitMemory(MemoryBackbone):
    """All declared state keys participate in the next step; no fake hidden state."""
    def __init__(self,cfg):
        nn.Module.__init__(self);self.cfg=cfg;self.kind=cfg.backbone;self.state={};self.last={};self.trace={};self.cache=[];self.capture=False
        self.preserve_initialization=set()
    def begin(self):
        if self.cfg.gradient_mode=='notd':self.state={}
        elif self.cfg.gradient_mode=='detach':self.state=tree_map(lambda x:x.detach(),self.state)
    def finish(self,output,**observations):
        self.last={'output':output,**observations}
        if self.capture:
            for k,v in self.last.items():self.trace.setdefault(k,[]).append(v)
        return output


class DilatedTCN(ExplicitMemory):
    def __init__(self,dim,cfg,spiking=False):
        super().__init__(cfg);w=cfg.width;self.embed=nn.Linear(dim,w);self.spiking=spiking
        self.taps=nn.ModuleList([nn.ModuleList([nn.Linear(w,w,bias=(j==0)) for j in range(3)]) for _ in range(cfg.depth)])
        self.norms=nn.ModuleList([nn.LayerNorm(w) for _ in range(cfg.depth)])
        self.cells=nn.ModuleList([Backbone.cell(self,w) for _ in range(cfg.depth)]) if spiking else None
    def forward(self,x):
        self.begin();h=self.embed(x.flatten(1))
        for i,taps in enumerate(self.taps):
            dilation=2**i;past=self.state.get('delay'+str(i),h[:,None,:0].reshape(len(h),0,h.shape[-1]))
            out=taps[0](h)
            for j in (1,2):
                if past.shape[1]>=j*dilation:out=out+taps[j](past[:,-j*dilation])
            self.state['delay'+str(i)]=torch.cat([past,h[:,None]],1)[:,-2*dilation:]
            out=self.norms[i](out+h);h=self.cells[i](out) if self.spiking else F.gelu(out)
        return self.finish(h)


class TemporalSpikeAttention(ExplicitMemory):
    """Causal binary Q/K/V prefix accumulation + recurrent cells, no softmax."""
    def __init__(self,dim,cfg):
        super().__init__(cfg);w=cfg.width;self.embed=nn.Linear(dim,w);self.qkv=nn.Linear(w,3*w)
        self.q=Backbone.cell(self,w);self.k=Backbone.cell(self,w);self.v=Backbone.cell(self,w)
        self.proj=nn.Linear(w,w);self.final=Backbone.cell(self,w)
    def forward(self,x):
        self.begin();e=self.embed(x.flatten(1));q,k,v=self.qkv(e).chunk(3,-1);q,k,v=self.q(q),self.k(k),self.v(v)
        outer=k[:,:,None]*v[:,None,:];kv=self.state.get('kv',torch.zeros_like(outer))+outer;self.state['kv']=kv
        h=self.final(self.proj(torch.einsum('bi,bij->bj',q,kv)/math.sqrt(self.cfg.width))+e)
        return self.finish(h)


class ResNet34(Backbone):
    def __init__(self,channels,cfg):
        super().__init__(cfg);self.stem=ConvUnit(self,channels,cfg.width);layers=[];inc=cfg.width
        for stage,n in enumerate((3,4,6,3)):
            out=cfg.width*2**stage
            for block in range(n):layers.append(SEWBlock(self,inc,out,2 if stage and block==0 else 1));inc=out
        self.blocks=nn.Sequential(*layers);self.proj=nn.Conv2d(inc,cfg.width,1);self.final=self.cell(cfg.width)
    def step(self,x):return self.final(self.proj(self.blocks(self.stem(x))))


class MembraneBlock(nn.Module):
    def __init__(self,owner,inc,out,stride):
        super().__init__();self.act1=owner.cell(inc);self.conv1=nn.Conv2d(inc,out,3,stride,padding=1)
        self.norm1=norm(out,owner.cfg);self.act2=owner.cell(out);self.conv2=nn.Conv2d(out,out,3,padding=1)
        self.skip=nn.Conv2d(inc,out,1,stride,bias=False) if inc!=out or stride!=1 else nn.Identity()
    def forward(self,x):return self.skip(x)+self.conv2(self.act2(self.norm1(self.conv1(self.act1(x)))))


class MSResNet18(Backbone):
    def __init__(self,channels,cfg):
        super().__init__(cfg);self.stem=nn.Conv2d(channels,cfg.width,3,padding=1);layers=[];inc=cfg.width
        for stage in range(4):
            out=cfg.width*2**stage
            for block in range(2):layers.append(MembraneBlock(self,inc,out,2 if stage and block==0 else 1));inc=out
        self.blocks=nn.Sequential(*layers);self.proj=nn.Conv2d(inc,cfg.width,1);self.final=self.cell(cfg.width)
    def step(self,x):return self.final(self.proj(self.blocks(self.stem(x))))


class SpikingVGG11(Backbone):
    def __init__(self,channels,cfg):
        super().__init__(cfg);layers=[];inc=channels
        for scale,count in zip((1,2,4,8,8),(1,1,2,2,2)):
            out=cfg.width*scale
            for _ in range(count):layers.append(ConvUnit(self,inc,out));inc=out
            layers.append(nn.AvgPool2d(2,ceil_mode=True))
        self.features=nn.Sequential(*layers);self.proj=nn.Conv2d(inc,cfg.width,1);self.final=self.cell(cfg.width)
    def step(self,x):return self.final(self.proj(self.features(x)))


class ConvLIFRNN(ExplicitMemory):
    def __init__(self,channels,cfg):
        super().__init__(cfg);w=cfg.width;self.input=nn.Conv2d(channels,w,3,padding=1);self.rec=nn.Conv2d(w,w,3,padding=1,bias=False)
        self.a=Backbone.cell(self,w);self.mix=nn.Conv2d(w,w,3,padding=1);self.b=Backbone.cell(self,w)
    def forward(self,x):
        self.begin();current=self.input(x);past=self.state.get('feedback',torch.zeros_like(current));h=self.b(self.mix(self.a(current+self.rec(past))));self.state['feedback']=h
        return self.finish(h)


class ElmanRNN(ExplicitMemory):
    def __init__(self,dim,cfg):
        super().__init__(cfg);self.layers=nn.ModuleList([nn.RNNCell(dim if i==0 else cfg.width,cfg.width,nonlinearity='tanh') for i in range(cfg.depth)])
    def forward(self,x):
        self.begin();h=x.flatten(1)
        for i,layer in enumerate(self.layers):h=layer(h,self.state.get('h'+str(i),h.new_zeros(len(h),self.cfg.width)));self.state['h'+str(i)]=h
        return self.finish(h,**self.state)


class CausalAttentionLayer(nn.Module):
    def __init__(self,w,convolution=False):
        super().__init__();self.heads=next(h for h in (8,4,2,1) if w%h==0);self.w=w;self.convolution=convolution
        self.n1=nn.LayerNorm(w);self.qkv=nn.Linear(w,3*w);self.proj=nn.Linear(w,w);self.n2=nn.LayerNorm(w)
        self.ff=nn.Sequential(nn.Linear(w,4*w),nn.GELU(),nn.Linear(4*w,w))
        if convolution:self.nc=nn.LayerNorm(w);self.conv=nn.Conv1d(w,w,5,groups=w);self.cm=nn.Linear(w,w)
    def step(self,x,state,prefix):
        b=len(x);q,k,v=self.qkv(self.n1(x)).chunk(3,-1);d=self.w//self.heads
        q,k,v=(z.reshape(b,self.heads,d) for z in (q,k,v));old=state.get(prefix+'keys')
        # Absolute sinusoidal position before Q/K projection is added by the owner.
        keys=k[:,:,None] if old is None else torch.cat([old,k[:,:,None]],2)
        vals=v[:,:,None] if old is None else torch.cat([state[prefix+'values'],v[:,:,None]],2)
        state[prefix+'keys']=keys;state[prefix+'values']=vals
        a=torch.softmax(torch.einsum('bhd,bhtd->bht',q,keys)/math.sqrt(d),-1)
        x=x+self.proj(torch.einsum('bht,bhtd->bhd',a,vals).reshape(b,self.w))
        if self.convolution:
            u=self.nc(x);past=state.get(prefix+'conv',u.new_zeros(b,self.w,4));window=torch.cat([past,u[:,:,None]],-1)
            state[prefix+'conv']=window[:,:,1:];x=x+self.cm(F.silu(self.conv(window).squeeze(-1)))
        return x+self.ff(self.n2(x))


class TransformerMemory(ExplicitMemory):
    def __init__(self,dim,cfg,convolution=False):
        super().__init__(cfg);self.embed=nn.Linear(dim,cfg.width);self.layers=nn.ModuleList([CausalAttentionLayer(cfg.width,convolution) for _ in range(cfg.depth)])
    def forward(self,x):
        self.begin();e=self.embed(x.flatten(1));t=self.position_t
        j=torch.arange(self.cfg.width,device=e.device,dtype=e.dtype);freq=torch.exp(-math.log(10000)*2*torch.floor(j/2)/self.cfg.width)
        position=torch.where((j.long()%2)==0,torch.sin(t*freq),torch.cos(t*freq));h=e+position
        for i,layer in enumerate(self.layers):h=layer.step(h,self.state,'l'+str(i)+'_')
        return self.finish(h)


class StructuredSSM(ExplicitMemory):
    """Complex diagonal recurrence stored as one real [real, imag] state."""
    def __init__(self,dim,cfg,lru=False):
        super().__init__(cfg);w=cfg.width;self.lru=lru;self.input=nn.Linear(dim,2*w);self.output=nn.Linear(2*w,w);self.skip=nn.Linear(dim,w)
        if lru:
            radius=torch.linspace(.8,.99,w);self.log_nu=nn.Parameter(torch.log(-torch.log(radius)));self.log_theta=nn.Parameter(torch.log(torch.linspace(.01,math.pi,w)))
            self.preserve_initialization={'log_nu','log_theta'}
        else:
            self.log_a=nn.Parameter(torch.log(torch.full((w,),.5)));self.omega=nn.Parameter(math.pi*torch.arange(w,dtype=torch.float32))
            self.log_dt=nn.Parameter(torch.linspace(math.log(.001),math.log(.1),w));self.preserve_initialization={'log_a','omega','log_dt'}
    def forward(self,x):
        self.begin();raw=x.flatten(1);u=self.input(raw);ur,ui=u.chunk(2,-1);previous=self.state.get('h',torch.zeros_like(u));hr,hi=previous.chunk(2,-1)
        if self.lru:
            radius=torch.exp(-torch.exp(self.log_nu));angle=torch.exp(self.log_theta);ar,ai=radius*angle.cos(),radius*angle.sin();scale=torch.sqrt((1-radius.square()).clamp_min(1e-8));br,bi=scale,torch.zeros_like(scale)
        else:
            delta=torch.exp(self.log_dt);real=-torch.exp(self.log_a);ar=torch.exp(delta*real)*torch.cos(delta*self.omega);ai=torch.exp(delta*real)*torch.sin(delta*self.omega)
            den=(real.square()+self.omega.square()).clamp_min(1e-12);br=((ar-1)*real+ai*self.omega)/den;bi=(ai*real-(ar-1)*self.omega)/den
        next_r=ar*hr-ai*hi+br*ur-bi*ui;next_i=ai*hr+ar*hi+bi*ur+br*ui
        self.state['h']=torch.cat([next_r,next_i],-1);out=self.output(self.state['h'])+self.skip(raw)
        return self.finish(out,h=self.state['h'])


class SelectiveSSM(ExplicitMemory):
    """Dense Mamba-1-style selective recurrence, no official fused kernel claim."""
    def __init__(self,dim,cfg):
        super().__init__(cfg);w=cfg.width;self.inner=2*w;self.n=8;self.input=nn.Linear(dim,2*self.inner)
        self.conv=nn.Conv1d(self.inner,self.inner,4,groups=self.inner);self.parameters_in=nn.Linear(self.inner,self.inner+2*self.n)
        self.log_a=nn.Parameter(torch.log(torch.arange(1,self.n+1,dtype=torch.float32)).repeat(self.inner,1));self.D=nn.Parameter(torch.ones(self.inner));self.output=nn.Linear(self.inner,w)
        self.preserve_initialization={'log_a','D'}
    def forward(self,x):
        self.begin();u,z=self.input(x.flatten(1)).chunk(2,-1);past=self.state.get('conv',u.new_zeros(len(u),self.inner,3));window=torch.cat([past,u[:,:,None]],-1);self.state['conv']=window[:,:,1:]
        u=F.silu(self.conv(window).squeeze(-1));delta,B,C=torch.split(self.parameters_in(u),[self.inner,self.n,self.n],-1);delta=F.softplus(delta).clamp_max(10)
        h=self.state.get('h',u.new_zeros(len(u),self.inner,self.n));transition=torch.exp(-delta[:,:,None]*torch.exp(self.log_a)[None])
        h=transition*h+delta[:,:,None]*B[:,None,:]*u[:,:,None];self.state['h']=h
        out=self.output(((h*C[:,None,:]).sum(-1)+self.D*u)*F.silu(z));return self.finish(out,h=h)


class CNNLSTM(ExplicitMemory):
    def __init__(self,channels,cfg):
        super().__init__(cfg);w=cfg.width;self.encoder=nn.Sequential(nn.Conv2d(channels,w,3,padding=1),nn.GELU(),nn.Conv2d(w,w,3,padding=1),nn.GELU())
        self.layers=nn.ModuleList([nn.LSTMCell(w,w) for _ in range(cfg.depth)])
    def forward(self,x):
        self.begin();h=self.encoder(x).mean((-2,-1));zero=h.new_zeros(h.shape)
        for i,layer in enumerate(self.layers):
            h,c=layer(h,(self.state.get('h'+str(i),zero),self.state.get('c'+str(i),zero)));self.state.update({'h'+str(i):h,'c'+str(i):c})
        return self.finish(h,**self.state)


class CausalConv3D(ExplicitMemory):
    def __init__(self,channels,cfg):
        super().__init__(cfg);self.conv=nn.Conv3d(channels,cfg.width,(3,3,3),padding=(0,1,1));self.mix=nn.Conv2d(cfg.width,cfg.width,3,padding=1)
    def forward(self,x):
        self.begin();past=self.state.get('frames',x.new_zeros(len(x),x.shape[1],2,*x.shape[-2:]));window=torch.cat([past,x[:,:,None]],2);self.state['frames']=window[:,:,1:]
        return self.finish(F.gelu(self.mix(F.gelu(self.conv(window).squeeze(2)))))


class ANNUNet(ExplicitMemory):
    def __init__(self,channels,cfg):
        super().__init__(cfg);w=cfg.width;self.e1=nn.Conv2d(channels,w,3,padding=1);self.e2=nn.Conv2d(w,2*w,3,stride=2,padding=1);self.b=nn.Conv2d(2*w,2*w,3,padding=1);self.d=nn.Conv2d(3*w,w,3,padding=1)
    def forward(self,x):
        self.begin();e=F.gelu(self.e1(x));b=F.gelu(self.b(F.gelu(self.e2(e))));up=F.interpolate(b,e.shape[-2:],mode='nearest');return self.finish(F.gelu(self.d(torch.cat([e,up],1))))


class ANNBlock(nn.Module):
    def __init__(self,inc,out,stride):
        super().__init__();self.c1=nn.Conv2d(inc,out,3,stride,padding=1);self.c2=nn.Conv2d(out,out,3,padding=1);self.n1=nn.GroupNorm(1,out);self.n2=nn.GroupNorm(1,out);self.skip=nn.Conv2d(inc,out,1,stride) if inc!=out or stride!=1 else nn.Identity()
    def forward(self,x):return F.relu(self.n2(self.c2(F.relu(self.n1(self.c1(x)))))+self.skip(x))


class ANNResNet18(ExplicitMemory):
    def __init__(self,channels,cfg):
        super().__init__(cfg);self.stem=nn.Conv2d(channels,cfg.width,3,padding=1);layers=[];inc=cfg.width
        for stage in range(4):
            out=cfg.width*2**stage
            for block in range(2):layers.append(ANNBlock(inc,out,2 if stage and block==0 else 1));inc=out
        self.blocks=nn.Sequential(*layers);self.proj=nn.Conv2d(inc,cfg.width,1)
    def forward(self,x):self.begin();return self.finish(self.proj(self.blocks(self.stem(x))))


class ConvNeXtBlock(nn.Module):
    def __init__(self,w):
        super().__init__();self.depthwise=nn.Conv2d(w,w,7,padding=3,groups=w);self.norm=nn.LayerNorm(w);self.expand=nn.Linear(w,4*w);self.reduce=nn.Linear(4*w,w)
    def forward(self,x):
        h=self.depthwise(x).permute(0,2,3,1);h=self.reduce(F.gelu(self.expand(self.norm(h)))).permute(0,3,1,2);return x+h


class ConvNeXt(ExplicitMemory):
    def __init__(self,channels,cfg):
        super().__init__(cfg);self.stem=nn.Conv2d(channels,cfg.width,3,padding=1);layers=[];inc=cfg.width
        for stage,count in enumerate((3,3,9,3)):
            out=cfg.width*2**stage
            if stage:layers.append(nn.Conv2d(inc,out,3,stride=2,padding=1))
            layers.extend(ConvNeXtBlock(out) for _ in range(count));inc=out
        self.blocks=nn.Sequential(*layers);self.proj=nn.Conv2d(inc,cfg.width,1)
    def forward(self,x):self.begin();return self.finish(self.proj(self.blocks(self.stem(x))))


class SpatialViT(ExplicitMemory):
    def __init__(self,channels,cfg):
        super().__init__(cfg);w=cfg.width;heads=next(h for h in (8,4,2,1) if w%h==0);self.patch=nn.Conv2d(channels,w,4,stride=4)
        self.blocks=nn.TransformerEncoder(nn.TransformerEncoderLayer(w,heads,4*w,dropout=0.,batch_first=True,norm_first=True),cfg.depth,enable_nested_tensor=False)
    def forward(self,x):
        self.begin();tokens=self.patch(x).flatten(2).transpose(1,2);p=torch.arange(tokens.shape[1],device=x.device,dtype=x.dtype)[:,None];j=torch.arange(self.cfg.width,device=x.device,dtype=x.dtype)[None];f=torch.exp(-math.log(10000)*2*torch.floor(j/2)/self.cfg.width)
        pos=torch.where(j.long()%2==0,torch.sin(p*f),torch.cos(p*f));return self.finish(self.blocks(tokens+pos).mean(1))


def build_extended(shape,cfg):
    name=cfg.backbone;dim=math.prod(shape);channels=shape[0]
    if name=='dilated_spiking_tcn':return DilatedTCN(dim,cfg,True)
    if name=='causal_spike_attention_ref':return TemporalSpikeAttention(dim,cfg)
    if name=='dilated_tcn':return DilatedTCN(dim,cfg)
    if name=='rnn_tanh':return ElmanRNN(dim,cfg)
    if name in ('transformer','conformer'):return TransformerMemory(dim,cfg,name=='conformer')
    if name in ('s4d_ref','lru_ref'):return StructuredSSM(dim,cfg,name=='lru_ref')
    if name=='mamba_ref':return SelectiveSSM(dim,cfg)
    table={'sew_resnet34':ResNet34,'ms_resnet18_ref':MSResNet18,'spiking_vgg11_ref':SpikingVGG11,'conv_lif_rnn':ConvLIFRNN,
        'cnn_lstm':CNNLSTM,'causal_conv3d':CausalConv3D,'ann_unet':ANNUNet,'ann_resnet18_ref':ANNResNet18,'convnext_ref':ConvNeXt,'vit_ref':SpatialViT}
    if name not in table:raise ValueError('Unknown extended architecture: '+name)
    if len(shape)!=3:raise ValueError('Image architecture requires C,H,W')
    return table[name](channels,cfg)
