"""Optional unmodified neuroseqbench integration, separate from diagnostic reference cells.
This is the benchmark maintainers' implementation, not a paper-score reproduction.
"""
from pathlib import Path
import sys,inspect,subprocess,hashlib
import torch
from torch import nn

COMMIT='af3c3204df0c3bedc9abfde3a01cc62f42effde5'
FAMILIES=('LIF','PLIF','ALIF','adLIF','GLIF','CLIF','TCLIF','LTC','PMSN','LMH','CELIF','DHSNN','SPSN')

def configure_upstream(root):
    p=Path(root).resolve()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
    if head!=COMMIT:raise ValueError(f'upstream commit mismatch: expected {COMMIT}, got {head}')
    diff=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=p,text=True).strip()
    if diff:raise ValueError('upstream source modified; classify patched implementation separately')
    sys.path.insert(0,str(p/'src'))
    return {'repository':'https://github.com/liyc5929/neuroseqbench','commit':head,
            'license':'GPL-3.0 source; external checkout not redistributed in this archive',
            'source_modified':False}

class UpstreamSequenceModel(nn.Module):
    def __init__(self,root,family,input_dim=8,width=16,T=12,output_dim=2,
                 objective='classification',readout='mean',prop_mode='STBP'):
        super().__init__();self.provenance=configure_upstream(root);self.family=family
        from neuroseqbench.network import neuron
        from neuroseqbench.network.trainer import SurrogateGradient
        cls=getattr(neuron,family);sig=inspect.signature(cls)
        possible={'neuron_num':width,'input_features':width,'time_step':T,
                  'surro_grad':SurrogateGradient('triangle',a=1.),'exec_mode':'serial',
                  'recurrent':False,'threshold':.3,'decay':.8,'prop_mode':prop_mode}
        kwargs={k:v for k,v in possible.items() if k in sig.parameters}
        if prop_mode!='STBP' and family!='LIF':raise ValueError('only native LIF exposes this propagation-mode control')
        self.embed=nn.Linear(input_dim,width);self.cell=cls(**kwargs);self.head=nn.Linear(width,output_dim)
        self.T=T;self.objective=objective;self.readout=readout;self.kwargs={k:str(v) for k,v in kwargs.items()}
        self.adapter_assumptions=[]
        if family=='CELIF':
            # Upstream neuron explicitly requires its architecture to supply TE.
            self.cell.TE=nn.Parameter(torch.zeros(width,T))
            self.adapter_assumptions.append('architecture supplies zero-initialized trainable TE; not original training configuration')
        self.output_trace=None

    def forward(self,x):
        tx=self.embed(x).transpose(0,1).contiguous()
        out=self.cell(tx)
        if isinstance(out,tuple):out=out[0]
        self.output_trace=out.transpose(0,1)
        a=self.output_trace[:,-1] if self.readout=='last' else self.output_trace.mean(1)
        return self.head(a)

    def native_terminal_state(self,x):
        if not hasattr(self.cell,'return_mem'):raise ValueError('upstream does not expose return_mem')
        self.cell.return_mem=True
        try:return self.cell(self.embed(x).transpose(0,1).contiguous())
        finally:self.cell.return_mem=False

    def metadata(self):
        module=inspect.getfile(type(self.cell));sha=hashlib.sha256(Path(module).read_bytes()).hexdigest()
        return {**self.provenance,'family':self.family,'upstream_constructor_kwargs':self.kwargs,
                'source_file_sha256':sha,'adapter_assumptions':self.adapter_assumptions,
                'output_trace_available':True,'all_internal_states_instrumented':False,
                'terminal_state_semantics':'upstream return_mem; pre/post varies by family and must be separately audited',
                'native_global_time_BN_possible':self.family in ('adLIF','PMSN'),
                'original_paper_score_reproduced':False}
