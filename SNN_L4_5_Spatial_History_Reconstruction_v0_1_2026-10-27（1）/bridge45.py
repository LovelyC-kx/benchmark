"""Unchanged spatial models from the user's L4.4 dependency tree."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'parent_l44'
if not PARENT.is_dir(): PARENT=ROOT/'parentl44'
sys.path.insert(0,str(PARENT))
from bridge44 import BenchmarkModel,ModelConfig,Cell,NEURONS,parameter_hash,save_tree,load_tree,tree_hash,write_json

def read_json(path):return json.loads(Path(path).read_text())
def config_key(backbone,neuron):return backbone+'__'+neuron
CONFIGS=tuple((b,n) for b in ('cnn2d','unet') for n in ('lif_hard','alif_ref','continuous'))
VIEWS=('last','mean')
def model_for(backbone,neuron):
    import torch
    torch.manual_seed(45001)
    cfg=ModelConfig(backbone=backbone,neuron=neuron,width=8,depth=2,beta=.85,gain=2.,threshold=.6,
        normalization='none',observation='post',readout='last')
    m=BenchmarkModel((5,16,16),(4,16,16),6,cfg,objective='reconstruction').eval()
    evidence=ROOT/'evidence' if (ROOT/'evidence/fitted/initializations').is_dir() else ROOT/'evidence1'
    snapshot=evidence/'fitted/initializations'/(config_key(backbone,neuron)+'.pt')
    if not snapshot.is_file(): raise FileNotFoundError('Required historical initialization missing; no silent random fallback: '+str(snapshot))
    m.load_state_dict(torch.load(snapshot,map_location='cpu',weights_only=True),strict=True)
    m.historical_checkpoint_status='untrained_initialization_not_benchmark_result'
    return m

def site_paths(model):
    sites=[n for n,m in model.named_modules() if isinstance(m,Cell)]
    return sites[0],sites[-1]

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
