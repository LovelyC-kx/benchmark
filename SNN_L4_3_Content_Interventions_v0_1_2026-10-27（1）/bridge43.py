"""Unchanged L4.2/L4.1 runtime and verified L3.7 actual-edge adapter."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'parent_l42'
if not PARENT.is_dir():
    if (ROOT/'bridge42.py').is_file(): PARENT=ROOT
    else:
        PARENT=next((base/'SNN_L4_2_Latent_Usability_v0_1_2026-10-27（1）' for base in ROOT.parents
                     if (base/'SNN_L4_2_Latent_Usability_v0_1_2026-10-27（1）'/'bridge42.py').is_file()),PARENT)
if not PARENT.is_dir(): raise FileNotFoundError('Missing archived L4.2 parent; use the canonical snnbench pipeline for new runs')
sys.path.insert(0,str(PARENT))
from bridge42 import (load_model,fixed_batch,parameter_hash,fit_head,save_tree,
    load_tree,tree_hash,write_json,independent_ids)
from bridge35 import Cell,StreamSession,BenchmarkModel,ModelConfig,NEURONS,BACKBONES,clone_tree
from bridge import _render
from capture35 import capture
if (ROOT/'parent_l41'/'communication_capture.py').is_file(): sys.path.append(str(ROOT/'parent_l41'))
from communication_capture import capture_edge
from readers42 import fit_mlp as _fit_mlp,predict,linear_bundle
SOURCES=('H1','H2','H3','H4','C')
RUN_DIRECTORY=PARENT/'parent_l41/evidence/source_smoke_run'

def fit_mlp(*args,**kwargs):
    bundle,record=_fit_mlp(*args,**kwargs)
    record['selection_objective']='Mean squared error on all 23 diagnostic targets: 20 continuous factor coordinates and 3 binary sign scores, with equal coordinate weight. No native SNN objective is optimized.'
    return bundle,record

def hybrid(cfg,recipient,donor,source,zero=False):
    """Only requested source content changes; all recipient measurement noise stays."""
    import numpy as np
    if source not in SOURCES:raise ValueError('Named physical source')
    j=SOURCES.index(source);out=dict(recipient);z=recipient['latents'].copy()
    z[:,j]=0 if zero else donor['latents'][:,j]
    out['latents']=z
    out['x']=_render(cfg,z,recipient['measurement_noise'],recipient['event_times'],recipient['source_present'])[:,:cfg.deadline_steps]
    if not np.array_equal(out['x'].shape,recipient['x'].shape):raise ValueError('Compatible recipient horizon')
    return out

def labels23(batch):
    import numpy as np
    z=batch['latents'].reshape(len(batch['x']),20).astype(float)
    return np.c_[z,np.where(z[:,[0,12,16]]>=0,1.,-1.)]

def baseline_sites(model,batch,times=(3,12,15)):
    r=capture(model,batch['x']);sites={}
    for layer in (0,1):
        for t in times:sites[f'l{layer}_t{t}']=r['carriers'][f'backbone.cells.{layer}/post'][:,t].astype(float)
    return sites,r['prediction']
